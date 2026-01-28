import logging
import os
from datetime import datetime
from typing import Any

import hydra
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from config.path import ABS_CONFIG_DIR
from src.data.module.train import TrainDataModule
from src.model.module.train import SPLADETrainingModule
from src.utils import log_if_rank_zero, set_seed
from src.utils.logging import (
    get_logger,
    suppress_accumulate_grad_stream_mismatch_warning,
    setup_tqdm_friendly_logging,
)
from src.utils.script_setup import configure_script_environment
from src.utils.trainer import (
    get_cpu_trainer_kwargs,
    get_gpu_trainer_kwargs,
    resolve_precision,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=True,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


def _maybe_mark_ddp_launcher(
    training_cfg: DictConfig, trainer_kwargs: dict[str, Any]
) -> None:
    """Flag the Lightning DDP launcher to silence duplicate logs."""
    strategy_name: str = str(getattr(training_cfg, "strategy", "auto")).lower()
    if strategy_name != "ddp":
        return
    rank_env: str | None = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank_env is not None:
        return
    devices: Any = trainer_kwargs.get("devices", 1)
    # Devices can be an int (count) or an explicit device list.
    if isinstance(devices, (list, tuple)):
        device_count: int = len(devices)
    else:
        device_count = int(devices)
    if device_count <= 1:
        return
    os.environ["SPLADE_DDP_LAUNCHER"] = "1"


def _resolve_checkpoint_dir(log_dir: str) -> str:
    """Return a checkpoint directory, suffixing when checkpoints already exist."""
    base_dir: str = os.path.join(log_dir, "checkpoints")
    # Only fork when previous checkpoints exist to avoid overwriting.
    if not os.path.isdir(base_dir):
        return base_dir
    entries: list[str] = os.listdir(base_dir)
    has_checkpoints: bool = any(entry.endswith(".ckpt") for entry in entries)
    if not has_checkpoints:
        return base_dir
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{base_dir}_{timestamp}"


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    setup_tqdm_friendly_logging()
    suppress_accumulate_grad_stream_mismatch_warning()
    os.makedirs(cfg.log_dir, exist_ok=True)

    set_seed(cfg.seed)
    log_if_rank_zero(logger, f"Random seed set to: {cfg.seed}")

    training_cfg: DictConfig = cfg.training
    trainer_kwargs: dict[str, Any] = (
        get_cpu_trainer_kwargs(training_cfg)
        if training_cfg.use_cpu
        else get_gpu_trainer_kwargs(training_cfg)
    )
    _maybe_mark_ddp_launcher(training_cfg, trainer_kwargs)
    precision: str = resolve_precision(training_cfg)

    model: SPLADETrainingModule = SPLADETrainingModule(cfg=cfg)
    data_module: TrainDataModule = TrainDataModule(cfg=cfg)

    checkpoint_dir: str = _resolve_checkpoint_dir(cfg.log_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if checkpoint_dir != os.path.join(cfg.log_dir, "checkpoints"):
        log_if_rank_zero(
            logger,
            "Existing checkpoints found; writing new checkpoints to "
            f"{checkpoint_dir}.",
        )
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step{step}-val_MRR_10{val_MRR_10:.4f}",
        monitor="val_MRR_10",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    wandb_cfg: DictConfig = cfg.training.wandb
    # Use the training config name for clearer W&B run identification.
    training_name: str = str(getattr(training_cfg, "name", "training"))
    wandb_logger: WandbLogger = WandbLogger(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=training_name,
        group=wandb_cfg.group,
        tags=list(wandb_cfg.tags) if wandb_cfg.tags is not None else None,
        mode=wandb_cfg.mode,
        log_model=wandb_cfg.log_model,
        save_dir=wandb_cfg.save_dir,
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    csv_logger: CSVLogger = CSVLogger(save_dir=cfg.log_dir, name="lightning_logs")

    trainer: L.Trainer = L.Trainer(
        deterministic=False,
        precision=precision,
        max_steps=cfg.training.max_steps,
        accumulate_grad_batches=cfg.training.grad_accumulation,
        val_check_interval=cfg.training.val_check_interval,
        limit_val_batches=cfg.training.limit_val_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
        default_root_dir=cfg.log_dir,
        logger=[wandb_logger, csv_logger],
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        **trainer_kwargs,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
