import logging
import os
from typing import Any

import hydra
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from config.path import ABS_CONFIG_DIR
from src.data.datamodule import TrainDataModule
from src.model.pl_module_train import SPLADETrainingModule
from src.utils import log_if_rank_zero, set_seed
from src.utils.logging import (
    get_logger,
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


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    setup_tqdm_friendly_logging()
    os.makedirs(cfg.log_dir, exist_ok=True)

    set_seed(cfg.seed)
    log_if_rank_zero(logger, f"Random seed set to: {cfg.seed}")

    model: SPLADETrainingModule = SPLADETrainingModule(cfg=cfg)
    data_module: TrainDataModule = TrainDataModule(cfg=cfg)

    training_cfg: DictConfig = cfg.training
    trainer_kwargs: dict[str, Any] = (
        get_cpu_trainer_kwargs(training_cfg)
        if training_cfg.use_cpu
        else get_gpu_trainer_kwargs(training_cfg)
    )
    precision: str = resolve_precision(training_cfg)

    checkpoint_dir: str = os.path.join(cfg.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step{step}-val_mrr10{val_mrr10:.4f}",
        monitor="val_mrr10",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    wandb_cfg: DictConfig = cfg.training.wandb
    wandb_logger: WandbLogger = WandbLogger(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=wandb_cfg.name,
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
    main()
