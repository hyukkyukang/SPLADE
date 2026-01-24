import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
from datetime import timedelta
from typing import Any

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.dataset.train_datamodule import TrainDataModule
from src.model.pl_module_train import SPLADETrainingModule
from src.utils import log_if_rank_zero, set_seed
from src.utils.logging import (
    get_logger,
    patch_hydra_argparser_for_python314,
    setup_tqdm_friendly_logging,
    suppress_dataloader_workers_warning,
    suppress_httpx_logging,
    suppress_pytorch_lightning_tips,
)


logger = get_logger(__name__, __file__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")
DDP_TIMEOUT_HOURS = 1

patch_hydra_argparser_for_python314()
suppress_pytorch_lightning_tips()
suppress_httpx_logging()
suppress_dataloader_workers_warning()


def _get_cpu_trainer_kwargs(cfg: DictConfig) -> dict[str, Any]:
    strategy_name = cfg.training.strategy
    num_devices = 1 if cfg.training.num_devices is None else cfg.training.num_devices
    kwargs: dict[str, Any] = {"accelerator": "cpu", "devices": num_devices}
    if strategy_name == "ddp":
        if num_devices > 1:
            kwargs["strategy"] = DDPStrategy(
                timeout=timedelta(hours=DDP_TIMEOUT_HOURS), static_graph=True
            )
        else:
            kwargs["strategy"] = "auto"
    elif strategy_name == "single":
        kwargs["devices"] = 1
        kwargs["strategy"] = "auto"
    else:
        raise ValueError(f"Invalid CPU strategy: {strategy_name}")
    return kwargs


def _get_gpu_trainer_kwargs(cfg: DictConfig) -> dict[str, Any]:
    strategy_name = cfg.training.strategy
    num_devices = torch.cuda.device_count()
    if cfg.training.num_devices is not None:
        num_devices = min(cfg.training.num_devices, num_devices)
    kwargs: dict[str, Any] = {"accelerator": "cuda", "devices": num_devices}
    if strategy_name == "ddp":
        kwargs["strategy"] = DDPStrategy(
            timeout=timedelta(hours=DDP_TIMEOUT_HOURS),
            static_graph=True,
            gradient_as_bucket_view=True,
        )
    elif strategy_name == "fsdp":
        kwargs["strategy"] = FSDPStrategy(timeout=timedelta(hours=DDP_TIMEOUT_HOURS))
    elif strategy_name == "deepspeed":
        kwargs["strategy"] = DeepSpeedStrategy()
    elif strategy_name == "single":
        device_id = int(getattr(cfg.training, "device_id", 0))
        kwargs = {
            "accelerator": "cuda",
            "devices": [device_id],
            "strategy": "auto",
        }
    else:
        raise ValueError(f"Invalid GPU strategy: {strategy_name}")
    return kwargs


def _get_precision(cfg: DictConfig) -> str:
    precision = cfg.training.precision
    if cfg.training.use_cpu and precision == "16-mixed":
        return "bf16-mixed"
    if not cfg.training.use_cpu and "bf16" in precision and not torch.cuda.is_bf16_supported():
        return "16-mixed"
    return precision


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    setup_tqdm_friendly_logging()
    os.makedirs(cfg.log_dir, exist_ok=True)

    set_seed(cfg.seed)
    log_if_rank_zero(logger, f"Random seed set to: {cfg.seed}")

    model = SPLADETrainingModule(cfg=cfg)
    data_module = TrainDataModule(cfg=cfg)

    trainer_kwargs = (
        _get_cpu_trainer_kwargs(cfg)
        if cfg.training.use_cpu
        else _get_gpu_trainer_kwargs(cfg)
    )
    precision = _get_precision(cfg)

    checkpoint_dir = os.path.join(cfg.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step{step}-val_mrr10{val_mrr10:.4f}",
        monitor="val_mrr10",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = L.Trainer(
        deterministic=False,
        precision=precision,
        max_steps=cfg.training.max_steps,
        accumulate_grad_batches=cfg.training.grad_accumulation,
        val_check_interval=cfg.training.val_check_interval,
        limit_val_batches=cfg.training.limit_val_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
        default_root_dir=cfg.log_dir,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        **trainer_kwargs,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
