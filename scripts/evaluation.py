import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# For Python 3.14 compatibility
from src.utils.logging import patch_hydra_argparser_for_python314

patch_hydra_argparser_for_python314()

import os
from datetime import timedelta
from typing import Any

import hydra
import lightning as L
import torch
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.eval_datamodule import EvalDataModule
from src.model.pl_module_eval import SPLADEEvaluationModule
from src.utils import log_if_rank_zero, set_seed
from src.utils.logging import (
    get_logger,
    setup_tqdm_friendly_logging,
    suppress_dataloader_workers_warning,
    suppress_httpx_logging,
    suppress_pytorch_lightning_tips,
)

logger = get_logger(__name__, __file__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")
DDP_TIMEOUT_HOURS = 1

suppress_pytorch_lightning_tips()
suppress_httpx_logging()
suppress_dataloader_workers_warning()


def _get_cpu_trainer_kwargs(cfg: DictConfig) -> dict[str, Any]:
    strategy_name: str = cfg.testing.strategy
    num_devices: int = 1 if cfg.testing.num_devices is None else cfg.testing.num_devices
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
    strategy_name: str = cfg.testing.strategy
    num_devices: int = torch.cuda.device_count()
    if cfg.testing.num_devices is not None:
        num_devices = min(cfg.testing.num_devices, num_devices)
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
        device_id: int = int(getattr(cfg.testing, "device_id", 0))
        kwargs = {
            "accelerator": "cuda",
            "devices": [device_id],
            "strategy": "auto",
        }
    else:
        raise ValueError(f"Invalid GPU strategy: {strategy_name}")
    return kwargs


def _get_precision(cfg: DictConfig) -> str:
    precision: str = cfg.testing.precision
    if cfg.testing.use_cpu and precision == "16-mixed":
        return "bf16-mixed"
    if (
        not cfg.testing.use_cpu
        and "bf16" in precision
        and not torch.cuda.is_bf16_supported()
    ):
        return "16-mixed"
    return precision


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate")
def main(cfg: DictConfig) -> None:
    setup_tqdm_friendly_logging()
    os.makedirs(cfg.log_dir, exist_ok=True)
    set_seed(cfg.seed)
    log_if_rank_zero(logger, f"Random seed set to: {cfg.seed}")

    eval_module: SPLADEEvaluationModule = SPLADEEvaluationModule(cfg=cfg)
    eval_module.eval()
    data_module: EvalDataModule = EvalDataModule(cfg=cfg)

    trainer_kwargs = (
        _get_cpu_trainer_kwargs(cfg)
        if cfg.testing.use_cpu
        else _get_gpu_trainer_kwargs(cfg)
    )
    precision: str = _get_precision(cfg)

    trainer = L.Trainer(
        precision=precision,
        default_root_dir=cfg.log_dir,
        logger=False,
        **trainer_kwargs,
    )

    results: list[dict[str, float]] = trainer.test(
        model=eval_module, datamodule=data_module
    )
    if results:
        metrics: dict[str, float] = results[0]
        for name, value in metrics.items():
            logger.info("%s %s: %.4f", str(cfg.dataset.name), name, value)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
