import logging
import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import RetrievalDataModule
from src.model.pl_module import RetrievalSearchLightningModule
from src.utils import log_if_rank_zero, set_seed
from src.utils.logging import (
    get_logger,
    setup_tqdm_friendly_logging,
    suppress_lightning_recommendation_tips,
)
from src.utils.model_utils import apply_checkpoint_model_config
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


def _initialize_run(cfg: DictConfig, *, suppress_lightning_tips: bool) -> None:
    setup_tqdm_friendly_logging()
    if suppress_lightning_tips:
        suppress_lightning_recommendation_tips()
    os.makedirs(cfg.log_dir, exist_ok=True)
    set_seed(cfg.seed)
    log_if_rank_zero(logger, f"Random seed set to: {cfg.seed}")


def _normalize_optional_path(value: Any) -> str | None:
    if value is None:
        return None
    text: str = str(value).strip()
    return text if text else None


def _resolve_model_source(cfg: DictConfig) -> DictConfig:
    testing_cfg: DictConfig = cfg.testing
    hf_model_path: str | None = _normalize_optional_path(
        getattr(testing_cfg, "hf_model_path", None)
    )
    checkpoint_path: str | None = _normalize_optional_path(
        getattr(testing_cfg, "checkpoint_path", None)
    )

    if hf_model_path:
        if checkpoint_path:
            raise ValueError(
                "Provide either testing.hf_model_path or "
                "testing.checkpoint_path, not both."
            )
        cfg.model.huggingface_name = hf_model_path
        log_if_rank_zero(logger, f"Using Hugging Face model: {hf_model_path}")
        return cfg

    if not checkpoint_path:
        raise ValueError(
            "testing.checkpoint_path must be set unless "
            "testing.hf_model_path is provided."
        )
    return cfg


def _append_rank_suffix(path: Path, rank: int) -> Path:
    suffix: str = path.suffix
    stem: str = path.stem
    if suffix:
        return path.with_name(f"{stem}.rank{rank}{suffix}")
    return path.with_name(f"{path.name}.rank{rank}")


def _merge_rank_outputs(cfg: DictConfig, trainer: L.Trainer) -> None:
    search_cfg: DictConfig | None = getattr(cfg, "search", None)
    merge_ranks: bool = bool(
        search_cfg.merge_ranks
        if search_cfg is not None and "merge_ranks" in search_cfg
        else False
    )
    if not merge_ranks:
        return

    world_size: int = int(trainer.world_size)
    global_rank: int = int(trainer.global_rank)
    if world_size <= 1:
        return

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if global_rank != 0:
        return

    if not bool(cfg.testing.save_run):
        log_if_rank_zero(
            logger,
            "Skipping rank merge because testing.save_run is false.",
            level="warning",
        )
        return

    run_path_value: str | None = cfg.testing.run_path
    if not run_path_value:
        raise ValueError("testing.run_path must be set to merge rank outputs.")

    run_path = Path(str(run_path_value))
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("w", encoding="utf-8") as output_handle:
        for rank in range(world_size):
            rank_path: Path = _append_rank_suffix(run_path, rank)
            if not rank_path.exists():
                log_if_rank_zero(
                    logger,
                    f"Missing rank output file: {rank_path}",
                    level="warning",
                )
                continue
            with rank_path.open("r", encoding="utf-8") as rank_handle:
                for line in rank_handle:
                    output_handle.write(line)
    log_if_rank_zero(logger, f"Merged rank outputs into {run_path}")


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="search")
def main(cfg: DictConfig) -> None:
    _initialize_run(cfg, suppress_lightning_tips=True)
    cfg = _resolve_model_source(cfg)
    cfg = apply_checkpoint_model_config(
        cfg,
        checkpoint_path=cfg.testing.checkpoint_path,
        logger=logger,
    )

    search_module = RetrievalSearchLightningModule(cfg=cfg)
    data_module = RetrievalDataModule(cfg=cfg)
    search_module.eval()

    testing_cfg: DictConfig = cfg.testing
    trainer_kwargs: dict[str, Any] = (
        get_cpu_trainer_kwargs(testing_cfg)
        if testing_cfg.use_cpu
        else get_gpu_trainer_kwargs(testing_cfg)
    )
    precision: str = resolve_precision(testing_cfg)

    trainer: L.Trainer = L.Trainer(
        precision=precision,
        default_root_dir=cfg.log_dir,
        logger=False,
        **trainer_kwargs,
    )
    trainer.test(model=search_module, datamodule=data_module)
    _merge_rank_outputs(cfg, trainer)
    log_if_rank_zero(logger, "Search complete")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
