from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig

from src.metric.retrieval import RetrievalMetrics
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import log_if_rank_zero
from src.utils.model_utils import build_splade_model, load_splade_checkpoint


def build_splade_model_with_checkpoint(
    cfg: DictConfig,
    *,
    use_cpu: bool,
    checkpoint_path: str | None,
    logger: logging.Logger,
) -> SpladeModel:
    """Build a SPLADE model and optionally load a checkpoint."""
    model: SpladeModel = build_splade_model(cfg, use_cpu=bool(use_cpu))
    if checkpoint_path:
        missing: list[str]
        unexpected: list[str]
        missing, unexpected = load_splade_checkpoint(
            model, checkpoint_path, logger=logger
        )
        log_if_rank_zero(
            logger,
            f"Loaded checkpoint. Missing: {len(missing)}, unexpected: {len(unexpected)}",
        )
    return model


def finalize_retrieval_metrics(
    metric_collection: RetrievalMetrics,
    module: L.LightningModule,
    logger: logging.Logger,
) -> None:
    """Gather, log, and reset retrieval metrics after evaluation."""
    trainer: L.Trainer = module.trainer
    world_size: int = int(trainer.world_size)
    all_gather_fn: Any | None = module.all_gather if world_size > 1 else None
    has_data: bool = metric_collection.gather(
        world_size=world_size, all_gather_fn=all_gather_fn
    )
    if not has_data:
        log_if_rank_zero(
            logger, "No predictions accumulated during testing.", level="warning"
        )
        return
    if trainer.is_global_zero:
        metrics: dict[str, torch.Tensor] = metric_collection.compute()
        module.log_dict(metrics, sync_dist=False, prog_bar=True, rank_zero_only=True)
    metric_collection.reset()
