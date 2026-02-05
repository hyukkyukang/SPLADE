import logging
from typing import Any, Callable

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig

from src.metric.retrieval import RetrievalMetrics
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import log_if_rank_zero
from src.utils.model_utils import build_splade_model, load_splade_checkpoint

_VALID_TORCH_COMPILE_MODES: tuple[str, ...] = (
    "default",
    "reduce-overhead",
    "max-autotune",
)


def resolve_cudagraph_mark_step() -> Callable[[], None] | None:
    if not hasattr(torch, "compiler"):
        return None
    compiler_mod = torch.compiler
    if not hasattr(compiler_mod, "cudagraph_mark_step_begin"):
        return None
    mark_step_fn = compiler_mod.cudagraph_mark_step_begin
    return mark_step_fn if callable(mark_step_fn) else None


def build_compile_kwargs(mode: str) -> dict[str, Any]:
    return {"mode": mode}


def validate_torch_compile_mode(mode_value: Any) -> tuple[str, dict[str, Any]]:
    compile_mode: str = str(mode_value).lower()
    if compile_mode not in _VALID_TORCH_COMPILE_MODES:
        raise ValueError(
            "Unsupported torch.compile mode: "
            f"{mode_value!r}. Expected one of "
            f"{sorted(_VALID_TORCH_COMPILE_MODES)}."
        )
    return compile_mode, build_compile_kwargs(compile_mode)


def resolve_query_sparsify_config(
    metadata: dict[str, Any],
) -> tuple[list[int], float, int | None]:
    exclude_ids: list[int] = [
        int(token_id) for token_id in metadata.get("exclude_token_ids") or []
    ]
    min_weight_value: float = float(metadata.get("min_weight") or 0.0)
    top_k_value: int | None = (
        None if metadata.get("top_k") is None else int(metadata["top_k"])
    )
    return exclude_ids, min_weight_value, top_k_value


def prepare_score_buffers(doc_count: int) -> tuple[np.ndarray, np.ndarray]:
    score_buffer = np.zeros(int(doc_count), dtype=np.float32)
    seen_buffer = np.zeros(int(doc_count), dtype=np.uint8)
    return score_buffer, seen_buffer


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
