import logging
from typing import Any

import lightning as L
import torch

from src.utils import log_if_rank_zero


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def extract_scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    scalar_metrics: dict[str, float] = {}
    metric_name: str
    metric_value: Any
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, torch.Tensor):
            if metric_value.numel() != 1:
                continue
            scalar_metrics[str(metric_name)] = float(metric_value.detach().cpu().item())
            continue
        if isinstance(metric_value, (int, float)):
            scalar_metrics[str(metric_name)] = float(metric_value)
    return scalar_metrics


def flatten_validate_results(results: list[dict[str, Any]]) -> dict[str, float]:
    if len(results) == 1:
        return extract_scalar_metrics(results[0])

    flattened: dict[str, float] = {}
    dataloader_idx: int
    metrics: dict[str, Any]
    for dataloader_idx, metrics in enumerate(results):
        scalar_metrics: dict[str, float] = extract_scalar_metrics(metrics)
        metric_name: str
        metric_value: float
        for metric_name, metric_value in scalar_metrics.items():
            flattened[f"dataloader_{dataloader_idx}.{metric_name}"] = metric_value
    return flattened


def extract_callback_validation_metrics(trainer: L.Trainer) -> dict[str, float]:
    callback_metrics: dict[str, float] = {}
    metric_name: str
    metric_value: Any
    for metric_name, metric_value in trainer.callback_metrics.items():
        key: str = str(metric_name)
        if not key.startswith("val_"):
            continue
        if isinstance(metric_value, torch.Tensor):
            if metric_value.numel() != 1:
                continue
            callback_metrics[key] = float(metric_value.detach().cpu().item())
            continue
        if isinstance(metric_value, (int, float)):
            callback_metrics[key] = float(metric_value)
    return callback_metrics


def partition_validation_metrics(
    all_metrics: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    reranking_metrics: dict[str, float] = {}
    nanobeir_metrics: dict[str, float] = {}
    metric_name: str
    metric_value: float
    for metric_name, metric_value in all_metrics.items():
        if metric_name.startswith("val_nanobeir_"):
            nanobeir_metrics[metric_name] = metric_value
            continue
        if metric_name.startswith("val_"):
            reranking_metrics[metric_name] = metric_value
    return reranking_metrics, nanobeir_metrics


def resolve_training_style_validation_metrics(
    *,
    validate_results: list[dict[str, Any]],
    trainer: L.Trainer,
    logger: logging.Logger,
    mismatch_atol: float = 1e-6,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Resolve standalone validation metrics using training-time semantics.

    Training-time consumers (checkpoint/early-stop/logging) read `val_*` values from
    trainer callback metrics. For standalone validation we keep that as source-of-truth
    and use raw validate-result scalars only as a fallback.
    """
    result_metrics: dict[str, float] = flatten_validate_results(validate_results)
    callback_metrics: dict[str, float] = extract_callback_validation_metrics(trainer)
    if not callback_metrics:
        return result_metrics, result_metrics, callback_metrics

    metric_name: str
    callback_value: float
    for metric_name, callback_value in callback_metrics.items():
        if metric_name not in result_metrics:
            continue
        result_value: float = float(result_metrics[metric_name])
        if abs(result_value - callback_value) <= float(mismatch_atol):
            continue
        log_if_rank_zero(
            logger,
            "Validation semantics parity check: callback metric differs from "
            f"validate-result metric for {metric_name}: "
            f"callback={callback_value}, result={result_value}. "
            "Using callback value to match training-time semantics.",
            level="warning",
        )
    return callback_metrics, result_metrics, callback_metrics


def log_metric_block(
    *,
    logger: logging.Logger,
    title: str,
    metrics: dict[str, float],
) -> None:
    log_if_rank_zero(logger, title)
    if not metrics:
        log_if_rank_zero(
            logger,
            "  no metrics available",
            level="warning",
        )
        return
    metric_name: str
    for metric_name in sorted(metrics):
        log_if_rank_zero(logger, f"  {metric_name}: {metrics[metric_name]}")
