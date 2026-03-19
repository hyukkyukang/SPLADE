import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import mlflow
import torch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

from config.path import ABS_CONFIG_DIR
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils import log_if_rank_zero
from src.utils.logging import get_logger
from src.utils.mlflow_utils import (
    build_mlflow_dataset_input_from_metadata,
    has_logged_mlflow_dataset_inputs,
    has_logged_mlflow_model_outputs,
    log_mlflow_model_output,
    resolve_mlflow_tags,
    sanitize_mlflow_metric_name,
)
from src.utils.model_utils import (
    apply_checkpoint_model_config,
    build_splade_model,
    load_splade_checkpoint,
)
from src.utils.script_setup import (
    configure_default_entrypoint_environment,
    initialize_run,
    normalize_optional_str,
    resolve_model_source,
)
from src.utils.sparse_encoder import (
    build_native_sparse_encoder_adapter,
    build_sparse_encoder_from_checkpoint,
    build_sparse_encoder_from_huggingface,
    resolve_nanobeir_backend,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_default_entrypoint_environment(
    load_env=True,
    set_matmul_precision=True,
)


def _resolve_eval_device(use_cpu: bool) -> torch.device:
    if use_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_model_source_for_nanobeir(cfg: DictConfig) -> tuple[str, str]:
    if bool(cfg.nanobeir.use_huggingface_model):
        return str(cfg.model.huggingface_name), "huggingface"
    checkpoint_path: str | None = normalize_optional_str(cfg.testing.checkpoint_path)
    if checkpoint_path is None:
        raise ValueError(
            "testing.checkpoint_path must be set when "
            "nanobeir.use_huggingface_model=false."
        )
    return checkpoint_path, "checkpoint"


def _build_sparse_encoder(
    cfg: DictConfig,
    *,
    device: torch.device,
    model_source_kind: str,
) -> Any:
    benchmark_backend: str
    reason: str | None
    benchmark_backend, reason = resolve_nanobeir_backend(cfg)
    if benchmark_backend == "native":
        if reason is not None:
            log_if_rank_zero(
                logger,
                "Using native sparse benchmark adapter. "
                f"SentenceTransformers MLM path disabled: {reason}",
                level="warning",
            )
        use_cpu_for_model_build: bool = device.type == "cpu"
        model: SpladeModel = build_splade_model(
            cfg,
            use_cpu=use_cpu_for_model_build,
        )
        if model_source_kind == "checkpoint":
            checkpoint_path: str | None = normalize_optional_str(
                cfg.testing.checkpoint_path
            )
            if checkpoint_path is None:
                raise ValueError(
                    "testing.checkpoint_path must be set for checkpoint evaluation."
                )
            load_splade_checkpoint(model, checkpoint_path, logger=logger)
        model.to(device)
        model.eval()
        return build_native_sparse_encoder_adapter(
            cfg=cfg,
            model=model,
            device=device,
            batch_size=int(cfg.nanobeir.batch_size),
        )

    if model_source_kind == "huggingface":
        return build_sparse_encoder_from_huggingface(cfg=cfg, device=device)
    checkpoint_path: str | None = normalize_optional_str(cfg.testing.checkpoint_path)
    if checkpoint_path is None:
        raise ValueError("testing.checkpoint_path must be set for checkpoint evaluation.")
    return build_sparse_encoder_from_checkpoint(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def _extract_sparsity_stats(evaluator: SparseNanoBEIREvaluator) -> dict[str, float]:
    raw_sparsity_stats: Any = getattr(evaluator, "sparsity_stats", None)
    if not isinstance(raw_sparsity_stats, dict):
        return {}
    stats: dict[str, float] = {}
    metric_name: str
    metric_value: Any
    for metric_name in (
        "query_active_dims",
        "query_sparsity_ratio",
        "corpus_active_dims",
        "corpus_sparsity_ratio",
        "avg_flops",
    ):
        metric_value = raw_sparsity_stats.get(metric_name)
        try:
            stats[metric_name] = float(metric_value)
        except (TypeError, ValueError):
            continue
    return stats


def _collect_numeric_metrics(
    *,
    results: dict[str, Any],
    sparsity_stats: dict[str, float],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metric_name: str
    metric_value: Any
    for metric_name, metric_value in results.items():
        try:
            metrics[metric_name] = float(metric_value)
        except (TypeError, ValueError):
            continue
    for metric_name, metric_value in sparsity_stats.items():
        metrics[metric_name] = float(metric_value)
    return metrics


def _build_mlflow_dataset_input(dataset_name: str) -> Any:
    resolved_name: str = str(dataset_name).strip() or "nanobeir_dataset"
    metadata: dict[str, str] = {
        "dataset_name": resolved_name,
        "benchmark": "NanoBEIR",
        "task": "sparse_retrieval",
        "split": "test",
    }
    return build_mlflow_dataset_input_from_metadata(
        dataset_name=resolved_name,
        context="evaluation",
        metadata=metadata,
        source=f"https://huggingface.co/datasets/mteb/{resolved_name}",
    )


def _resolve_logged_model_name(
    cfg: DictConfig, model_source: str, model_source_kind: str
) -> str:
    model_cfg: DictConfig | None = cfg.get("model")
    if isinstance(model_cfg, DictConfig):
        hf_name: str | None = normalize_optional_str(model_cfg.get("huggingface_name"))
        if hf_name:
            candidate: str = Path(hf_name).name.strip()
            if candidate:
                return candidate
        model_name: str | None = normalize_optional_str(model_cfg.get("name"))
        if model_name:
            return model_name

    if model_source_kind == "checkpoint":
        source_name: str = Path(model_source).name.strip()
        if source_name.lower().endswith(".ckpt"):
            parent_name: str = Path(model_source).parent.name.strip()
            if parent_name:
                return parent_name
    fallback_name: str = Path(model_source).name.strip()
    if fallback_name:
        return fallback_name
    return "nanobeir-eval-model"


def _log_mlflow_run_datasets_and_model(
    *,
    cfg: DictConfig,
    run_id: str,
    tracking_uri: str | None,
    model_source: str,
    model_source_kind: str,
) -> None:
    mlflow_client: MlflowClient = MlflowClient(tracking_uri=tracking_uri)
    run = mlflow_client.get_run(run_id)

    if not has_logged_mlflow_dataset_inputs(run):
        dataset_inputs: list[Any] = [
            _build_mlflow_dataset_input(dataset_name=str(name))
            for name in cfg.nanobeir.datasets
        ]
        if dataset_inputs:
            mlflow_client.log_inputs(run_id=run_id, datasets=dataset_inputs)

    if has_logged_mlflow_model_outputs(run):
        return

    model_cfg: DictConfig | None = cfg.get("model")
    model_type: str = (
        str(model_cfg.get("type", "splade"))
        if isinstance(model_cfg, DictConfig)
        else "splade"
    )
    model_tags: dict[str, str] = {
        "benchmark": "NanoBEIR",
        "model_source": model_source,
        "model_source_kind": model_source_kind,
        "run_id": run_id,
    }
    if isinstance(model_cfg, DictConfig):
        hf_model_name: str | None = normalize_optional_str(model_cfg.get("huggingface_name"))
        if hf_model_name is not None:
            model_tags["huggingface_name"] = hf_model_name

    logged_model_name: str = _resolve_logged_model_name(
        cfg=cfg,
        model_source=model_source,
        model_source_kind=model_source_kind,
    )
    log_mlflow_model_output(
        mlflow_client=mlflow_client,
        run=run,
        run_id=run_id,
        logged_model_name=logged_model_name,
        model_type=model_type,
        model_tags=model_tags,
        tracking_uri=tracking_uri,
        step=0,
    )


def _log_to_mlflow(
    *,
    cfg: DictConfig,
    model_source: str,
    model_source_kind: str,
    numeric_metrics: dict[str, float],
    output_path: str | None,
) -> None:
    if "mlflow" not in cfg:
        return
    mlflow_cfg: DictConfig = cfg.mlflow
    if not bool(mlflow_cfg.get("enabled", True)):
        return

    tracking_uri: str | None = normalize_optional_str(mlflow_cfg.get("tracking_uri"))
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name: str = (
        normalize_optional_str(mlflow_cfg.get("experiment_name")) or "Eval-MTEB"
    )
    run_name: str | None = normalize_optional_str(mlflow_cfg.get("run_name"))
    mlflow.set_experiment(experiment_name)

    tags: dict[str, str] = resolve_mlflow_tags(
        mlflow_cfg.get("tags"), field_name="mlflow.tags"
    )
    tags.setdefault("benchmark", "nanobeir")
    tags.setdefault("evaluator", "SparseNanoBEIR")
    tags.setdefault("model_source_kind", model_source_kind)

    dataset_names: list[str] = [str(name) for name in cfg.nanobeir.datasets]
    params: dict[str, Any] = {
        "model_source": model_source,
        "model_source_kind": model_source_kind,
        "nanobeir.datasets": ",".join(dataset_names),
        "nanobeir.batch_size": int(cfg.nanobeir.batch_size),
        "nanobeir.max_seq_length": int(cfg.nanobeir.max_seq_length),
        "nanobeir.use_huggingface_model": bool(cfg.nanobeir.use_huggingface_model),
    }

    sanitized_metrics: dict[str, float] = {}
    metric_name: str
    metric_value: float
    for metric_name, metric_value in numeric_metrics.items():
        safe_name: str = sanitize_mlflow_metric_name(f"nanobeir.{metric_name}")
        sanitized_metrics[safe_name] = float(metric_value)

    with mlflow.start_run(run_name=run_name) as active_run:
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_params(params)
        if sanitized_metrics:
            mlflow.log_metrics(sanitized_metrics)
        if output_path is not None and os.path.isfile(output_path):
            mlflow.log_artifact(output_path)
        run_id: str | None = (
            str(active_run.info.run_id)
            if active_run is not None and active_run.info is not None
            else None
        )
        if run_id is None:
            return
        _log_mlflow_run_datasets_and_model(
            cfg=cfg,
            run_id=run_id,
            tracking_uri=tracking_uri,
            model_source=model_source,
            model_source_kind=model_source_kind,
        )
        log_if_rank_zero(
            logger,
            "Logged NanoBEIR metrics to MLflow "
            f"experiment={experiment_name} run_id={run_id}",
        )


def run(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)
    cfg = resolve_model_source(cfg, logger=logger, set_nanobeir_flag=True)
    hf_model_path: str | None = normalize_optional_str(cfg.testing.hf_model_path)
    checkpoint_exclude_keys: tuple[str, ...] = (
        "encode_path",
        "index_path",
        "encode_dir",
        "index_dir",
        "sparse_top_k",
        "sparse_min_weight",
        "max_input_length",
    )
    if hf_model_path is not None:
        checkpoint_exclude_keys = ("huggingface_name",) + checkpoint_exclude_keys
    cfg = apply_checkpoint_model_config(
        cfg,
        checkpoint_path=cfg.testing.checkpoint_path,
        logger=logger,
        exclude_keys=checkpoint_exclude_keys,
    )

    device: torch.device = _resolve_eval_device(bool(cfg.testing.use_cpu))
    model_source: str
    model_source_kind: str
    model_source, model_source_kind = _resolve_model_source_for_nanobeir(cfg)
    log_if_rank_zero(
        logger,
        f"Running NanoBEIR evaluation on {model_source_kind}: {model_source}",
    )
    log_if_rank_zero(logger, f"Using evaluation device: {device}")

    sparse_encoder: Any = _build_sparse_encoder(
        cfg,
        device=device,
        model_source_kind=model_source_kind,
    )
    eval_fn: Any = getattr(sparse_encoder, "eval", None)
    if callable(eval_fn):
        eval_fn()

    dataset_names: list[str] = [str(name) for name in cfg.nanobeir.datasets]
    evaluator: SparseNanoBEIREvaluator = SparseNanoBEIREvaluator(
        dataset_names=dataset_names,
        batch_size=int(cfg.nanobeir.batch_size),
    )

    with torch.no_grad():
        results: dict[str, Any] = evaluator(sparse_encoder)

    sparsity_stats: dict[str, float] = _extract_sparsity_stats(evaluator)
    numeric_metrics: dict[str, float] = _collect_numeric_metrics(
        results=results,
        sparsity_stats=sparsity_stats,
    )

    metric_name: str
    metric_value: float
    for metric_name, metric_value in sorted(numeric_metrics.items()):
        log_if_rank_zero(logger, f"{metric_name}: {metric_value}")

    mlflow_enabled: bool = "mlflow" in cfg and bool(cfg.mlflow.get("enabled", True))
    should_save_json: bool = bool(cfg.nanobeir.save_json) or mlflow_enabled
    output_path: str | None = None
    if should_save_json:
        output_path = os.path.join(cfg.log_dir, "nanobeir_metrics.json")
        payload: dict[str, Any] = {
            "benchmark": "nanobeir",
            "model_source": model_source,
            "model_source_kind": model_source_kind,
            "datasets": dataset_names,
            "results": results,
            "sparsity_stats": sparsity_stats,
            "numeric_metrics": numeric_metrics,
        }
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, indent=2)
        log_if_rank_zero(logger, f"Saved NanoBEIR metrics to {output_path}")

    try:
        _log_to_mlflow(
            cfg=cfg,
            model_source=model_source,
            model_source_kind=model_source_kind,
            numeric_metrics=numeric_metrics,
            output_path=output_path,
        )
    except Exception as exc:
        log_if_rank_zero(
            logger,
            f"MLflow logging failed for NanoBEIR evaluation: {exc}",
            level="warning",
        )


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate_mteb")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
