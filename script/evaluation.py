import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.utils import log_if_rank_zero
from src.utils.evaluation_mode import enforce_retrieval_evaluation_isolation
from src.utils.logging import (
    get_logger,
    is_rank_zero as is_logging_rank_zero,
    suppress_urllib3_insecure_request_warning,
)
from src.utils.mlflow_utils import (
    add_mlflow_model_config_tags,
    build_mlflow_dataset_input_from_metadata,
    configure_mlflow_tls,
    has_logged_mlflow_dataset_inputs,
    has_logged_mlflow_model_outputs,
    log_mlflow_model_output,
    resolve_mlflow_logged_model_name,
    resolve_mlflow_model_type,
    resolve_mlflow_tags,
    sanitize_mlflow_metric_name,
)
from src.utils.metrics_io import to_jsonable
from src.utils.model_utils import apply_checkpoint_model_config, resolve_tagged_output_dir
from src.utils.normalize import normalize_optional_bool
from src.utils.script_setup import (
    configure_default_entrypoint_environment,
    initialize_run,
    normalize_optional_str,
    normalize_tag,
    resolve_model_source,
    resolve_trainer_settings,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_default_entrypoint_environment(
    load_env=True,
    set_matmul_precision=True,
)
suppress_urllib3_insecure_request_warning()


def _resolve_index_path(cfg: DictConfig) -> Path:
    index_dir_value: str = str(cfg.encoding.index_dir)
    if not index_dir_value:
        raise ValueError("encoding.index_dir must be set for retrieval evaluation.")
    index_path: Path = resolve_tagged_output_dir(
        index_dir_value,
        model_name=str(cfg.model.name),
        tag=cfg.encoding.index_tag,
    )
    if not index_path.exists():
        raise FileNotFoundError(
            f"Retrieval index does not exist: {index_path}. "
            "Build the index first with script/index.py."
        )
    return index_path


def _resolve_mlflow_run_name(cfg: DictConfig) -> str:
    """Resolve the MLflow run name from config/tag with sensible fallback."""
    mlflow_cfg: DictConfig = cfg.mlflow
    explicit_run_name: str | None = normalize_optional_str(mlflow_cfg.get("run_name"))
    if explicit_run_name is not None:
        return explicit_run_name
    tag_value: str | None = normalize_tag(cfg.get("tag"))
    if tag_value is not None:
        return tag_value
    return f"{str(cfg.model.name)}-{str(cfg.dataset.name)}"


def _build_mlflow_tags(
    cfg: DictConfig, *, model_source_kind: str
) -> dict[str, str]:
    """Build MLflow tags for retrieval evaluation runs."""
    tags: dict[str, str] = resolve_mlflow_tags(
        cfg.mlflow.get("tags"), field_name="mlflow.tags"
    )
    tags.setdefault("evaluation_type", str(cfg.evaluation.type))
    tags.setdefault("evaluation_mode", "retrieval_index_based")
    tags.setdefault("dataset_name", str(cfg.dataset.name))
    tags.setdefault("model_name", str(cfg.model.name))
    tags.setdefault("model_source_kind", model_source_kind)
    tags.setdefault("log_dir", str(cfg.log_dir))
    tag_value: str | None = normalize_tag(cfg.get("tag"))
    if tag_value is not None:
        tags.setdefault("tag", tag_value)
    return tags


def _build_mlflow_dataset_input(
    dataset_cfg: DictConfig, *, dataset_name: str, context: str
) -> Any:
    """Create an MLflow dataset input for the evaluation dataset config."""

    def _as_text(key: str, default: str = "") -> str:
        value: str | None = normalize_optional_str(dataset_cfg.get(key))
        if value is None:
            return default
        return value

    resolved_dataset_name: str = str(dataset_name).strip() or _as_text("name", "dataset")
    metadata: dict[str, str] = {
        "dataset_name": resolved_dataset_name,
        "dataset_type": _as_text("type"),
        "split": _as_text("split"),
        "hf_name": _as_text("hf_name"),
        "hf_subset": _as_text("hf_subset"),
        "hf_split": _as_text("hf_split"),
        "beir_dataset": _as_text("beir_dataset"),
    }
    hf_name: str | None = normalize_optional_str(dataset_cfg.get("hf_name"))
    source_url: str | None = (
        f"https://huggingface.co/datasets/{hf_name}" if hf_name is not None else None
    )
    return build_mlflow_dataset_input_from_metadata(
        dataset_name=resolved_dataset_name,
        context=context,
        metadata=metadata,
        source=source_url,
    )


def _build_mlflow_qrels_input(dataset_cfg: DictConfig, *, dataset_name: str) -> Any | None:
    """Create an MLflow dataset input for qrels/ground-truth metadata when available."""

    def _as_text(key: str, default: str = "") -> str:
        value: str | None = normalize_optional_str(dataset_cfg.get(key))
        if value is None:
            return default
        return value

    qrels_hf_name: str | None = normalize_optional_str(dataset_cfg.get("qrels_hf_name"))
    qrels_hf_subset: str | None = normalize_optional_str(dataset_cfg.get("qrels_hf_subset"))
    raw_qrels_hf_split: str | None = normalize_optional_str(
        dataset_cfg.get("qrels_hf_split")
    )
    if (
        qrels_hf_name is None
        and qrels_hf_subset is None
        and raw_qrels_hf_split is None
    ):
        return None
    qrels_hf_split: str = raw_qrels_hf_split or _as_text("hf_split") or _as_text("split")

    resolved_qrels_name: str = str(dataset_name).strip() or "dataset"
    metadata: dict[str, str] = {
        "dataset_name": resolved_qrels_name,
        "dataset_type": _as_text("type"),
        "beir_dataset": _as_text("beir_dataset"),
        "qrels_hf_name": qrels_hf_name or _as_text("hf_name"),
        "qrels_hf_subset": qrels_hf_subset or "",
        "qrels_hf_split": qrels_hf_split,
    }
    source_repo: str | None = qrels_hf_name or normalize_optional_str(dataset_cfg.get("hf_name"))
    source_url: str | None = (
        f"https://huggingface.co/datasets/{source_repo}"
        if source_repo is not None
        else None
    )
    return build_mlflow_dataset_input_from_metadata(
        dataset_name=f"{resolved_qrels_name}_qrels",
        context="ground_truth",
        metadata=metadata,
        source=source_url,
    )


def _resolve_model_source_for_logging(cfg: DictConfig) -> tuple[str, str]:
    """Resolve the primary model source used for evaluation logging."""
    checkpoint_path: str | None = normalize_optional_str(cfg.testing.get("checkpoint_path"))
    if checkpoint_path is not None:
        return checkpoint_path, "checkpoint"
    hf_model_path: str | None = normalize_optional_str(cfg.testing.get("hf_model_path"))
    if hf_model_path is not None:
        return hf_model_path, "huggingface"
    huggingface_name: str | None = normalize_optional_str(cfg.model.get("huggingface_name"))
    if huggingface_name is not None:
        return huggingface_name, "huggingface"
    model_name: str | None = normalize_optional_str(cfg.model.get("name"))
    if model_name is not None:
        return model_name, "config"
    return "unknown", "config"


def _collect_numeric_metrics(test_results: list[dict[str, Any]]) -> dict[str, float]:
    """Flatten trainer test results into a numeric metric mapping."""
    metrics: dict[str, float] = {}
    dataloader_idx: int
    dataloader_metrics: dict[str, Any]
    use_dataloader_prefix: bool = len(test_results) > 1
    for dataloader_idx, dataloader_metrics in enumerate(test_results):
        if not isinstance(dataloader_metrics, dict):
            continue
        metric_name: str
        metric_value: Any
        for metric_name, metric_value in dataloader_metrics.items():
            try:
                numeric_value: float = float(metric_value)
            except (TypeError, ValueError):
                continue
            resolved_name: str = (
                f"dataloader_{dataloader_idx}.{metric_name}"
                if use_dataloader_prefix
                else str(metric_name)
            )
            metrics[resolved_name] = numeric_value
    return metrics


def _build_mlflow_params(
    cfg: DictConfig,
    *,
    model_source: str,
    model_source_kind: str,
    index_path: Path,
) -> dict[str, Any]:
    """Build a compact MLflow parameter block for retrieval evaluation."""
    params: dict[str, Any] = {
        "evaluation_mode": "retrieval_index_based",
        "evaluation.type": str(cfg.evaluation.type),
        "model.name": str(cfg.model.name),
        "model_source": model_source,
        "model_source_kind": model_source_kind,
        "dataset.name": str(cfg.dataset.name),
        "dataset.type": str(cfg.dataset.type),
        "index_path": str(index_path),
    }

    def _set_if_present(key: str, value: Any) -> None:
        if value is None:
            return
        params[key] = value if isinstance(value, (bool, int, float, str)) else str(value)

    _set_if_present("model.family", normalize_optional_str(cfg.model.get("family")))
    _set_if_present("model.type", normalize_optional_str(cfg.model.get("type")))
    _set_if_present(
        "model.huggingface_name", normalize_optional_str(cfg.model.get("huggingface_name"))
    )
    _set_if_present("dataset.split", normalize_optional_str(cfg.dataset.get("split")))
    _set_if_present("dataset.hf_name", normalize_optional_str(cfg.dataset.get("hf_name")))
    _set_if_present("dataset.hf_subset", normalize_optional_str(cfg.dataset.get("hf_subset")))
    _set_if_present("dataset.hf_split", normalize_optional_str(cfg.dataset.get("hf_split")))
    _set_if_present(
        "dataset.beir_dataset", normalize_optional_str(cfg.dataset.get("beir_dataset"))
    )
    _set_if_present(
        "dataset.qrels_hf_name", normalize_optional_str(cfg.dataset.get("qrels_hf_name"))
    )
    _set_if_present(
        "dataset.qrels_hf_subset",
        normalize_optional_str(cfg.dataset.get("qrels_hf_subset")),
    )
    _set_if_present(
        "dataset.qrels_hf_split",
        normalize_optional_str(cfg.dataset.get("qrels_hf_split")),
    )
    _set_if_present(
        "testing.checkpoint_path",
        normalize_optional_str(cfg.testing.get("checkpoint_path")),
    )
    _set_if_present(
        "testing.hf_model_path",
        normalize_optional_str(cfg.testing.get("hf_model_path")),
    )
    _set_if_present("testing.batch_size", cfg.testing.get("batch_size"))
    _set_if_present("testing.precision", normalize_optional_str(cfg.testing.get("precision")))
    _set_if_present("testing.num_devices", cfg.testing.get("num_devices"))
    _set_if_present(
        "testing.scoring_method", normalize_optional_str(cfg.testing.get("scoring_method"))
    )
    _set_if_present(
        "testing.scoring_backend",
        normalize_optional_str(cfg.testing.get("scoring_backend")),
    )
    _set_if_present("testing.wand_block_size", cfg.testing.get("wand_block_size"))
    _set_if_present("testing.sparse_top_k", cfg.testing.get("sparse_top_k"))
    _set_if_present("testing.sparse_min_weight", cfg.testing.get("sparse_min_weight"))
    _set_if_present("encoding.index_tag", normalize_optional_str(cfg.encoding.get("index_tag")))

    metadata_path: Path = index_path / "metadata.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata: dict[str, Any] = json.load(metadata_file)
        metadata_key: str
        for metadata_key in (
            "doc_count",
            "nnz",
            "value_dtype",
            "encoded_value_dtype",
            "top_k",
            "min_weight",
            "block_size",
        ):
            if metadata_key in metadata:
                _set_if_present(f"index.{metadata_key}", metadata[metadata_key])
    return params


def _log_mlflow_run_datasets_and_model(
    *,
    cfg: DictConfig,
    run_id: str,
    tracking_uri: str | None,
    model_source: str,
    model_source_kind: str,
) -> None:
    """Populate MLflow dataset inputs and model outputs for retrieval evaluation."""
    mlflow_client: MlflowClient = MlflowClient(tracking_uri=tracking_uri)
    run = mlflow_client.get_run(run_id)

    if not has_logged_mlflow_dataset_inputs(run):
        dataset_inputs: list[Any] = []
        dataset_cfg: DictConfig | None = cfg.get("dataset")
        if isinstance(dataset_cfg, DictConfig):
            dataset_inputs.append(
                _build_mlflow_dataset_input(
                    dataset_cfg,
                    dataset_name=str(dataset_cfg.get("name", "evaluation_dataset")),
                    context="evaluation",
                )
            )
            qrels_input: Any | None = _build_mlflow_qrels_input(
                dataset_cfg,
                dataset_name=str(dataset_cfg.get("name", "evaluation_dataset")),
            )
            if qrels_input is not None:
                dataset_inputs.append(qrels_input)
        if dataset_inputs:
            mlflow_client.log_inputs(run_id=run_id, datasets=dataset_inputs)

    if has_logged_mlflow_model_outputs(run):
        return

    model_cfg: DictConfig | None = cfg.get("model")
    if not isinstance(model_cfg, DictConfig):
        raise TypeError("cfg.model must be set for MLflow model logging.")

    model_tags: dict[str, str] = {
        "evaluation_type": str(cfg.evaluation.type),
        "evaluation_mode": "retrieval_index_based",
        "dataset_name": str(cfg.dataset.name),
        "model_source": model_source,
        "model_source_kind": model_source_kind,
        "run_id": run_id,
    }
    hf_model_name: str | None = normalize_optional_str(model_cfg.get("huggingface_name"))
    if hf_model_name is not None:
        model_tags["huggingface_name"] = hf_model_name
    model_tags = add_mlflow_model_config_tags(model_tags, model_cfg)

    logged_model_name: str = resolve_mlflow_logged_model_name(model_cfg)
    log_mlflow_model_output(
        mlflow_client=mlflow_client,
        run=run,
        run_id=run_id,
        logged_model_name=logged_model_name,
        model_type=resolve_mlflow_model_type(model_cfg),
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
    output_path: Path,
    index_path: Path,
) -> None:
    """Log retrieval evaluation results, metadata, and artifacts to MLflow."""
    if not is_logging_rank_zero() or "mlflow" not in cfg:
        return

    mlflow_cfg: DictConfig = cfg.mlflow
    mlflow_enabled: bool | None = normalize_optional_bool(mlflow_cfg.get("enabled"))
    if mlflow_enabled is False:
        return

    configure_mlflow_tls(
        mlflow_cfg,
        sampling_interval_field="mlflow.system_metrics_sampling_interval",
        samples_before_logging_field="mlflow.system_metrics_samples_before_logging",
    )
    tracking_uri: str | None = normalize_optional_str(mlflow_cfg.get("tracking_uri"))
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name: str = (
        normalize_optional_str(mlflow_cfg.get("experiment_name")) or "Eval-MSMARCO"
    )
    mlflow.set_experiment(experiment_name)

    run_kwargs: dict[str, Any] = {"run_name": _resolve_mlflow_run_name(cfg)}
    system_metrics_enabled: bool | None = normalize_optional_bool(
        mlflow_cfg.get("system_metrics_enabled")
    )
    if system_metrics_enabled is not None:
        run_kwargs["log_system_metrics"] = bool(system_metrics_enabled)

    tags: dict[str, str] = _build_mlflow_tags(
        cfg, model_source_kind=model_source_kind
    )
    params: dict[str, Any] = _build_mlflow_params(
        cfg,
        model_source=model_source,
        model_source_kind=model_source_kind,
        index_path=index_path,
    )
    sanitized_metrics: dict[str, float] = {
        sanitize_mlflow_metric_name(metric_name): float(metric_value)
        for metric_name, metric_value in numeric_metrics.items()
    }

    with mlflow.start_run(**run_kwargs) as active_run:
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        if sanitized_metrics:
            mlflow.log_metrics(sanitized_metrics)
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
        if bool(mlflow_cfg.get("log_artifacts", False)):
            if output_path.is_file():
                mlflow.log_artifact(output_path.as_posix())

            evaluate_log_path: Path = Path(cfg.log_dir) / "evaluate.log"
            if evaluate_log_path.is_file():
                mlflow.log_artifact(
                    evaluate_log_path.as_posix(), artifact_path="logs"
                )

            metadata_path: Path = index_path / "metadata.json"
            if metadata_path.is_file():
                mlflow.log_artifact(metadata_path.as_posix(), artifact_path="index")

        log_if_rank_zero(
            logger,
            "Logged retrieval evaluation metrics to MLflow "
            f"experiment={experiment_name} run_id={run_id}",
        )


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate")
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)
    cfg = enforce_retrieval_evaluation_isolation(cfg, logger=logger)

    cfg = resolve_model_source(cfg, logger=logger, set_nanobeir_flag=False)
    cfg = apply_checkpoint_model_config(
        cfg,
        checkpoint_path=cfg.testing.checkpoint_path,
        logger=logger,
    )

    index_path: Path = _resolve_index_path(cfg)
    log_if_rank_zero(logger, f"Using retrieval index: {index_path}")

    from src.data.pl_module import RetrievalDataModule
    from src.model.pl_module import RetrievalEvalLightningModule

    eval_module: RetrievalEvalLightningModule = RetrievalEvalLightningModule(cfg=cfg)
    data_module: RetrievalDataModule = RetrievalDataModule(cfg=cfg)
    eval_module.eval()

    trainer_kwargs, precision = resolve_trainer_settings(cfg.testing)
    trainer: L.Trainer = L.Trainer(
        precision=precision,
        deterministic=False,
        default_root_dir=cfg.log_dir,
        logger=False,
        enable_checkpointing=False,
        use_distributed_sampler=False,
        **trainer_kwargs,
    )

    test_results: list[dict[str, Any]] = trainer.test(
        model=eval_module, datamodule=data_module
    )
    if not trainer.is_global_zero:
        return
    if not test_results:
        log_if_rank_zero(
            logger,
            "trainer.test returned no metrics.",
            level="warning",
        )
        return

    log_if_rank_zero(logger, "Retrieval Evaluation Metrics:")
    dataloader_idx: int
    metrics: dict[str, Any]
    for dataloader_idx, metrics in enumerate(test_results):
        if not metrics:
            log_if_rank_zero(
                logger,
                f"dataloader_{dataloader_idx}: empty metrics",
                level="warning",
            )
            continue
        metric_name: str
        for metric_name in sorted(metrics):
            log_if_rank_zero(
                logger,
                f"dataloader_{dataloader_idx} {metric_name}: {metrics[metric_name]}",
            )

    output_path: str = os.path.join(cfg.log_dir, "evaluation_metrics.json")
    payload: dict[str, Any] = {
        "evaluation_mode": "retrieval_index_based",
        "checkpoint_path": cfg.testing.checkpoint_path,
        "hf_model_path": cfg.testing.hf_model_path,
        "index_path": str(index_path),
        "results": to_jsonable(test_results),
    }
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)
    log_if_rank_zero(logger, f"Saved evaluation metrics to {output_path}")

    numeric_metrics: dict[str, float] = _collect_numeric_metrics(test_results)
    model_source: str
    model_source_kind: str
    model_source, model_source_kind = _resolve_model_source_for_logging(cfg)
    try:
        _log_to_mlflow(
            cfg=cfg,
            model_source=model_source,
            model_source_kind=model_source_kind,
            numeric_metrics=numeric_metrics,
            output_path=Path(output_path),
            index_path=index_path,
        )
    except Exception as exc:
        log_if_rank_zero(
            logger,
            f"MLflow logging failed for retrieval evaluation: {exc}",
            level="warning",
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
