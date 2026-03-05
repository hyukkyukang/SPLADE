import logging
import os
from collections.abc import Callable
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities import Dataset as MlflowDataset
from mlflow.entities import DatasetInput, InputTag
from mlflow.entities.logged_model_output import LoggedModelOutput
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from src.utils.logging import log_if_rank_zero
from src.utils.normalize import normalize_optional_bool, normalize_optional_str


def resolve_mlflow_tags(raw_tags: Any, *, field_name: str = "mlflow.tags") -> dict[str, str]:
    """Resolve MLflow tags from optional config/raw mappings."""
    if raw_tags is None:
        return {}
    resolved_tags: Any = (
        OmegaConf.to_container(raw_tags, resolve=True)
        if OmegaConf.is_config(raw_tags)
        else raw_tags
    )
    if not isinstance(resolved_tags, dict):
        raise TypeError(f"{field_name} must be a mapping.")
    tags: dict[str, str] = {}
    key: Any
    value: Any
    for key, value in resolved_tags.items():
        if value is None:
            continue
        tags[str(key)] = str(value)
    return tags


def sanitize_mlflow_metric_name(metric_name: str) -> str:
    """Return a metric name that only contains MLflow-safe characters."""
    allowed_chars: set[str] = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_-./ "
    )
    safe_name: str = "".join(
        char if char in allowed_chars else "_" for char in str(metric_name)
    ).strip()
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    if not safe_name:
        return "metric"
    return safe_name


def has_logged_mlflow_dataset_inputs(run: Any) -> bool:
    """Check whether an MLflow run already has dataset inputs."""
    run_inputs: Any = getattr(run, "inputs", None)
    if run_inputs is None:
        return False
    dataset_inputs: Any = getattr(run_inputs, "dataset_inputs", None)
    return bool(dataset_inputs)


def has_logged_mlflow_model_outputs(run: Any) -> bool:
    """Check whether an MLflow run already has model outputs."""
    run_outputs: Any = getattr(run, "outputs", None)
    if run_outputs is None:
        return False
    model_outputs: Any = getattr(run_outputs, "model_outputs", None)
    return bool(model_outputs)


def build_mlflow_dataset_input_from_metadata(
    *,
    dataset_name: str,
    context: str,
    metadata: dict[str, Any],
    source: str | None = None,
) -> DatasetInput:
    """Create a DatasetInput entity from a metadata payload."""
    dataset = mlflow.data.from_pandas(
        pd.DataFrame([metadata]), name=dataset_name, source=source
    )
    dataset_dict: dict[str, Any] = dataset.to_dict()
    dataset_entity = MlflowDataset(
        name=str(dataset_dict["name"]),
        digest=str(dataset_dict["digest"]),
        source_type=str(dataset_dict["source_type"]),
        source=str(dataset_dict["source"]),
        schema=(
            str(dataset_dict["schema"])
            if dataset_dict.get("schema") is not None
            else None
        ),
        profile=(
            str(dataset_dict["profile"])
            if dataset_dict.get("profile") is not None
            else None
        ),
    )
    return DatasetInput(
        dataset=dataset_entity,
        tags=[InputTag("mlflow.data.context", context)],
    )


def log_mlflow_model_output(
    *,
    mlflow_client: MlflowClient,
    run: Any,
    run_id: str,
    logged_model_name: str,
    model_type: str,
    model_tags: dict[str, str],
    tracking_uri: str | None,
    step: int = 0,
) -> None:
    """Create (or reuse fallback) logged model and attach it to run outputs."""
    create_logged_model = getattr(mlflow_client, "create_logged_model", None)
    if callable(create_logged_model):
        logged_model = create_logged_model(
            experiment_id=str(run.info.experiment_id),
            name=logged_model_name,
            source_run_id=run_id,
            tags=model_tags,
            model_type=model_type,
        )
    else:
        # Fallback for MLflow versions without `MlflowClient.create_logged_model`.
        original_tracking_uri: str | None = mlflow.get_tracking_uri()
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        try:
            logged_model = mlflow.create_external_model(
                name=logged_model_name,
                source_run_id=run_id,
                tags=model_tags,
                model_type=model_type,
                experiment_id=str(run.info.experiment_id),
            )
        finally:
            if tracking_uri is not None:
                mlflow.set_tracking_uri(original_tracking_uri)

    mlflow_client.log_outputs(
        run_id=run_id,
        models=[LoggedModelOutput(model_id=str(logged_model.model_id), step=int(step))],
    )


def configure_mlflow_tls(
    mlflow_cfg: Any,
    *,
    sampling_interval_field: str = "training.mlflow.system_metrics_sampling_interval",
    samples_before_logging_field: str = "training.mlflow.system_metrics_samples_before_logging",
) -> None:
    """Apply MLflow client/TLS settings from config into environment variables."""
    insecure_tls: bool | None = normalize_optional_bool(mlflow_cfg.get("insecure_tls"))
    if insecure_tls is not None:
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = (
            "true" if insecure_tls else "false"
        )

    server_cert_path: str | None = normalize_optional_str(
        mlflow_cfg.get("server_cert_path")
    )
    if server_cert_path is not None:
        os.environ["MLFLOW_TRACKING_SERVER_CERT_PATH"] = server_cert_path

    client_cert_path: str | None = normalize_optional_str(
        mlflow_cfg.get("client_cert_path")
    )
    if client_cert_path is not None:
        os.environ["MLFLOW_TRACKING_CLIENT_CERT_PATH"] = client_cert_path

    system_metrics_enabled: bool | None = normalize_optional_bool(
        mlflow_cfg.get("system_metrics_enabled")
    )
    if system_metrics_enabled is not None:
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = (
            "true" if system_metrics_enabled else "false"
        )

    def _resolve_positive_int(value: Any, *, field_name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be an integer, got bool.")
        try:
            parsed_value: int = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{field_name} must be an integer, got {type(value).__name__}."
            ) from exc
        if parsed_value <= 0:
            raise ValueError(f"{field_name} must be > 0, got {parsed_value}.")
        return parsed_value

    system_metrics_sampling_interval: int | None = _resolve_positive_int(
        mlflow_cfg.get("system_metrics_sampling_interval"),
        field_name=sampling_interval_field,
    )
    if system_metrics_sampling_interval is not None:
        os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = str(
            system_metrics_sampling_interval
        )

    system_metrics_samples_before_logging: int | None = _resolve_positive_int(
        mlflow_cfg.get("system_metrics_samples_before_logging"),
        field_name=samples_before_logging_field,
    )
    if system_metrics_samples_before_logging is not None:
        os.environ["MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING"] = str(
            system_metrics_samples_before_logging
        )


def is_truthy_env_flag(value: str | None) -> bool:
    """Parse common truthy environment flag strings."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def start_mlflow_system_metrics_monitor(
    *,
    mlflow_logger: Any,
    mlflow_cfg: Any,
    logger: logging.Logger,
    is_logging_rank_zero: Callable[[], bool],
) -> str | None:
    """Attach MLflow fluent system-metrics monitoring to the logger run."""
    if not is_logging_rank_zero():
        return None

    system_metrics_enabled: bool | None = normalize_optional_bool(
        mlflow_cfg.get("system_metrics_enabled")
    )
    if system_metrics_enabled is None and not is_truthy_env_flag(
        os.environ.get("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING")
    ):
        return None
    if system_metrics_enabled is False:
        return None

    run_id: str | None = getattr(mlflow_logger, "run_id", None)
    if run_id is None:
        return None

    active_run = mlflow.active_run()
    if active_run is not None:
        active_run_id: str = active_run.info.run_id
        if active_run_id != run_id:
            log_if_rank_zero(
                logger,
                "An active MLflow run is already set "
                f"({active_run_id}); skipping system metrics monitor for {run_id}.",
                level="warning",
            )
            return None
        return run_id

    mlflow.start_run(run_id=run_id, log_system_metrics=system_metrics_enabled)
    return run_id


def finish_mlflow_system_metrics_monitor(
    *,
    run_id: str | None,
    status: str,
    logger: logging.Logger,
    is_logging_rank_zero: Callable[[], bool],
) -> None:
    """Finish a fluent MLflow run opened for system-metrics monitoring."""
    if run_id is None or not is_logging_rank_zero():
        return

    active_run = mlflow.active_run()
    if active_run is None:
        return

    active_run_id: str = active_run.info.run_id
    if active_run_id != run_id:
        log_if_rank_zero(
            logger,
            "Active MLflow run changed before shutdown "
            f"({active_run_id} != {run_id}); skipping mlflow.end_run().",
            level="warning",
        )
        return
    mlflow.end_run(status=status)
