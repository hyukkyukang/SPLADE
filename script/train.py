import logging
import os
from datetime import datetime
from typing import Any

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, Logger, MLFlowLogger
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import TrainDataModule
from src.model.pl_module import SPLADETrainingModule
from src.utils import log_if_rank_zero
from src.utils.logging import (
    get_logger,
    is_rank_zero as is_logging_rank_zero,
    suppress_accumulate_grad_stream_mismatch_warning,
    suppress_urllib3_insecure_request_warning,
)
from src.utils.mlflow_utils import (
    build_mlflow_dataset_input_from_metadata,
    configure_mlflow_tls,
    finish_mlflow_system_metrics_monitor,
    has_logged_mlflow_dataset_inputs,
    has_logged_mlflow_model_outputs,
    log_mlflow_model_output,
    resolve_mlflow_logged_model_name as resolve_mlflow_logged_model_name_impl,
    resolve_mlflow_tags,
    start_mlflow_system_metrics_monitor,
    warn_if_mlflow_gpu_metrics_unavailable,
)
from src.utils.normalize import normalize_optional_bool
from src.utils.progress_bar import StepAwareRichProgressBar
from src.utils.script_setup import (
    configure_default_entrypoint_environment,
    initialize_run,
    normalize_optional_str,
    normalize_tag,
    resolve_trainer_settings,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_default_entrypoint_environment(
    load_env=True,
    set_matmul_precision=True,
)
suppress_urllib3_insecure_request_warning()


def _maybe_mark_ddp_launcher(
    training_cfg: DictConfig, trainer_kwargs: dict[str, Any]
) -> None:
    """Flag the Lightning DDP launcher to silence duplicate logs."""
    strategy_name: str = str(training_cfg.strategy).lower()
    if strategy_name != "ddp":
        return
    rank_env: str | None = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank_env is not None:
        return
    devices: Any = trainer_kwargs.get("devices", 1)
    # Devices can be an int (count) or an explicit device list.
    if isinstance(devices, (list, tuple)):
        device_count: int = len(devices)
    else:
        device_count = int(devices)
    if device_count <= 1:
        return
    os.environ["SPLADE_DDP_LAUNCHER"] = "1"


def _resolve_checkpoint_dir(log_dir: str) -> str:
    """Return a checkpoint directory, suffixing when checkpoints already exist."""
    base_dir: str = os.path.join(log_dir, "checkpoints")
    # Only fork when previous checkpoints exist to avoid overwriting.
    if not os.path.isdir(base_dir):
        return base_dir
    entries: list[str] = os.listdir(base_dir)
    has_checkpoints: bool = any(entry.endswith(".ckpt") for entry in entries)
    if not has_checkpoints:
        return base_dir
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{base_dir}_{timestamp}"


def _normalize_checkpoint_path(path: str) -> str:
    """Normalize a checkpoint path into an absolute filesystem path."""
    return os.path.abspath(os.path.expanduser(path))


def _resolve_checkpoint_paths(training_cfg: DictConfig) -> tuple[str | None, str | None]:
    """Resolve and validate training checkpoint config values."""
    init_checkpoint_path: str | None = normalize_optional_str(
        training_cfg.init_checkpoint_path
    )
    resume_checkpoint_path: str | None = normalize_optional_str(
        training_cfg.resume_checkpoint_path
    )
    if init_checkpoint_path is not None and resume_checkpoint_path is not None:
        raise ValueError(
            "training.init_checkpoint_path and training.resume_checkpoint_path "
            "cannot both be set. Use init_checkpoint_path for weights-only "
            "initialization, or resume_checkpoint_path for full training resume."
        )
    normalized_resume_checkpoint_path: str | None = None
    if resume_checkpoint_path is not None:
        normalized_resume_checkpoint_path = _normalize_checkpoint_path(
            resume_checkpoint_path
        )
        if not os.path.isfile(normalized_resume_checkpoint_path):
            raise FileNotFoundError(
                "training.resume_checkpoint_path does not point to an existing "
                f"checkpoint file: {normalized_resume_checkpoint_path}"
            )
    training_cfg.init_checkpoint_path = init_checkpoint_path
    training_cfg.resume_checkpoint_path = normalized_resume_checkpoint_path
    return init_checkpoint_path, normalized_resume_checkpoint_path


def _resolve_active_checkpoint_dir(log_dir: str, resume_checkpoint_path: str | None) -> str:
    """Choose checkpoint output directory for fresh vs resumed runs."""
    if resume_checkpoint_path is not None:
        return os.path.dirname(resume_checkpoint_path)
    return _resolve_checkpoint_dir(log_dir)


def _validate_expected_train_dataset(cfg: DictConfig) -> None:
    """Validate required dataset constraints declared by a training entry config."""
    expected_dataset_name: str | None = normalize_optional_str(
        cfg.get("required_train_dataset_name")
    )
    if expected_dataset_name is None:
        return
    actual_dataset_name: str = str(cfg.train_dataset.name)
    if actual_dataset_name != expected_dataset_name:
        raise ValueError(
            "This training config requires "
            f"train_dataset={expected_dataset_name}, got {actual_dataset_name}. "
            "Remove dataset@train_dataset override or use the required dataset."
        )


def _build_progress_bar(training_cfg: DictConfig) -> StepAwareRichProgressBar | None:
    """Build a training progress bar callback when enabled."""
    progress_cfg: DictConfig | None = training_cfg.get("progress_bar")
    if progress_cfg is None or not bool(progress_cfg.enabled):
        return None
    refresh_rate_value: float = float(progress_cfg.refresh_rate)
    return StepAwareRichProgressBar(refresh_rate=refresh_rate_value)


def _resolve_mlflow_run_name(training_cfg: DictConfig, tag_value: str | None) -> str:
    """Resolve the MLflow run name with config and tag precedence."""
    mlflow_cfg: DictConfig = training_cfg.mlflow
    explicit_run_name: str | None = normalize_optional_str(mlflow_cfg.get("run_name"))
    if explicit_run_name is not None:
        return explicit_run_name
    if tag_value is not None:
        return tag_value
    return str(training_cfg.name)


def _resolve_mlflow_logged_model_name(model_cfg: DictConfig) -> str:
    """Resolve the MLflow logged-model display name from model config."""
    return resolve_mlflow_logged_model_name_impl(model_cfg)


def _build_mlflow_tags(
    cfg: DictConfig, training_cfg: DictConfig, tag_value: str | None
) -> dict[str, str]:
    """Build MLflow tags from config, adding standard run metadata tags."""
    mlflow_cfg: DictConfig = training_cfg.mlflow
    tags: dict[str, str] = resolve_mlflow_tags(
        mlflow_cfg.get("tags"), field_name="training.mlflow.tags"
    )
    tags.setdefault("training_name", str(training_cfg.name))
    tags.setdefault("log_dir", str(cfg.log_dir))
    if tag_value is not None:
        tags.setdefault("tag", tag_value)
    return tags


def _build_mlflow_dataset_input(
    dataset_cfg: DictConfig, *, dataset_name: str, context: str
) -> Any:
    """Create an MLflow dataset input entity from a dataset config section."""

    def _as_text(key: str, default: str = "") -> str:
        value: str | None = normalize_optional_str(dataset_cfg.get(key))
        if value is None:
            return default
        return value

    resolved_dataset_name: str = _as_text("name", dataset_name)
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


def _log_mlflow_run_datasets_and_model(
    cfg: DictConfig, training_cfg: DictConfig, mlflow_cfg: DictConfig, mlflow_logger: MLFlowLogger
) -> None:
    """Populate MLflow run Inputs/Outputs fields for datasets and model."""
    if not is_logging_rank_zero():
        return

    run_id: str | None = mlflow_logger.run_id
    if run_id is None:
        return

    tracking_uri: str | None = normalize_optional_str(mlflow_cfg.get("tracking_uri"))
    mlflow_client: MlflowClient = MlflowClient(tracking_uri=tracking_uri)
    run = mlflow_client.get_run(run_id)

    if not has_logged_mlflow_dataset_inputs(run):
        dataset_inputs: list[Any] = []
        train_dataset_cfg: DictConfig | None = cfg.get("train_dataset")
        if isinstance(train_dataset_cfg, DictConfig):
            dataset_inputs.append(
                _build_mlflow_dataset_input(
                    train_dataset_cfg, dataset_name="train_dataset", context="training"
                )
            )

        val_dataset_cfg: DictConfig | None = cfg.get("val_dataset")
        if isinstance(val_dataset_cfg, DictConfig):
            dataset_inputs.append(
                _build_mlflow_dataset_input(
                    val_dataset_cfg, dataset_name="val_dataset", context="validation"
                )
            )

        if dataset_inputs:
            mlflow_client.log_inputs(run_id=run_id, datasets=dataset_inputs)

    if has_logged_mlflow_model_outputs(run):
        return

    model_cfg: DictConfig | None = cfg.get("model")
    if not isinstance(model_cfg, DictConfig):
        raise TypeError("cfg.model must be set for MLflow model logging.")

    model_type: str = str(model_cfg.get("type", "splade"))
    model_tags: dict[str, str] = {
        "training_name": str(training_cfg.name),
        "run_id": run_id,
    }
    hf_model_name: str | None = normalize_optional_str(model_cfg.get("huggingface_name"))
    if hf_model_name is not None:
        model_tags["huggingface_name"] = hf_model_name

    logged_model_name: str = _resolve_mlflow_logged_model_name(model_cfg)
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


def _build_lightning_loggers(
    cfg: DictConfig, training_cfg: DictConfig
) -> tuple[list[Logger], str | None]:
    """Build CSV + MLflow logger stack for the training run."""
    tag_value: str | None = normalize_tag(cfg.tag)
    csv_logger: CSVLogger = CSVLogger(save_dir=cfg.log_dir, name="lightning_logs")
    loggers: list[Logger] = [csv_logger]
    # Keep debug runs local-only to preserve current behavior.
    is_debug_tag: bool = tag_value is not None and tag_value.lower() == "debug"
    if is_debug_tag:
        return loggers, None

    mlflow_cfg: DictConfig = training_cfg.mlflow
    mlflow_enabled: bool | None = normalize_optional_bool(mlflow_cfg.get("enabled"))
    if mlflow_enabled is False:
        return loggers, None
    configure_mlflow_tls(mlflow_cfg)
    warn_if_mlflow_gpu_metrics_unavailable(
        mlflow_cfg=mlflow_cfg,
        logger=logger,
        is_logging_rank_zero=is_logging_rank_zero,
    )

    mlflow_run_id: str | None = normalize_optional_str(mlflow_cfg.get("run_id"))
    experiment_name: str = "Train-SPLADE"
    mlflow_logger: MLFlowLogger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=_resolve_mlflow_run_name(training_cfg, tag_value),
        tracking_uri=normalize_optional_str(mlflow_cfg.get("tracking_uri")),
        save_dir=str(mlflow_cfg.save_dir),
        log_model=False,
        prefix=str(mlflow_cfg.get("prefix", "")),
        artifact_location=normalize_optional_str(mlflow_cfg.get("artifact_location")),
        tags=_build_mlflow_tags(cfg, training_cfg, tag_value),
        run_id=mlflow_run_id,
    )
    if mlflow_run_id is None:
        resolved_hparams: Any = OmegaConf.to_container(cfg, resolve=True)
        mlflow_logger.log_hyperparams(resolved_hparams)
    else:
        # Lightning calls `logger.log_hyperparams` again inside `trainer.fit()`.
        # Keep it a no-op for run-id resumes so immutable MLflow params don't fail.
        mlflow_logger.log_hyperparams = (  # type: ignore[method-assign]
            lambda *args, **kwargs: None
        )
        log_if_rank_zero(
            logger,
            "MLflow run_id provided; skipping hyperparameter re-log to avoid "
            "immutable parameter conflicts on resumed runs.",
        )
    managed_system_metrics_run_id: str | None = start_mlflow_system_metrics_monitor(
        mlflow_logger=mlflow_logger,
        mlflow_cfg=mlflow_cfg,
        logger=logger,
        is_logging_rank_zero=is_logging_rank_zero,
    )
    try:
        _log_mlflow_run_datasets_and_model(cfg, training_cfg, mlflow_cfg, mlflow_logger)
    except Exception as exc:
        log_if_rank_zero(
            logger,
            "Failed to log MLflow dataset/model metadata: "
            f"{type(exc).__name__}: {exc}",
            level="warning",
        )
    loggers.append(mlflow_logger)
    return loggers, managed_system_metrics_run_id


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)
    suppress_accumulate_grad_stream_mismatch_warning()

    training_cfg: DictConfig = cfg.training
    trainer_kwargs, precision = resolve_trainer_settings(training_cfg)
    _maybe_mark_ddp_launcher(training_cfg, trainer_kwargs)
    _, resume_checkpoint_path = _resolve_checkpoint_paths(training_cfg)
    _validate_expected_train_dataset(cfg)

    model: SPLADETrainingModule = SPLADETrainingModule(cfg=cfg)
    data_module: TrainDataModule = TrainDataModule(cfg=cfg)

    checkpoint_dir: str = _resolve_active_checkpoint_dir(
        cfg.log_dir, resume_checkpoint_path
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    if resume_checkpoint_path is not None:
        log_if_rank_zero(
            logger,
            "Resuming training from "
            f"{resume_checkpoint_path}; writing new checkpoints to {checkpoint_dir}.",
        )
    elif checkpoint_dir != os.path.join(cfg.log_dir, "checkpoints"):
        log_if_rank_zero(
            logger,
            "Existing checkpoints found; writing new checkpoints to "
            f"{checkpoint_dir}.",
        )
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step{step}-val_MRR_10{val_MRR_10:.4f}",
        monitor="val_MRR_10",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    lightning_loggers, managed_system_metrics_run_id = _build_lightning_loggers(
        cfg, training_cfg
    )

    max_grad_norm_value: float | None = training_cfg.max_grad_norm
    # Lightning disables clipping when the value is <= 0.
    gradient_clip_val: float = (
        max(float(max_grad_norm_value), 0.0) if max_grad_norm_value is not None else 0.0
    )
    progress_bar: StepAwareRichProgressBar | None = _build_progress_bar(training_cfg)
    callbacks: list[Callback] = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
    ]
    if progress_bar is not None:
        callbacks.append(progress_bar)

    trainer: L.Trainer = L.Trainer(
        deterministic=False,
        precision=precision,
        max_steps=cfg.training.max_steps,
        accumulate_grad_batches=cfg.training.grad_accumulation,
        val_check_interval=cfg.training.val_check_interval,
        limit_val_batches=cfg.training.limit_val_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
        default_root_dir=cfg.log_dir,
        logger=lightning_loggers,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        **trainer_kwargs,
    )

    trainer_status: str = "FINISHED"
    try:
        if resume_checkpoint_path is not None:
            trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint_path)
        else:
            trainer.fit(model, datamodule=data_module)
    except Exception:
        trainer_status = "FAILED"
        raise
    finally:
        finish_mlflow_system_metrics_monitor(
            run_id=managed_system_metrics_run_id,
            status=trainer_status,
            logger=logger,
            is_logging_rank_zero=is_logging_rank_zero,
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
