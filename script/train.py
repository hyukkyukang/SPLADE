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
from omegaconf import DictConfig, OmegaConf

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import TrainDataModule
from src.model.pl_module import SPLADETrainingModule
from src.utils import log_if_rank_zero
from src.utils.logging import (
    get_logger,
    suppress_accumulate_grad_stream_mismatch_warning,
    suppress_urllib3_insecure_request_warning,
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


def _build_mlflow_tags(
    cfg: DictConfig, training_cfg: DictConfig, tag_value: str | None
) -> dict[str, str]:
    """Build MLflow tags from config, adding standard run metadata tags."""
    mlflow_cfg: DictConfig = training_cfg.mlflow
    raw_tags: Any = mlflow_cfg.get("tags")
    if raw_tags is None:
        tag_mapping: dict[str, Any] = {}
    else:
        resolved_tags: Any = (
            OmegaConf.to_container(raw_tags, resolve=True)
            if OmegaConf.is_config(raw_tags)
            else raw_tags
        )
        if not isinstance(resolved_tags, dict):
            raise TypeError("training.mlflow.tags must be a mapping.")
        tag_mapping = dict(resolved_tags)

    tags: dict[str, str] = {}
    key: Any
    value: Any
    for key, value in tag_mapping.items():
        if value is None:
            continue
        tags[str(key)] = str(value)
    tags.setdefault("training_name", str(training_cfg.name))
    tags.setdefault("log_dir", str(cfg.log_dir))
    if tag_value is not None:
        tags.setdefault("tag", tag_value)
    return tags


def _configure_mlflow_tls(mlflow_cfg: DictConfig) -> None:
    """Apply MLflow client environment settings from config into env vars."""
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


def _build_lightning_loggers(cfg: DictConfig, training_cfg: DictConfig) -> list[Logger]:
    """Build CSV + MLflow logger stack for the training run."""
    tag_value: str | None = normalize_tag(cfg.tag)
    csv_logger: CSVLogger = CSVLogger(save_dir=cfg.log_dir, name="lightning_logs")
    loggers: list[Logger] = [csv_logger]
    # Keep debug runs local-only to preserve current behavior.
    is_debug_tag: bool = tag_value is not None and tag_value.lower() == "debug"
    if is_debug_tag:
        return loggers

    mlflow_cfg: DictConfig = training_cfg.mlflow
    mlflow_enabled: bool | None = normalize_optional_bool(mlflow_cfg.get("enabled"))
    if mlflow_enabled is False:
        return loggers
    _configure_mlflow_tls(mlflow_cfg)

    mlflow_run_id: str | None = normalize_optional_str(mlflow_cfg.get("run_id"))
    mlflow_logger: MLFlowLogger = MLFlowLogger(
        experiment_name=str(mlflow_cfg.experiment_name),
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
    loggers.append(mlflow_logger)
    return loggers


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

    lightning_loggers: list[Logger] = _build_lightning_loggers(cfg, training_cfg)

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

    if resume_checkpoint_path is not None:
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint_path)
    else:
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
