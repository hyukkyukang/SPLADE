import json
import logging
import os
from copy import deepcopy
from typing import Any

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import TrainDataModule
from src.model.pl_module import SPLADETrainingModule
from src.utils import log_if_rank_zero
from src.utils.checkpoint_compat import add_state_dict_prefix_aliases
from src.utils.logging import get_logger
from src.utils.metrics_io import (
    log_metric_block,
    partition_validation_metrics,
    resolve_training_style_validation_metrics,
    to_jsonable,
)
from src.utils.script_setup import (
    configure_default_entrypoint_environment,
    initialize_run,
    normalize_optional_path,
    resolve_trainer_settings,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_default_entrypoint_environment(
    load_env=True,
    set_matmul_precision=True,
)


def _load_checkpoint_hparams(checkpoint_path: str) -> DictConfig:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    hparams: Any | None = checkpoint.get("hyper_parameters")
    if hparams is None:
        hparams = checkpoint.get("hparams")
    if hparams is None:
        raise ValueError(
            "Checkpoint does not contain hyperparameters. "
            "Cannot reproduce training validation config."
        )
    if isinstance(hparams, DictConfig):
        return hparams
    if isinstance(hparams, dict):
        return OmegaConf.create(hparams)
    if hasattr(hparams, "__dict__"):
        return OmegaConf.create(dict(vars(hparams)))
    raise TypeError("Unsupported checkpoint hyperparameters format.")


def _copy_section_from_checkpoint(
    cfg: DictConfig, checkpoint_cfg: DictConfig, section_name: str
) -> None:
    if section_name not in checkpoint_cfg:
        return
    section_cfg: Any = checkpoint_cfg.get(section_name)

    def _merge_known_keys(base: Any, override: Any) -> Any:
        if isinstance(base, dict) and isinstance(override, dict):
            merged: dict[str, Any] = deepcopy(base)
            for key, override_value in override.items():
                if key not in base:
                    # Drop deprecated/unknown keys from old checkpoints.
                    continue
                merged[key] = _merge_known_keys(base[key], override_value)
            return merged
        return deepcopy(override)

    merged_section: Any
    if section_name in cfg and cfg.get(section_name) is not None:
        base_container: Any = OmegaConf.to_container(cfg.get(section_name), resolve=False)
        override_container: Any = OmegaConf.to_container(section_cfg, resolve=False)
        merged_section = _merge_known_keys(base_container, override_container)
    else:
        merged_section = OmegaConf.to_container(section_cfg, resolve=False)
    with open_dict(cfg):
        cfg[section_name] = OmegaConf.create(merged_section)


def _apply_checkpoint_run_config(cfg: DictConfig, checkpoint_path: str) -> DictConfig:
    checkpoint_cfg: DictConfig = _load_checkpoint_hparams(checkpoint_path)
    for section_name in ("model", "train_dataset", "val_dataset", "training", "nanobeir"):
        _copy_section_from_checkpoint(cfg, checkpoint_cfg, section_name)

    # Always validate the explicit checkpoint path passed to this script.
    with open_dict(cfg):
        cfg.training.init_checkpoint_path = None
        cfg.training.resume_checkpoint_path = None
    return cfg


def _resolve_output_json_path(cfg: DictConfig) -> str:
    output_path: str | None = normalize_optional_path(cfg.validation.output_json_path)
    if output_path is None:
        return os.path.join(cfg.log_dir, "validation_metrics.json")
    return output_path


def _run_consistency_checks(reranking_metrics: dict[str, float]) -> None:
    required_prefixes: tuple[str, ...] = (
        "val_MRR_",
        "val_nDCG_",
        "val_Recall_",
    )
    if "val_loss" not in reranking_metrics:
        log_if_rank_zero(
            logger,
            "Validation consistency check: missing val_loss in reranking metrics.",
            level="warning",
        )
    metric_prefix: str
    for metric_prefix in required_prefixes:
        has_prefix_metric: bool = any(
            metric_name.startswith(metric_prefix)
            for metric_name in reranking_metrics.keys()
        )
        if has_prefix_metric:
            continue
        log_if_rank_zero(
            logger,
            "Validation consistency check: missing reranking metric prefix "
            f"{metric_prefix!r}.",
            level="warning",
        )


def _load_checkpoint_state_with_compat(
    model: SPLADETrainingModule, checkpoint_path: str
) -> tuple[list[str], list[str]]:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    state_dict: dict[str, torch.Tensor] = checkpoint["state_dict"]
    remapped_state_dict: dict[str, torch.Tensor] = add_state_dict_prefix_aliases(
        state_dict,
        aliases=(
            ("model._orig_mod.module.", "model."),
            ("model._orig_mod.", "model."),
        ),
    )

    load_result = model.load_state_dict(remapped_state_dict, strict=False)
    return load_result.missing_keys, load_result.unexpected_keys


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="validation")
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)

    checkpoint_path: str | None = normalize_optional_path(cfg.validation.checkpoint_path)
    if checkpoint_path is None:
        raise ValueError("validation.checkpoint_path must be set.")

    if bool(cfg.validation.use_checkpoint_config):
        cfg = _apply_checkpoint_run_config(cfg, checkpoint_path)
        log_if_rank_zero(
            logger,
            "Loaded model/data/training sections from checkpoint hyperparameters.",
        )

    if "nanobeir" in cfg and not bool(cfg.validation.include_nanobeir):
        with open_dict(cfg):
            cfg.nanobeir.enabled = False

    model: SPLADETrainingModule = SPLADETrainingModule(cfg=cfg)
    data_module: TrainDataModule = TrainDataModule(cfg=cfg)

    trainer_kwargs, precision = resolve_trainer_settings(cfg.training)
    trainer: L.Trainer = L.Trainer(
        deterministic=False,
        precision=precision,
        default_root_dir=cfg.log_dir,
        logger=False,
        enable_checkpointing=False,
        **trainer_kwargs,
    )

    missing_keys, unexpected_keys = _load_checkpoint_state_with_compat(
        model=model, checkpoint_path=checkpoint_path
    )
    if missing_keys:
        log_if_rank_zero(
            logger,
            f"Checkpoint compatibility load missing_keys={len(missing_keys)}",
            level="warning",
        )
    if unexpected_keys:
        log_if_rank_zero(
            logger,
            f"Checkpoint compatibility load unexpected_keys={len(unexpected_keys)}",
            level="warning",
        )

    results: list[dict[str, Any]] = trainer.validate(
        model=model, datamodule=data_module, ckpt_path=None
    )
    if not results:
        log_if_rank_zero(logger, "Validation returned no metrics.", level="warning")
        return

    validation_metrics: dict[str, float]
    raw_validate_metrics: dict[str, float]
    callback_metrics: dict[str, float]
    (
        validation_metrics,
        raw_validate_metrics,
        callback_metrics,
    ) = resolve_training_style_validation_metrics(
        validate_results=results,
        trainer=trainer,
        logger=logger,
    )

    reranking_metrics: dict[str, float]
    nanobeir_metrics: dict[str, float]
    reranking_metrics, nanobeir_metrics = partition_validation_metrics(
        validation_metrics
    )
    _run_consistency_checks(reranking_metrics)

    log_metric_block(
        logger=logger,
        title="MSMARCO Reranking Validation Metrics:",
        metrics=reranking_metrics,
    )
    log_metric_block(
        logger=logger,
        title="NanoBEIR Validation Metrics:",
        metrics=nanobeir_metrics,
    )

    output_payload: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "reranking_results": to_jsonable(reranking_metrics),
        "nanobeir_results": to_jsonable(nanobeir_metrics),
        "resolved_validation_metrics": to_jsonable(validation_metrics),
        "raw_validate_metrics": to_jsonable(raw_validate_metrics),
        "results": to_jsonable(results),
        "callback_metrics": to_jsonable(callback_metrics),
    }
    output_json_path: str = _resolve_output_json_path(cfg)
    output_dir: str = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(output_payload, json_file, indent=2)
    log_if_rank_zero(logger, f"Saved validation metrics to {output_json_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
