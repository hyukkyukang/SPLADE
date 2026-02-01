import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

from config.path import ABS_CONFIG_DIR
from src.utils import log_if_rank_zero, set_seed
from src.utils.logging import get_logger, setup_tqdm_friendly_logging
from src.utils.script_setup import configure_script_environment
from src.utils.model_utils import build_splade_model, load_splade_checkpoint
from src.utils.sparse_encoder import (
    build_doc_only_sparse_encoder_adapter,
    build_sparse_encoder_from_checkpoint,
    resolve_nanobeir_compatibility,
)

logger: logging.Logger = get_logger("scripts.evaluate_nanobeir", __file__)

configure_script_environment(
    load_env=False,
    set_tokenizers_parallelism=True,
    set_matmul_precision=True,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


def _resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve the torch device from the testing config."""
    use_cpu: bool = bool(cfg.testing.use_cpu)
    if use_cpu:
        return torch.device("cpu")

    device_id: int | None = cfg.testing.device_id
    if torch.cuda.is_available():
        if device_id is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{int(device_id)}")
    return torch.device("cpu")


def _load_checkpoint_hparams(checkpoint_path: str) -> DictConfig | None:
    """Load the config stored in a Lightning checkpoint when available."""
    # Load the checkpoint on CPU to avoid device mismatches.
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    # Lightning stores hyperparameters under standardized keys.
    hparams: Any | None = checkpoint.get("hyper_parameters")
    if hparams is None:
        hparams = checkpoint.get("hparams")
    if hparams is None:
        return None
    if isinstance(hparams, DictConfig):
        return hparams
    if isinstance(hparams, dict):
        return OmegaConf.create(hparams)
    if hasattr(hparams, "__dict__"):
        # Preserve attribute-based configs by expanding their dict representation.
        hparams_dict: dict[str, Any] = dict(vars(hparams))
        return OmegaConf.create(hparams_dict)
    return None


def _resolve_checkpoint_log_dir(checkpoint_path: str) -> str | None:
    """Infer the training log directory from a checkpoint path."""
    checkpoint_dir: str = os.path.dirname(checkpoint_path)
    checkpoint_dir_name: str = os.path.basename(checkpoint_dir)
    # Lightning checkpoints live under <log_dir>/checkpoints* directories.
    if checkpoint_dir_name.startswith("checkpoints"):
        return os.path.dirname(checkpoint_dir)
    return None


def _load_yaml_config(config_path: str) -> DictConfig | None:
    """Load a YAML config file into a DictConfig."""
    if not os.path.isfile(config_path):
        return None
    try:
        loaded_cfg: DictConfig = OmegaConf.load(config_path)
    except Exception as exc:
        # Surface config loading errors but keep evaluation running.
        log_if_rank_zero(
            logger,
            f"Failed to load config at {config_path}: {exc}",
            level="warning",
        )
        return None
    return loaded_cfg


def _extract_model_config(root_cfg: DictConfig) -> DictConfig | None:
    """Extract the model config from a training or W&B config."""
    model_section: Any | None = getattr(root_cfg, "model", None)
    if model_section is None:
        return None
    model_cfg: Any = model_section
    # W&B configs store the actual values under a `value` key.
    if OmegaConf.is_config(model_cfg) and "value" in model_cfg:
        model_cfg = model_cfg.get("value")
    if OmegaConf.is_config(model_cfg):
        return model_cfg
    if isinstance(model_cfg, dict):
        return OmegaConf.create(model_cfg)
    return None


def _find_latest_wandb_config_path(log_dir: str) -> str | None:
    """Find the most recent W&B config.yaml under a log directory."""
    wandb_root: Path = Path(log_dir) / "wandb"
    if not wandb_root.is_dir():
        return None
    # Search W&B run directories for saved config.yaml files.
    config_paths: list[Path] = list(wandb_root.rglob("config.yaml"))
    if not config_paths:
        return None
    latest_path: Path = max(config_paths, key=lambda path: path.stat().st_mtime)
    return str(latest_path)


def _load_model_config_from_log_dir(log_dir: str) -> DictConfig | None:
    """Resolve model config from Hydra or W&B files in the log directory."""
    hydra_config_path: str = os.path.join(log_dir, ".hydra", "config.yaml")
    hydra_cfg: DictConfig | None = _load_yaml_config(hydra_config_path)
    if hydra_cfg is not None:
        hydra_model_cfg: DictConfig | None = _extract_model_config(hydra_cfg)
        if hydra_model_cfg is not None:
            log_if_rank_zero(
                logger, f"Using model config from {hydra_config_path}."
            )
            return hydra_model_cfg

    wandb_config_path: str | None = _find_latest_wandb_config_path(log_dir)
    if wandb_config_path is None:
        return None
    wandb_cfg: DictConfig | None = _load_yaml_config(wandb_config_path)
    if wandb_cfg is None:
        return None
    wandb_model_cfg: DictConfig | None = _extract_model_config(wandb_cfg)
    if wandb_model_cfg is None:
        return None
    log_if_rank_zero(logger, f"Using model config from {wandb_config_path}.")
    return wandb_model_cfg


def _apply_checkpoint_model_config(
    cfg: DictConfig, checkpoint_path: str
) -> DictConfig:
    """Override the model config with the checkpoint config when present."""
    use_checkpoint_config: bool = bool(
        getattr(cfg.testing, "use_checkpoint_config", True)
    )
    if not use_checkpoint_config:
        return cfg

    model_cfg: DictConfig | None = None
    model_cfg_source: str | None = None

    checkpoint_cfg: DictConfig | None = _load_checkpoint_hparams(checkpoint_path)
    if checkpoint_cfg is not None:
        # Prefer the exact config saved alongside the checkpoint.
        model_cfg = _extract_model_config(checkpoint_cfg)
        if model_cfg is not None:
            model_cfg_source = "checkpoint"

    if model_cfg is None:
        # Fall back to the training log directory when checkpoints lack hparams.
        log_dir: str | None = _resolve_checkpoint_log_dir(checkpoint_path)
        if log_dir is not None:
            model_cfg = _load_model_config_from_log_dir(log_dir)
            if model_cfg is not None:
                model_cfg_source = log_dir

    if model_cfg is None:
        log_if_rank_zero(
            logger,
            "Checkpoint config not found; using evaluation config.",
            level="warning",
        )
        return cfg

    merged_cfg: DictConfig = OmegaConf.merge(cfg, {"model": model_cfg})
    model_name: str = str(merged_cfg.model.name)
    hf_name: str = str(merged_cfg.model.huggingface_name)
    source_label: str = model_cfg_source or "unknown"
    log_if_rank_zero(logger, f"Using {source_label} model config: {model_name} ({hf_name}).")
    return merged_cfg


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate_nanobeir"
)
def main(cfg: DictConfig) -> None:
    setup_tqdm_friendly_logging()
    os.makedirs(cfg.log_dir, exist_ok=True)
    set_seed(cfg.seed)
    log_if_rank_zero(logger, f"Random seed set to: {cfg.seed}")

    checkpoint_path_value: str | None = cfg.testing.checkpoint_path
    if not checkpoint_path_value:
        raise ValueError("testing.checkpoint_path must be set for NanoBEIR eval.")
    checkpoint_path: str = str(checkpoint_path_value)
    cfg: DictConfig = _apply_checkpoint_model_config(cfg, checkpoint_path)

    # Normalize dataset names to strings for the evaluator.
    dataset_names: list[str] = []
    dataset_name: str
    for dataset_name in cfg.nanobeir.datasets:
        dataset_names.append(str(dataset_name))
    if not dataset_names:
        raise ValueError("nanobeir.datasets must contain at least one dataset name.")

    batch_size: int = int(cfg.nanobeir.batch_size)
    save_json: bool = bool(cfg.nanobeir.save_json)

    # Resolve device before model instantiation to keep tensors aligned.
    device: torch.device = _resolve_device(cfg)
    log_if_rank_zero(logger, f"Using device: {device}")

    doc_only_enabled: bool = bool(getattr(cfg.model, "doc_only", False))
    if doc_only_enabled:
        model: Any = build_splade_model(cfg, use_cpu=cfg.testing.use_cpu)
        missing: list[str]
        unexpected: list[str]
        missing, unexpected = load_splade_checkpoint(model, checkpoint_path)
        log_if_rank_zero(
            logger,
            f"Loaded checkpoint. Missing: {len(missing)}, unexpected: {len(unexpected)}",
        )
        sparse_encoder: Any = build_doc_only_sparse_encoder_adapter(
            cfg=cfg,
            model=model,
            device=device,
            batch_size=batch_size,
        )
    else:
        compatible: bool
        reason: str | None
        compatible, reason = resolve_nanobeir_compatibility(cfg)
        if not compatible:
            raise ValueError(f"NanoBEIR evaluation incompatible: {reason}")
        # Convert the Lightning checkpoint into a SparseEncoder for NanoBEIR.
        sparse_encoder = build_sparse_encoder_from_checkpoint(
            cfg=cfg, checkpoint_path=checkpoint_path, device=device
        )

    # Run NanoBEIR proxy evaluation and collect metrics.
    evaluator: SparseNanoBEIREvaluator = SparseNanoBEIREvaluator(
        dataset_names=dataset_names,
        batch_size=batch_size,
    )

    results: dict[str, Any] = evaluator(sparse_encoder)
    metric_name: str
    metric_value: Any
    for metric_name, metric_value in results.items():
        log_if_rank_zero(logger, f"NanoBEIR {metric_name}: {metric_value}")

    if save_json:
        output_path: str = os.path.join(cfg.log_dir, "nanobeir_metrics.json")
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(results, json_file, indent=2)
        log_if_rank_zero(logger, f"Saved NanoBEIR metrics to {output_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
