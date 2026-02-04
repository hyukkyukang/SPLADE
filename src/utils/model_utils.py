import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import log_if_rank_zero


def resolve_model_dtype(dtype_name: str, use_cpu: bool) -> torch.dtype | None:
    """Resolve model dtype with a safe CPU fallback."""
    normalized: str = str(dtype_name).lower()
    dtype_map: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    resolved: torch.dtype | None = dtype_map.get(normalized)
    if use_cpu and resolved in {torch.float16, torch.bfloat16}:
        # CPU kernels typically expect float32 for stability.
        return torch.float32
    return resolved


def build_splade_model(cfg: DictConfig, *, use_cpu: bool) -> SpladeModel:
    """Build a SPLADE model from config with dtype handling."""
    dtype: torch.dtype | None = resolve_model_dtype(cfg.model.dtype, use_cpu)
    # Detect SPLADE-doc mode via explicit flag or config name suffix.
    name_value: str = str(cfg.model.name).lower()
    doc_only_flag: bool = bool(cfg.model.doc_only)
    doc_only: bool = doc_only_flag or name_value.endswith(("_doc", "-doc"))
    return SpladeModel(
        model_name=cfg.model.huggingface_name,
        query_pooling=cfg.model.query_pooling,
        doc_pooling=cfg.model.doc_pooling,
        sparse_activation=cfg.model.sparse_activation,
        attn_implementation=cfg.model.attn_implementation,
        dtype=dtype,
        normalize=cfg.model.normalize,
        doc_only=doc_only,
    )


def _strip_checkpoint_prefix(key: str, prefixes: Sequence[str]) -> str | None:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return None


def _strip_compiled_wrapper_segments(key: str) -> str:
    """Remove torch.compile wrapper segments from a state dict key."""
    # Handle nested optimized modules like encoder._orig_mod.(module.)*
    cleaned: str = key.replace("._orig_mod.module.", ".")
    cleaned = cleaned.replace("._orig_mod.", ".")
    return cleaned


def _expand_splade_encoder_aliases(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Ensure encoder weights are available under wrapper aliases."""
    prefixes: tuple[str, ...] = (
        "encoder.",
        "_query_encoder_wrapper.encoder.",
        "_doc_encoder_wrapper.encoder.",
        "_query_encoder_fn.encoder.",
        "_doc_encoder_fn.encoder.",
    )
    suffix_values: dict[str, Any] = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                suffix: str = key[len(prefix) :]
                if suffix not in suffix_values:
                    suffix_values[suffix] = value
                break
    if not suffix_values:
        return state_dict
    expanded: dict[str, Any] = dict(state_dict)
    for suffix, value in suffix_values.items():
        for prefix in prefixes:
            alias_key: str = prefix + suffix
            if alias_key not in expanded:
                expanded[alias_key] = value
    return expanded


def _normalize_checkpoint_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes: tuple[str, ...] = (
        "model._orig_mod.module.",
        "model._orig_mod.",
        "model.module.",
        "model.",
    )
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        stripped = _strip_checkpoint_prefix(key, prefixes)
        if stripped is None:
            continue
        normalized[_strip_compiled_wrapper_segments(stripped)] = value
    if normalized:
        return _expand_splade_encoder_aliases(normalized)

    # Fall back to raw model state_dict, but clean compiled wrapper segments.
    cleaned: dict[str, Any] = {}
    changed: bool = False
    for key, value in state_dict.items():
        cleaned_key: str = _strip_compiled_wrapper_segments(key)
        if cleaned_key != key:
            changed = True
        cleaned[cleaned_key] = value
    cleaned_or_original: dict[str, Any] = cleaned if changed else state_dict
    return _expand_splade_encoder_aliases(cleaned_or_original)


def load_splade_checkpoint(
    model: SpladeModel, checkpoint_path: str, logger: logging.Logger | None = None
) -> tuple[list[str], list[str]]:
    """Load a Lightning checkpoint into the SPLADE encoder."""
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    state_dict: dict[str, Any] = checkpoint.get("state_dict", checkpoint)
    state_dict_to_load: dict[str, Any] = _normalize_checkpoint_state_dict(state_dict)
    incompatible: Any = model.load_state_dict(state_dict_to_load, strict=False)
    missing_keys: list[str] = list(incompatible.missing_keys)
    unexpected_keys: list[str] = list(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        error_logger = logger or logging.getLogger(__name__)
        error_logger.error(
            "Checkpoint parameter mismatch: missing=%d, unexpected=%d.",
            len(missing_keys),
            len(unexpected_keys),
        )
        if missing_keys:
            error_logger.error("Missing keys (sample): %s", ", ".join(missing_keys[:10]))
        if unexpected_keys:
            error_logger.error(
                "Unexpected keys (sample): %s", ", ".join(unexpected_keys[:10])
            )
        raise RuntimeError(
            "Checkpoint parameters do not match the current model definition."
        )
    return missing_keys, unexpected_keys


def _load_checkpoint_hparams(checkpoint_path: str) -> DictConfig | None:
    """Load checkpoint hyperparameters when available."""
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
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
        return OmegaConf.create(dict(vars(hparams)))
    return None


def _extract_model_config(root_cfg: DictConfig) -> DictConfig | None:
    """Extract model config from a root Hydra/W&B config."""
    if "model" not in root_cfg:
        return None
    model_cfg: Any = root_cfg.model
    if OmegaConf.is_config(model_cfg) and "value" in model_cfg:
        model_cfg = model_cfg.get("value")
    if OmegaConf.is_config(model_cfg):
        return model_cfg
    if isinstance(model_cfg, dict):
        return OmegaConf.create(model_cfg)
    return None


def _format_model_value(value: Any) -> str:
    """Normalize model config values for readable logs."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return repr(value)


def _format_model_config_block(
    title: str, model_cfg: DictConfig | None, keys: Sequence[str]
) -> str:
    """Render a multi-line, ordered model config summary."""
    if model_cfg is None:
        return f"{title}: missing"
    lines: list[str] = [f"{title}:"]
    for key in keys:
        value: Any = model_cfg.get(key) if key in model_cfg else None
        lines.append(f"  {key}: {_format_model_value(value)}")
    return "\n".join(lines)


def resolve_tagged_output_dir(
    base_dir: str | Path, *, model_name: str, tag: object | None
) -> Path:
    """Resolve output dir as {base_dir}/{model_name}/{tag?}."""
    base_path: Path = Path(base_dir)
    model_segment: str = str(model_name).strip() or "model"
    tag_value: str | None = None
    if tag is not None:
        normalized: str = str(tag).strip()
        if normalized:
            tag_value = normalized
    if tag_value is None:
        return base_path / model_segment
    return base_path / model_segment / tag_value


def apply_checkpoint_model_config(
    cfg: DictConfig,
    checkpoint_path: str | None,
    *,
    logger: logging.Logger,
    exclude_keys: Iterable[str] = (
        "encode_path",
        "index_path",
        "encode_dir",
        "index_dir",
        "sparse_top_k",
        "sparse_min_weight",
    ),
) -> DictConfig:
    """Override cfg.model with checkpoint model config, preserving runtime fields."""
    if not checkpoint_path:
        return cfg

    summary_keys: tuple[str, ...] = (
        "name",
        "huggingface_name",
        "dtype",
        "doc_only",
        "query_pooling",
        "doc_pooling",
        "sparse_activation",
        "normalize",
        "max_input_length",
    )
    runtime_model_cfg: DictConfig | None = cfg.model if "model" in cfg else None
    log_if_rank_zero(
        logger, f"Loading checkpoint model config from {checkpoint_path}."
    )

    hparams_cfg: DictConfig | None = _load_checkpoint_hparams(checkpoint_path)
    if hparams_cfg is None:
        log_if_rank_zero(
            logger,
            "Checkpoint lacks hyperparameters; keeping runtime model config.",
            level="warning",
        )
        return cfg

    checkpoint_model_cfg: DictConfig | None = _extract_model_config(hparams_cfg)
    if checkpoint_model_cfg is None:
        log_if_rank_zero(
            logger,
            "Checkpoint lacks model config; keeping runtime model config.",
            level="warning",
        )
        return cfg

    log_if_rank_zero(
        logger,
        _format_model_config_block(
            "Checkpoint model config",
            checkpoint_model_cfg,
            summary_keys,
        ),
    )

    exclude_set: set[str] = {str(key) for key in exclude_keys}
    model_dict: Any = OmegaConf.to_container(checkpoint_model_cfg, resolve=True)
    if not isinstance(model_dict, dict):
        log_if_rank_zero(
            logger,
            "Checkpoint model config is not a mapping; keeping runtime config.",
            level="warning",
        )
        return cfg

    filtered_model_cfg: DictConfig = OmegaConf.create(
        {key: value for key, value in model_dict.items() if key not in exclude_set}
    )
    merged_cfg: DictConfig = OmegaConf.merge(cfg, {"model": filtered_model_cfg})

    runtime_keys: set[str] = (
        {str(key) for key in runtime_model_cfg.keys()}
        if runtime_model_cfg is not None
        else set()
    )
    filtered_keys: set[str] = {str(key) for key in filtered_model_cfg.keys()}
    kept_keys: list[str] = [
        key for key in summary_keys if key in runtime_keys and key not in filtered_keys
    ]
    if kept_keys:
        log_if_rank_zero(
            logger,
            _format_model_config_block(
                "Runtime model config (not overridden)",
                merged_cfg.model,
                kept_keys,
            ),
        )
    else:
        log_if_rank_zero(logger, "Runtime model config (not overridden): none")
    return merged_cfg
