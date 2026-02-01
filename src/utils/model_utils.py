from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from src.model.retriever.sparse.neural.splade import SpladeModel


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
    doc_only_flag: bool = bool(getattr(cfg.model, "doc_only", False))
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


def load_splade_checkpoint(
    model: SpladeModel, checkpoint_path: str
) -> tuple[list[str], list[str]]:
    """Load a Lightning checkpoint into the SPLADE encoder."""
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    state_dict: dict[str, Any] = checkpoint.get("state_dict", checkpoint)
    filtered: dict[str, Any] = {
        key.replace("model.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    # If no prefixed keys exist, assume the state_dict is already a model dict.
    state_dict_to_load: dict[str, Any] = filtered or state_dict
    incompatible: Any = model.load_state_dict(state_dict_to_load, strict=False)
    missing_keys: list[str] = list(incompatible.missing_keys)
    unexpected_keys: list[str] = list(incompatible.unexpected_keys)
    return missing_keys, unexpected_keys
