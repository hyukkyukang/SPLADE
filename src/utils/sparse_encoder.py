from __future__ import annotations

import logging
from typing import Any, Iterable

import torch
from omegaconf import DictConfig
from sentence_transformers import SparseEncoder
from sentence_transformers.models import Normalize
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from src.utils.model_utils import resolve_model_dtype

logger: logging.Logger = logging.getLogger("src.utils.sparse_encoder")


def _strip_prefix(value: str, prefixes: Iterable[str]) -> str | None:
    """Return value with the first matching prefix stripped."""
    prefix: str
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _extract_mlm_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extract MLM-only weights from a Lightning checkpoint state dict."""
    prefixes: list[str] = [
        "model.encoder._orig_mod.mlm.",
        "encoder._orig_mod.mlm.",
        "model.encoder.mlm.",
        "encoder.mlm.",
        "model.mlm.",
        "mlm.",
    ]
    mlm_state: dict[str, torch.Tensor] = {}
    key: str
    value: torch.Tensor
    for key, value in state_dict.items():
        stripped_key: str | None = _strip_prefix(key, prefixes)
        if stripped_key is None:
            continue
        mlm_state[stripped_key] = value

    if not mlm_state:
        raise ValueError(
            "No MLM weights found in checkpoint. Expected keys starting with "
            "'model.encoder.mlm.' or 'encoder.mlm.'."
        )
    return mlm_state


def _load_mlm_transformer(
    cfg: DictConfig,
    checkpoint_path: str,
) -> MLMTransformer:
    """Load an MLMTransformer and override weights from Lightning checkpoint."""
    # Resolve dtype and attention implementation to match the training setup.
    dtype: torch.dtype | None = resolve_model_dtype(
        str(cfg.model.dtype), bool(cfg.testing.use_cpu)
    )
    model_args: dict[str, Any] = {}
    attn_implementation: str | None = getattr(cfg.model, "attn_implementation", None)
    if attn_implementation:
        model_args["attn_implementation"] = attn_implementation
    if dtype is not None:
        model_args["torch_dtype"] = dtype

    max_input_length: int = int(getattr(cfg.model, "max_input_length", 256))
    tokenizer_args: dict[str, Any] = {"model_max_length": max_input_length}
    nanobeir_cfg: DictConfig | None = getattr(cfg, "nanobeir", None)
    cache_dir: str | None = getattr(nanobeir_cfg, "cache_dir", None)

    # Initialize the SentenceTransformers MLMTransformer and overwrite weights.
    mlm_transformer: MLMTransformer = MLMTransformer(
        model_name_or_path=str(cfg.model.huggingface_name),
        max_seq_length=max_input_length,
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        cache_dir=cache_dir,
    )

    # Load the Lightning checkpoint on CPU to avoid device mismatches.
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    raw_state_dict: dict[str, Any] = checkpoint.get("state_dict", checkpoint)
    state_dict: dict[str, torch.Tensor] = {}
    raw_key: str
    raw_value: Any
    for raw_key, raw_value in raw_state_dict.items():
        if isinstance(raw_value, torch.Tensor):
            state_dict[raw_key] = raw_value
    mlm_state_dict: dict[str, torch.Tensor] = _extract_mlm_state_dict(state_dict)
    incompatible: Any = mlm_transformer.auto_model.load_state_dict(
        mlm_state_dict, strict=False
    )
    missing_keys: list[str] = list(incompatible.missing_keys)
    unexpected_keys: list[str] = list(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        logger.warning(
            "Loaded MLM weights with missing=%d unexpected=%d",
            len(missing_keys),
            len(unexpected_keys),
        )
    return mlm_transformer


def build_sparse_encoder_from_checkpoint(
    cfg: DictConfig,
    checkpoint_path: str,
    device: torch.device,
) -> SparseEncoder:
    """Build a SentenceTransformers SparseEncoder from a Lightning checkpoint."""
    query_pooling: str = str(cfg.model.query_pooling)
    doc_pooling: str = str(cfg.model.doc_pooling)
    if query_pooling != doc_pooling:
        raise ValueError("NanoBEIR evaluation requires query_pooling == doc_pooling.")

    sparse_activation: str = str(cfg.model.sparse_activation)
    activation_function: str
    if sparse_activation == "log1p_relu":
        # SentenceTransformers SpladePooling applies log1p after ReLU.
        activation_function = "relu"
    elif sparse_activation == "relu":
        raise ValueError(
            "Sparse activation 'relu' is not supported because "
            "SentenceTransformers SpladePooling always applies log1p."
        )
    else:
        raise ValueError(f"Unsupported sparse_activation: {sparse_activation}")

    # Build the MLM + pooling stack and optional normalization.
    mlm_transformer: MLMTransformer = _load_mlm_transformer(
        cfg=cfg, checkpoint_path=checkpoint_path
    )
    splade_pooling: SpladePooling = SpladePooling(
        pooling_strategy=doc_pooling,
        activation_function=activation_function,
        word_embedding_dimension=mlm_transformer.get_sentence_embedding_dimension(),
    )

    modules: list[Any] = [mlm_transformer, splade_pooling]
    if bool(getattr(cfg.model, "normalize", False)):
        # SentenceTransformers Normalize module mirrors L2 normalization.
        modules.append(Normalize())

    sparse_encoder: SparseEncoder = SparseEncoder(
        modules=modules, similarity_fn_name="dot"
    )
    sparse_encoder.to(device)
    sparse_encoder.eval()
    return sparse_encoder
