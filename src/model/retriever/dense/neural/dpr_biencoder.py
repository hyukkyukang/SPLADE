import logging
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import DictConfig
from transformers import BertConfig, BertModel

from src.model.retriever.dense.neural.hf_dense import (
    DenseEncoder,
    DenseRetrievalModel,
    _DenseEncoderWrapper,
)

_DPR_QUERY_PREFIX: str = "question_model."
_DPR_CTX_PREFIX: str = "ctx_model."
_ALLOWED_BERT_MISSING_KEYS: frozenset[str] = frozenset(
    {"embeddings.position_ids", "embeddings.token_type_ids"}
)


def _get_model_cfg_value(model_cfg: DictConfig, key: str, default: Any = None) -> Any:
    if model_cfg is None:
        return default
    bert_cfg: Any = model_cfg.get("bert_config")
    if bert_cfg is not None and hasattr(bert_cfg, "get"):
        nested_value: Any = bert_cfg.get(key)
        if nested_value is not None:
            return nested_value
    return model_cfg.get(key, default)


def _extract_checkpoint_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    for key in ("model_dict", "state_dict", "model_state_dict"):
        state_dict: Any = checkpoint.get(key)
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Checkpoint does not expose a valid model state_dict.")


def _extract_prefixed_state_dict(
    state_dict: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    extracted: dict[str, torch.Tensor] = {}
    key: str
    value: Any
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        extracted[key[len(prefix) :]] = value
    if not extracted:
        raise ValueError(f"Checkpoint is missing required DPR tower prefix: {prefix}")
    return extracted


def _infer_num_attention_heads(hidden_size: int) -> int:
    preferred_head_dims: tuple[int, ...] = (
        64,
        128,
        80,
        96,
        48,
        32,
        24,
        16,
        4,
        8,
        2,
        1,
    )
    head_dim: int
    for head_dim in preferred_head_dims:
        if hidden_size % head_dim != 0:
            continue
        num_heads: int = hidden_size // head_dim
        if num_heads > 0:
            return num_heads
    raise ValueError(
        "Unable to infer num_attention_heads from hidden_size="
        f"{hidden_size}. Set model.bert_config.num_attention_heads explicitly."
    )


def infer_dpr_bert_config_from_state_dict(
    state_dict: dict[str, Any],
    *,
    model_cfg: DictConfig | None = None,
    prefix: str = _DPR_QUERY_PREFIX,
) -> BertConfig:
    word_embeddings: torch.Tensor = state_dict[f"{prefix}embeddings.word_embeddings.weight"]
    position_embeddings: torch.Tensor = state_dict[
        f"{prefix}embeddings.position_embeddings.weight"
    ]
    token_type_embeddings: torch.Tensor = state_dict[
        f"{prefix}embeddings.token_type_embeddings.weight"
    ]
    hidden_size: int = int(word_embeddings.shape[1])
    intermediate_weight: torch.Tensor = state_dict[
        f"{prefix}encoder.layer.0.intermediate.dense.weight"
    ]
    intermediate_size: int = int(intermediate_weight.shape[0])
    layer_prefix: str = f"{prefix}encoder.layer."
    layer_ids: set[int] = set()
    key: str
    for key in state_dict:
        if not key.startswith(layer_prefix):
            continue
        suffix: str = key[len(layer_prefix) :]
        parts: list[str] = suffix.split(".", 1)
        if not parts or not parts[0].isdigit():
            continue
        layer_ids.add(int(parts[0]))
    num_hidden_layers: int = max(layer_ids) + 1 if layer_ids else 0
    if num_hidden_layers <= 0:
        raise ValueError("Unable to infer DPR num_hidden_layers from checkpoint.")

    explicit_heads: Any = None
    if model_cfg is not None:
        explicit_heads = _get_model_cfg_value(model_cfg, "num_attention_heads")
    num_attention_heads: int = (
        int(explicit_heads)
        if explicit_heads is not None
        else _infer_num_attention_heads(hidden_size)
    )
    return BertConfig(
        vocab_size=int(_get_model_cfg_value(model_cfg, "vocab_size", word_embeddings.shape[0])),
        hidden_size=int(_get_model_cfg_value(model_cfg, "hidden_size", hidden_size)),
        num_hidden_layers=int(
            _get_model_cfg_value(model_cfg, "num_hidden_layers", num_hidden_layers)
        ),
        num_attention_heads=num_attention_heads,
        intermediate_size=int(
            _get_model_cfg_value(model_cfg, "intermediate_size", intermediate_size)
        ),
        max_position_embeddings=int(
            _get_model_cfg_value(
                model_cfg,
                "max_position_embeddings",
                position_embeddings.shape[0],
            )
        ),
        type_vocab_size=int(
            _get_model_cfg_value(model_cfg, "type_vocab_size", token_type_embeddings.shape[0])
        ),
        hidden_dropout_prob=float(
            _get_model_cfg_value(model_cfg, "hidden_dropout_prob", 0.1)
        ),
        attention_probs_dropout_prob=float(
            _get_model_cfg_value(model_cfg, "attention_probs_dropout_prob", 0.1)
        ),
        layer_norm_eps=float(_get_model_cfg_value(model_cfg, "layer_norm_eps", 1e-12)),
        pad_token_id=int(_get_model_cfg_value(model_cfg, "pad_token_id", 0)),
    )


def infer_dpr_bert_config_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_cfg: DictConfig | None = None,
) -> BertConfig:
    checkpoint: dict[str, Any] = torch.load(str(checkpoint_path), map_location="meta")
    state_dict: dict[str, Any] = _extract_checkpoint_state_dict(checkpoint)
    return infer_dpr_bert_config_from_state_dict(state_dict, model_cfg=model_cfg)


def _resolve_bert_config(
    *,
    model_cfg: DictConfig,
    checkpoint_path: str | None,
) -> BertConfig:
    required_keys: tuple[str, ...] = (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
    )
    if all(_get_model_cfg_value(model_cfg, key) is not None for key in required_keys):
        return BertConfig(
            vocab_size=int(_get_model_cfg_value(model_cfg, "vocab_size")),
            hidden_size=int(_get_model_cfg_value(model_cfg, "hidden_size")),
            num_hidden_layers=int(_get_model_cfg_value(model_cfg, "num_hidden_layers")),
            num_attention_heads=int(_get_model_cfg_value(model_cfg, "num_attention_heads")),
            intermediate_size=int(_get_model_cfg_value(model_cfg, "intermediate_size")),
            max_position_embeddings=int(
                _get_model_cfg_value(model_cfg, "max_position_embeddings", 512)
            ),
            type_vocab_size=int(_get_model_cfg_value(model_cfg, "type_vocab_size", 2)),
            hidden_dropout_prob=float(
                _get_model_cfg_value(model_cfg, "hidden_dropout_prob", 0.1)
            ),
            attention_probs_dropout_prob=float(
                _get_model_cfg_value(model_cfg, "attention_probs_dropout_prob", 0.1)
            ),
            layer_norm_eps=float(_get_model_cfg_value(model_cfg, "layer_norm_eps", 1e-12)),
            pad_token_id=int(_get_model_cfg_value(model_cfg, "pad_token_id", 0)),
        )
    if checkpoint_path is None:
        raise ValueError(
            "DPR bi-encoder requires either model.bert_config fields or a checkpoint "
            "path from which the BERT config can be inferred."
        )
    return infer_dpr_bert_config_from_checkpoint(checkpoint_path, model_cfg=model_cfg)


class DPRBiEncoderModel(DenseRetrievalModel):
    """Dense bi-encoder with separate question and context backbones."""

    def __init__(
        self,
        *,
        family: str,
        query_encoder: DenseEncoder,
        ctx_encoder: DenseEncoder,
        query_pooling: str,
        doc_pooling: str,
        query_window_pooling: str,
        doc_window_pooling: str,
        similarity: str,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        if int(query_encoder.embedding_dim) != int(ctx_encoder.embedding_dim):
            raise ValueError(
                "DPR bi-encoder requires query/doc embedding dimensions to match."
            )
        self.family: str = str(family).lower()
        self.query_encoder: DenseEncoder = query_encoder
        self.ctx_encoder: DenseEncoder = ctx_encoder
        self.encoder: DenseEncoder = self.ctx_encoder
        self.query_pooling: str = str(query_pooling)
        self.doc_pooling: str = str(doc_pooling)
        self.query_window_pooling: str = str(query_window_pooling)
        self.doc_window_pooling: str = str(doc_window_pooling)
        self.similarity: str = str(similarity).strip().lower()
        self.normalize: bool = bool(normalize or self.similarity == "cosine")
        self.embedding_dim: int = int(self.query_encoder.embedding_dim)
        self._query_encoder_wrapper: _DenseEncoderWrapper = _DenseEncoderWrapper(
            self.query_encoder,
            self.query_pooling,
        )
        self._doc_encoder_wrapper: _DenseEncoderWrapper = _DenseEncoderWrapper(
            self.ctx_encoder,
            self.doc_pooling,
        )
        self._query_encoder_fn = self._query_encoder_wrapper
        self._doc_encoder_fn = self._doc_encoder_wrapper


def build_dpr_biencoder_model(
    *,
    model_cfg: DictConfig,
    dtype: torch.dtype | None,
    checkpoint_path: str | None,
) -> DPRBiEncoderModel:
    bert_config: BertConfig = _resolve_bert_config(
        model_cfg=model_cfg,
        checkpoint_path=checkpoint_path,
    )
    question_backbone: BertModel = BertModel(bert_config)
    ctx_backbone: BertModel = BertModel(bert_config)
    if dtype is not None:
        question_backbone.to(dtype=dtype)
        ctx_backbone.to(dtype=dtype)
    query_encoder = DenseEncoder(
        model_name=str(model_cfg.get("huggingface_name") or "dpr_question_encoder"),
        backbone=question_backbone,
    )
    ctx_encoder = DenseEncoder(
        model_name=str(model_cfg.get("huggingface_name") or "dpr_ctx_encoder"),
        backbone=ctx_backbone,
    )
    return DPRBiEncoderModel(
        family=str(model_cfg.get("family", "dense")),
        query_encoder=query_encoder,
        ctx_encoder=ctx_encoder,
        query_pooling=str(model_cfg.get("query_pooling", "cls")),
        doc_pooling=str(model_cfg.get("doc_pooling", "cls")),
        query_window_pooling=str(
            model_cfg.get("query_window_pooling", model_cfg.get("query_pooling", "cls"))
        ),
        doc_window_pooling=str(
            model_cfg.get("doc_window_pooling", model_cfg.get("doc_pooling", "cls"))
        ),
        similarity=str(model_cfg.get("similarity", "dot")),
        normalize=bool(model_cfg.get("normalize", False)),
    )


def _filter_allowed_missing_keys(keys: Iterable[str]) -> list[str]:
    filtered: list[str] = []
    key: str
    for key in keys:
        if key in _ALLOWED_BERT_MISSING_KEYS:
            continue
        filtered.append(key)
    return filtered


def _validate_backbone_incompatible_keys(
    *,
    incompatible: Any,
    tower_name: str,
    logger: logging.Logger | None,
) -> None:
    missing_keys: list[str] = _filter_allowed_missing_keys(
        list(getattr(incompatible, "missing_keys", []))
    )
    unexpected_keys: list[str] = list(getattr(incompatible, "unexpected_keys", []))
    if not missing_keys and not unexpected_keys:
        return
    error_logger = logger or logging.getLogger(__name__)
    error_logger.error(
        "%s checkpoint parameter mismatch: missing=%d, unexpected=%d.",
        tower_name,
        len(missing_keys),
        len(unexpected_keys),
    )
    if missing_keys:
        error_logger.error("Missing %s keys (sample): %s", tower_name, ", ".join(missing_keys[:10]))
    if unexpected_keys:
        error_logger.error(
            "Unexpected %s keys (sample): %s",
            tower_name,
            ", ".join(unexpected_keys[:10]),
        )
    raise RuntimeError(
        f"{tower_name} checkpoint parameters do not match the current model definition."
    )


def load_dpr_biencoder_checkpoint(
    model: DPRBiEncoderModel,
    checkpoint_path: str,
    logger: logging.Logger | None = None,
) -> tuple[list[str], list[str]]:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    state_dict: dict[str, Any] = _extract_checkpoint_state_dict(checkpoint)
    question_state_dict: dict[str, torch.Tensor] = _extract_prefixed_state_dict(
        state_dict,
        prefix=_DPR_QUERY_PREFIX,
    )
    ctx_state_dict: dict[str, torch.Tensor] = _extract_prefixed_state_dict(
        state_dict,
        prefix=_DPR_CTX_PREFIX,
    )
    question_incompatible: Any = model.query_encoder.backbone.load_state_dict(
        question_state_dict,
        strict=False,
    )
    ctx_incompatible: Any = model.ctx_encoder.backbone.load_state_dict(
        ctx_state_dict,
        strict=False,
    )
    _validate_backbone_incompatible_keys(
        incompatible=question_incompatible,
        tower_name="question_model",
        logger=logger,
    )
    _validate_backbone_incompatible_keys(
        incompatible=ctx_incompatible,
        tower_name="ctx_model",
        logger=logger,
    )
    return [], []


__all__ = [
    "DPRBiEncoderModel",
    "build_dpr_biencoder_model",
    "infer_dpr_bert_config_from_checkpoint",
    "infer_dpr_bert_config_from_state_dict",
    "load_dpr_biencoder_checkpoint",
]
