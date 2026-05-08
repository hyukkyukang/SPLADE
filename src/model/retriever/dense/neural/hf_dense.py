from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel

from src.utils.logging import suppress_output_if_not_rank_zero
from src.utils.transformers import build_pretrained_model


def _resolve_module_dtype(
    module: nn.Module,
    *,
    fallback: torch.dtype = torch.float32,
) -> torch.dtype:
    parameter: nn.Parameter
    for parameter in module.parameters():
        return parameter.dtype
    buffer: torch.Tensor
    for buffer in module.buffers():
        return buffer.dtype
    dtype_attr: Any = getattr(module, "dtype", None)
    if isinstance(dtype_attr, torch.dtype):
        return dtype_attr
    return fallback


def _extract_last_hidden_state(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    hidden_state: Any = getattr(model_output, "last_hidden_state", None)
    if isinstance(hidden_state, torch.Tensor):
        return hidden_state
    if (
        isinstance(model_output, tuple)
        and len(model_output) > 0
        and isinstance(model_output[0], torch.Tensor)
    ):
        return model_output[0]
    raise ValueError(
        "Dense encoder forward did not expose last_hidden_state-compatible output."
    )


def _extract_pooler_output(model_output: Any) -> torch.Tensor | None:
    if isinstance(model_output, torch.Tensor):
        return None
    pooler_output: Any = getattr(model_output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    return None


def _masked_mean_pool(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask_float: torch.Tensor = mask.to(dtype=hidden_states.dtype)
    denom: torch.Tensor = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (hidden_states * mask_float.unsqueeze(-1)).sum(dim=1) / denom


def _masked_max_pool(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    masked_hidden: torch.Tensor = hidden_states.masked_fill(
        ~mask.unsqueeze(-1), float("-inf")
    )
    pooled: torch.Tensor = masked_hidden.max(dim=1).values
    empty_rows: torch.Tensor = ~mask.any(dim=1)
    if bool(empty_rows.any()):
        pooled = pooled.clone()
        pooled[empty_rows] = 0
    return pooled


def _resolve_last_token_positions(mask: torch.Tensor) -> torch.Tensor:
    token_counts: torch.Tensor = mask.to(dtype=torch.long).sum(dim=1)
    return token_counts.clamp_min(1) - 1


class DenseEncoder(nn.Module):
    """HF backbone that emits one dense vector per input sequence."""

    def __init__(
        self,
        model_name: str,
        *,
        huggingface_model_class: str = "AutoModel",
        attn_implementation: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        model_revision: str | None = None,
        local_files_only: bool | None = None,
        backbone: PreTrainedModel | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            kwargs: dict[str, Any] = {}
            if attn_implementation is not None:
                kwargs["attn_implementation"] = attn_implementation
            if dtype is not None:
                kwargs["dtype"] = dtype
            with suppress_output_if_not_rank_zero():
                self.backbone = build_pretrained_model(
                    model_name,
                    model_class_name=huggingface_model_class,
                    trust_remote_code=trust_remote_code,
                    revision=model_revision,
                    local_files_only=local_files_only,
                    **kwargs,
                )
        else:
            self.backbone = backbone
        hidden_size: Any = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, "dim", None)
        if hidden_size is None:
            raise ValueError("Dense model config must define hidden_size or dim.")
        self.embedding_dim: int = int(hidden_size)
        self.dtype: torch.dtype = _resolve_module_dtype(self.backbone)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
        *,
        pooling: str,
    ) -> torch.Tensor:
        outputs: Any = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states: torch.Tensor = _extract_last_hidden_state(outputs)
        pooler_output: torch.Tensor | None = _extract_pooler_output(outputs)
        mask: torch.Tensor = (
            attention_mask if pooling_mask is None else pooling_mask
        ).to(dtype=torch.bool)
        normalized_pooling: str = str(pooling).strip().lower()
        if normalized_pooling == "pooler":
            if pooler_output is None:
                return hidden_states[:, 0]
            return pooler_output
        if normalized_pooling == "cls":
            return hidden_states[:, 0]
        if normalized_pooling == "mean":
            return _masked_mean_pool(hidden_states, mask)
        if normalized_pooling == "max":
            return _masked_max_pool(hidden_states, mask)
        if normalized_pooling == "last_token":
            positions: torch.Tensor = _resolve_last_token_positions(mask)
            batch_indices: torch.Tensor = torch.arange(
                hidden_states.shape[0], device=hidden_states.device
            )
            return hidden_states[batch_indices, positions]
        raise ValueError(
            "Unsupported dense token pooling. Expected one of: "
            "cls, mean, max, last_token."
        )


class _DenseEncoderWrapper(nn.Module):
    def __init__(self, encoder: DenseEncoder, pooling: str) -> None:
        super().__init__()
        self.encoder: DenseEncoder = encoder
        self.pooling: str = str(pooling)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mask=pooling_mask,
            pooling=self.pooling,
        )


class DenseRetrievalModel(nn.Module):
    """Base class for dense retrieval models with shared postprocessing."""

    family: str
    query_pooling: str
    doc_pooling: str
    query_window_pooling: str
    doc_window_pooling: str
    similarity: str
    normalize: bool
    embedding_dim: int
    _query_encoder_wrapper: nn.Module
    _doc_encoder_wrapper: nn.Module
    _query_encoder_fn: Callable[..., torch.Tensor]
    _doc_encoder_fn: Callable[..., torch.Tensor]

    def postprocess_query_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def postprocess_doc_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self._query_encoder_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mask=pooling_mask,
        )
        return self.postprocess_query_embeddings(embeddings)

    def encode_docs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self._doc_encoder_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mask=pooling_mask,
        )
        return self.postprocess_doc_embeddings(embeddings)


class DenseModel(DenseRetrievalModel):
    """Dense retrieval model with separate query/doc pooling and normalization."""

    def __init__(
        self,
        *,
        family: str,
        model_name: str,
        huggingface_model_class: str,
        query_pooling: str,
        doc_pooling: str,
        query_window_pooling: str,
        doc_window_pooling: str,
        similarity: str,
        attn_implementation: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        normalize: bool = False,
        trust_remote_code: bool = False,
        model_revision: str | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        super().__init__()
        self.family: str = str(family).lower()
        self.encoder: DenseEncoder = DenseEncoder(
            model_name=model_name,
            huggingface_model_class=huggingface_model_class,
            attn_implementation=attn_implementation,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            model_revision=model_revision,
            local_files_only=local_files_only,
        )
        self.query_pooling: str = str(query_pooling)
        self.doc_pooling: str = str(doc_pooling)
        self.query_window_pooling: str = str(query_window_pooling)
        self.doc_window_pooling: str = str(doc_window_pooling)
        self.similarity: str = str(similarity).strip().lower()
        self.normalize: bool = bool(normalize or self.similarity == "cosine")
        self.embedding_dim: int = int(self.encoder.embedding_dim)
        self._query_encoder_wrapper: _DenseEncoderWrapper = _DenseEncoderWrapper(
            self.encoder,
            self.query_pooling,
        )
        self._doc_encoder_wrapper: _DenseEncoderWrapper = _DenseEncoderWrapper(
            self.encoder,
            self.doc_pooling,
        )
        self._query_encoder_fn: Callable[..., torch.Tensor] = (
            self._query_encoder_wrapper
        )
        self._doc_encoder_fn: Callable[..., torch.Tensor] = self._doc_encoder_wrapper

__all__ = [
    "DenseEncoder",
    "DenseModel",
    "DenseRetrievalModel",
]
