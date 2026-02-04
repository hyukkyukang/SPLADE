from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM

from src.utils.logging import suppress_output_if_not_rank_zero


class _Log1pRelu(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.relu(logits))


class _Log1pSoftplus(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.log1p(F.softplus(logits))


class _Softplus(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softplus(logits)


class _Relu(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.relu(logits)


def _resolve_activation_module(activation: str) -> nn.Module:
    if activation == "log1p_relu":
        return _Log1pRelu()
    if activation == "log1p_softplus":
        return _Log1pSoftplus()
    if activation == "softplus":
        return _Softplus()
    if activation == "relu":
        return _Relu()
    raise ValueError(f"Unsupported sparse activation: {activation}")


class SpladeEncoder(nn.Module):
    # --- Special methods ---
    def __init__(
        self,
        model_name: str,
        sparse_activation: str,
        attn_implementation: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        kwargs: dict[str, Any] = {}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        if dtype is not None:
            kwargs["dtype"] = dtype
        # Load the masked language model backbone.
        self.mlm: AutoModelForMaskedLM
        # Avoid duplicate load reports on non-zero ranks.
        with suppress_output_if_not_rank_zero():
            self.mlm = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)
        self.sparse_activation: str = sparse_activation
        self.activation: nn.Module = _resolve_activation_module(sparse_activation)
        self._neg_inf: torch.Tensor
        self.register_buffer("_neg_inf", torch.tensor(float("-inf")), persistent=False)

    # --- Protected methods ---
    def _pool_sparse(
        self,
        token_scores: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mode: torch.Tensor,
    ) -> torch.Tensor:
        # Expand mask for token-wise pooling.
        mask: torch.Tensor = attention_mask.unsqueeze(-1).to(token_scores.dtype)
        pooled_sum: torch.Tensor = (token_scores * mask).sum(dim=1)
        neg_inf: torch.Tensor = self._neg_inf.to(
            dtype=token_scores.dtype, device=token_scores.device
        )
        masked: torch.Tensor = token_scores.masked_fill(mask == 0, neg_inf)
        pooled_max: torch.Tensor = torch.clamp(masked.max(dim=1).values, min=0.0)
        pooling_value: torch.Tensor = pooling_mode.to(
            dtype=token_scores.dtype, device=token_scores.device
        )
        pooled: torch.Tensor = pooled_sum + (pooled_max - pooled_sum) * pooling_value
        return pooled

    # --- Public methods ---
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mode: torch.Tensor,
    ) -> torch.Tensor:
        outputs: Any = self.mlm(input_ids=input_ids, attention_mask=attention_mask)
        logits: torch.Tensor = outputs.logits
        token_scores: torch.Tensor = self.activation(logits)
        embeddings: torch.Tensor = self._pool_sparse(
            token_scores, attention_mask, pooling_mode
        )
        return embeddings


class _SpladeEncoderWrapper(nn.Module):
    def __init__(self, encoder: SpladeEncoder, pooling_mode: torch.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("_pooling_mode", pooling_mode, persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mode=self._pooling_mode,
        )


class SpladeModel(nn.Module):
    # --- Special methods ---
    def __init__(
        self,
        model_name: str,
        query_pooling: str,
        doc_pooling: str,
        sparse_activation: str,
        attn_implementation: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        normalize: bool = False,
        doc_only: bool = False,
    ) -> None:
        super().__init__()
        # Build encoder shared by query and document pooling.
        self.encoder: SpladeEncoder = SpladeEncoder(
            model_name=model_name,
            sparse_activation=sparse_activation,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        self.query_pooling: str = query_pooling
        self.doc_pooling: str = doc_pooling
        self._query_pooling_mode: torch.Tensor
        self.register_buffer(
            "_query_pooling_mode",
            torch.tensor(self._resolve_pooling_mode(query_pooling)),
            persistent=False,
        )
        self._doc_pooling_mode: torch.Tensor
        self.register_buffer(
            "_doc_pooling_mode",
            torch.tensor(self._resolve_pooling_mode(doc_pooling)),
            persistent=False,
        )
        self._query_encoder_wrapper: _SpladeEncoderWrapper = _SpladeEncoderWrapper(
            self.encoder, self._query_pooling_mode
        )
        self._doc_encoder_wrapper: _SpladeEncoderWrapper = _SpladeEncoderWrapper(
            self.encoder, self._doc_pooling_mode
        )
        self._query_encoder_fn: Callable[..., torch.Tensor] = self._query_encoder_wrapper
        self._doc_encoder_fn: Callable[..., torch.Tensor] = self._doc_encoder_wrapper
        self.normalize: bool = normalize
        self.doc_only: bool = bool(doc_only)
        exclude_token_ids: torch.Tensor = self._build_query_exclude_token_ids()
        vocab_size: int = int(self.encoder.mlm.config.vocab_size)
        exclude_mask: torch.Tensor = self._build_query_exclude_mask(
            exclude_token_ids, vocab_size
        )
        self.register_buffer(
            "_query_exclude_token_ids", exclude_token_ids, persistent=False
        )
        self.register_buffer("_query_exclude_mask", exclude_mask, persistent=False)
        self._query_encode_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            self._encode_query_terms if self.doc_only else self._encode_query_mlm
        )

    # --- Protected methods ---
    @staticmethod
    def _resolve_pooling_mode(pooling: str) -> float:
        if pooling == "sum":
            return 0.0
        if pooling == "max":
            return 1.0
        raise ValueError(f"Unsupported pooling: {pooling}")

    def _build_query_exclude_token_ids(self) -> torch.Tensor:
        """Collect special token IDs to exclude from query bag-of-words."""
        config: Any = self.encoder.mlm.config
        candidate_ids: list[int] = []
        try:
            value: Any = config.pad_token_id
        except AttributeError:
            value = None
        if value is not None:
            token_id: int = int(value)
            if token_id >= 0:
                candidate_ids.append(token_id)
        try:
            value = config.cls_token_id
        except AttributeError:
            value = None
        if value is not None:
            token_id = int(value)
            if token_id >= 0:
                candidate_ids.append(token_id)
        try:
            value = config.sep_token_id
        except AttributeError:
            value = None
        if value is not None:
            token_id = int(value)
            if token_id >= 0:
                candidate_ids.append(token_id)
        try:
            value = config.bos_token_id
        except AttributeError:
            value = None
        if value is not None:
            token_id = int(value)
            if token_id >= 0:
                candidate_ids.append(token_id)
        try:
            value = config.eos_token_id
        except AttributeError:
            value = None
        if value is not None:
            token_id = int(value)
            if token_id >= 0:
                candidate_ids.append(token_id)
        unique_ids: list[int] = sorted(set(candidate_ids))
        return torch.tensor(unique_ids, dtype=torch.long)

    def _build_query_exclude_mask(
        self, exclude_ids: torch.Tensor, vocab_size: int
    ) -> torch.Tensor:
        """Build a vocab-sized mask for excluded query tokens."""
        if int(exclude_ids.numel()) == 0:
            return torch.empty((0,), dtype=torch.bool)
        mask: torch.Tensor = torch.zeros(int(vocab_size), dtype=torch.bool)
        mask[exclude_ids] = True
        return mask

    def _encode_query_terms(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode queries as a bag-of-words over input tokens."""
        batch_size: int = int(input_ids.shape[0])
        vocab_size: int = int(self.encoder.mlm.config.vocab_size)
        device: torch.device = input_ids.device
        dtype: torch.dtype = self.encoder.mlm.dtype

        token_ids: torch.Tensor = input_ids.to(dtype=torch.long)
        # Mask out padding and special tokens before counting terms.
        token_mask: torch.Tensor = attention_mask.to(dtype=torch.bool)
        exclude_mask: torch.Tensor = self._query_exclude_mask
        if int(exclude_mask.numel()) > 0:
            exclude_mask = exclude_mask.to(device=device)
            token_mask = token_mask & ~exclude_mask[token_ids]
        token_values: torch.Tensor = token_mask.to(dtype=dtype)
        bow: torch.Tensor = torch.zeros(
            (batch_size, vocab_size), dtype=dtype, device=device
        )
        # Accumulate term counts for each query in the batch.
        bow.scatter_add_(1, token_ids, token_values)
        return bow

    def _encode_query_mlm(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode queries using the MLM-based SPLADE encoder."""
        embeddings: torch.Tensor = self._query_encoder_fn(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return embeddings

    # --- Public methods ---
    def encode_queries(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self._query_encode_fn(input_ids, attention_mask)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_docs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self._doc_encoder_fn(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q: torch.Tensor = self.encode_queries(query_input_ids, query_attention_mask)
        d: torch.Tensor = self.encode_docs(doc_input_ids, doc_attention_mask)
        return q, d
