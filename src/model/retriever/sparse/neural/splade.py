from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM

from src.model.retriever.base import BaseRetriever
from src.model.retriever.registry import RETRIEVER_REGISTRY
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
        token_scores: torch.Tensor = self.activation(outputs.logits)
        embeddings: torch.Tensor = self._pool_sparse(
            token_scores, attention_mask, pooling_mode
        )
        return embeddings


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
        self.normalize: bool = normalize

    # --- Protected methods ---
    @staticmethod
    def _resolve_pooling_mode(pooling: str) -> float:
        if pooling == "sum":
            return 0.0
        if pooling == "max":
            return 1.0
        raise ValueError(f"Unsupported pooling: {pooling}")

    # --- Public methods ---
    def encode_queries(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mode=self._query_pooling_mode,
        )
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_docs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mode=self._doc_pooling_mode,
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


# Register SPLADE under the canonical config name.
@RETRIEVER_REGISTRY.register("splade")
class SPLADE(BaseRetriever):
    """SPLADE retriever registered for config usage."""
