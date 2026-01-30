from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM

from src.model.retriever.base import BaseRetriever
from src.model.retriever.registry import RETRIEVER_REGISTRY
from src.utils.logging import suppress_output_if_not_rank_zero

def _apply_activation(logits: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "log1p_relu":
        activated: torch.Tensor = torch.log1p(torch.relu(logits))
        return activated
    if activation == "log1p_softplus":
        activated: torch.Tensor = torch.log1p(F.softplus(logits))
        return activated
    if activation == "softplus":
        activated: torch.Tensor = F.softplus(logits)
        return activated
    if activation == "relu":
        activated: torch.Tensor = torch.relu(logits)
        return activated
    raise ValueError(f"Unsupported sparse activation: {activation}")


def _pool_sparse(
    token_scores: torch.Tensor, attention_mask: torch.Tensor, pooling: str
) -> torch.Tensor:
    # Expand mask for token-wise pooling.
    mask: torch.Tensor = attention_mask.unsqueeze(-1).to(token_scores.dtype)
    if pooling == "sum":
        pooled_sum: torch.Tensor = (token_scores * mask).sum(dim=1)
        return pooled_sum
    if pooling == "max":
        neg_inf: float = torch.finfo(token_scores.dtype).min
        masked: torch.Tensor = token_scores.masked_fill(mask == 0, neg_inf)
        pooled: torch.Tensor = masked.max(dim=1).values
        clipped: torch.Tensor = torch.clamp(pooled, min=0.0)
        return clipped
    raise ValueError(f"Unsupported pooling: {pooling}")


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

    # --- Public methods ---
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str,
        normalize: bool,
    ) -> torch.Tensor:
        outputs: Any = self.mlm(input_ids=input_ids, attention_mask=attention_mask)
        token_scores: torch.Tensor = _apply_activation(
            outputs.logits, self.sparse_activation
        )
        embeddings: torch.Tensor = _pool_sparse(token_scores, attention_mask, pooling)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
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
        self.normalize: bool = normalize

    # --- Public methods ---
    def encode_queries(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling=self.query_pooling,
            normalize=self.normalize,
        )

    def encode_docs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling=self.doc_pooling,
            normalize=self.normalize,
        )

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
