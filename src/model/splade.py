from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM


@dataclass
class SpladeOutput:
    embeddings: torch.Tensor
    regularization: torch.Tensor


def _apply_activation(logits: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "log1p_relu":
        return torch.log1p(torch.relu(logits))
    if activation == "relu":
        return torch.relu(logits)
    raise ValueError(f"Unsupported sparse activation: {activation}")


def _pool_sparse(
    token_scores: torch.Tensor, attention_mask: torch.Tensor, pooling: str
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(token_scores.dtype)
    if pooling == "sum":
        return (token_scores * mask).sum(dim=1)
    if pooling == "max":
        neg_inf = torch.finfo(token_scores.dtype).min
        masked = token_scores.masked_fill(mask == 0, neg_inf)
        pooled = masked.max(dim=1).values
        return torch.clamp(pooled, min=0.0)
    raise ValueError(f"Unsupported pooling: {pooling}")


class SpladeEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        sparse_activation: str,
        attn_implementation: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        kwargs = {}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.mlm = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)
        self.sparse_activation = sparse_activation

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str,
        normalize: bool,
    ) -> torch.Tensor:
        outputs = self.mlm(input_ids=input_ids, attention_mask=attention_mask)
        token_scores = _apply_activation(outputs.logits, self.sparse_activation)
        embeddings = _pool_sparse(token_scores, attention_mask, pooling)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


class SpladeModel(nn.Module):
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
        self.encoder = SpladeEncoder(
            model_name=model_name,
            sparse_activation=sparse_activation,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        self.query_pooling = query_pooling
        self.doc_pooling = doc_pooling
        self.normalize = normalize

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
        q = self.encode_queries(query_input_ids, query_attention_mask)
        d = self.encode_docs(doc_input_ids, doc_attention_mask)
        return q, d
