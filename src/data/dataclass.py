from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Query:
    """Lightweight query representation."""

    query_id: str
    text: str


@dataclass(frozen=True)
class Document:
    """Lightweight document representation."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class DataTuple:
    """Dataset item metadata for positive/negative pools."""

    qid: str
    pos_ids: List[str]
    pos_scores: List[float]
    neg_ids: List[str]


@dataclass(frozen=True)
class RerankingDataItem:
    """Training item with query/document tensors and labels."""

    data_idx: int
    qid: str
    pos_ids: List[str]
    neg_ids: List[str]
    query_text: str
    doc_texts: List[str]
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    doc_mask: torch.Tensor
    pos_mask: torch.Tensor
    teacher_scores: torch.Tensor


@dataclass(frozen=True)
class RetrievalDataItem:
    """Evaluation item with relevance judgments and query tensors."""

    data_idx: int
    qid: str
    relevance_judgments: Dict[str, float]
    query_text: str
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor


class QueryDataset(Dataset[Query]):
    """Dataset wrapper for query objects."""

    def __init__(self, queries: List[Query]) -> None:
        self._queries: List[Query] = queries

    def __len__(self) -> int:
        return len(self._queries)

    def __getitem__(self, idx: int) -> Query:
        return self._queries[idx]


class CorpusDataset(Dataset[Document]):
    """Dataset wrapper for document objects."""

    def __init__(self, documents: List[Document]) -> None:
        self._documents: List[Document] = documents

    def __len__(self) -> int:
        return len(self._documents)

    def __getitem__(self, idx: int) -> Document:
        return self._documents[idx]
