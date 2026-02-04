from dataclasses import dataclass
from typing import Dict, Sequence

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


class QueryDataset(Dataset[Query]):
    """Torch dataset wrapper for query objects."""

    # --- Special methods ---
    def __init__(self, queries: Sequence[Query]) -> None:
        # Store a local copy to prevent unexpected external mutation.
        self._queries: list[Query] = list(queries)

    def __len__(self) -> int:
        # Return the number of available query items.
        return len(self._queries)

    def __getitem__(self, idx: int) -> Query:
        # Provide index-based access for DataLoader compatibility.
        return self._queries[idx]


class CorpusDataset(Dataset[Document]):
    """Torch dataset wrapper for document objects."""

    # --- Special methods ---
    def __init__(self, docs: Sequence[Document]) -> None:
        # Store a local copy to prevent unexpected external mutation.
        self._docs: list[Document] = list(docs)

    def __len__(self) -> int:
        # Return the number of available document items.
        return len(self._docs)

    def __getitem__(self, idx: int) -> Document:
        # Provide index-based access for DataLoader compatibility.
        return self._docs[idx]


@dataclass(frozen=True)
class MetaItem:
    """Training metadata for positive/negative pools."""

    qid: str
    pos_ids: list[str]
    neg_ids: list[str]
    pos_scores: list[float] | None
    neg_scores: list[float] | None
    # Optional inline texts for datasets without corpus/query ids.
    query_text: str | None = None
    pos_texts: list[str] | None = None
    neg_texts: list[str] | None = None


@dataclass(frozen=True)
class RerankingDataItem:
    """Query/document tensors with relevance signals."""

    data_idx: int
    qid: str
    pos_ids: list[str]
    neg_ids: list[str]
    query_text: str
    doc_texts: list[str]
    # Shape: (seq_len,)
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    # Shape: (num_docs, seq_len)
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    # Shape: (num_docs,)
    doc_mask: torch.Tensor
    pos_mask: torch.Tensor
    teacher_scores: torch.Tensor


@dataclass(frozen=True)
class RetrievalDataItem:
    """Evaluation item with query tensors and qrels."""

    data_idx: int
    qid: str
    relevance_judgments: Dict[str, float]
    query_text: str
    # Shape: (seq_len,)
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor


@dataclass(frozen=True)
class EncodingDataItem:
    """Encoding item with document metadata."""

    data_idx: int
    doc_id: str
    doc_text: str


@dataclass(frozen=True)
class TrainingDataItem(RerankingDataItem):
    """Training item with optional label metadata."""

    # Shape: (num_docs,)
    labels: torch.Tensor
    pos_scores: torch.Tensor | None
    neg_scores: torch.Tensor | None
