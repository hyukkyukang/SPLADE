from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
