from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from src.utils.file_io import read_jsonl, read_tsv


@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class Query:
    query_id: str
    text: str


class RetrievalDataset:
    def __init__(self, corpus_path: str, queries_path: str, qrels_path: str) -> None:
        if not Path(corpus_path).exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        if not Path(queries_path).exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")
        if not Path(qrels_path).exists():
            raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

        self.corpus = self._load_corpus(corpus_path)
        self.queries = self._load_queries(queries_path)
        self.qrels = self._load_qrels(qrels_path)

    @classmethod
    def from_data(
        cls, corpus: list[Document], queries: list[Query], qrels: dict[str, dict[str, int]]
    ) -> "RetrievalDataset":
        obj = cls.__new__(cls)
        obj.corpus = corpus
        obj.queries = queries
        obj.qrels = qrels
        return obj

    @classmethod
    def from_hf(
        cls, hf_name: str, split: str = "test", cache_dir: str | None = None
    ) -> "RetrievalDataset":
        corpus_ds = cls._load_hf_split(hf_name, "corpus", split, cache_dir)
        queries_ds = cls._load_hf_split(hf_name, "queries", split, cache_dir)
        qrels_ds = cls._load_hf_split(hf_name, "qrels", split, cache_dir)

        corpus: list[Document] = []
        for row in corpus_ds:
            doc_id = str(row.get("_id") or row.get("doc_id") or row.get("id"))
            title = row.get("title") or ""
            text = row.get("text") or row.get("contents") or row.get("body") or ""
            full_text = (title + "\n" + text).strip()
            corpus.append(Document(doc_id=doc_id, text=full_text))

        queries: list[Query] = []
        for row in queries_ds:
            qid = str(row.get("_id") or row.get("query_id") or row.get("id"))
            text = row.get("text") or row.get("query") or ""
            queries.append(Query(query_id=qid, text=text))

        qrels: dict[str, dict[str, int]] = {}
        for row in qrels_ds:
            qid = str(row.get("query-id") or row.get("query_id") or row.get("qid"))
            doc_id = str(
                row.get("corpus-id")
                or row.get("doc_id")
                or row.get("pid")
                or row.get("docid")
            )
            score = int(row.get("score") or row.get("relevance") or row.get("rel") or 0)
            qrels.setdefault(qid, {})[doc_id] = score

        return cls.from_data(corpus=corpus, queries=queries, qrels=qrels)

    @staticmethod
    def _load_corpus(path: str) -> list[Document]:
        docs: list[Document] = []
        for row in read_jsonl(path):
            doc_id = str(row.get("_id") or row.get("doc_id") or row.get("id"))
            title = row.get("title") or ""
            text = row.get("text") or row.get("contents") or row.get("body") or ""
            full_text = (title + "\n" + text).strip()
            docs.append(Document(doc_id=doc_id, text=full_text))
        return docs

    @staticmethod
    def _load_queries(path: str) -> list[Query]:
        queries: list[Query] = []
        for row in read_jsonl(path):
            qid = str(row.get("_id") or row.get("query_id") or row.get("id"))
            text = row.get("text") or row.get("query") or ""
            queries.append(Query(query_id=qid, text=text))
        return queries

    @staticmethod
    def _load_qrels(path: str) -> dict[str, dict[str, int]]:
        qrels: dict[str, dict[str, int]] = {}
        rows = read_tsv(path, has_header=False)
        for row in rows:
            if len(row) < 3:
                continue
            qid = row[0]
            doc_id = row[1]
            rel = int(row[2])
            qrels.setdefault(qid, {})[doc_id] = rel
        return qrels

    @staticmethod
    def _load_hf_split(
        hf_name: str, config: str, split: str, cache_dir: str | None
    ):
        try:
            return load_dataset(hf_name, config, split=split, cache_dir=cache_dir)
        except Exception:
            return load_dataset(hf_name, config, split="train", cache_dir=cache_dir)


class CorpusDataset:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int) -> Document:
        return self.docs[idx]


class QueryDataset:
    def __init__(self, queries: list[Query]) -> None:
        self.queries = queries

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Query:
        return self.queries[idx]
