from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class Query:
    query_id: str
    text: str


class RetrievalDataset:
    def __init__(self) -> None:
        raise ValueError(
            "Local file datasets are no longer supported. "
            "Use RetrievalDataset.from_hf() instead."
        )

    @classmethod
    def from_data(
        cls,
        corpus: list[Document],
        queries: list[Query],
        qrels: dict[str, dict[str, int]],
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
    def _load_hf_split(hf_name: str, config: str, split: str, cache_dir: str | None):
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
