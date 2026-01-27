from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, Iterable

import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from src.data.collators import RetrievalCollator
from src.data.dataclass import Document, Query, RetrievalDataItem
from src.data.dataset.base import BaseDataset


class RetrievalDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig | None,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        resolved_global_cfg: DictConfig = global_cfg or cfg
        BaseDataset.__init__(
            self, cfg=cfg, global_cfg=resolved_global_cfg, tokenizer=tokenizer
        )
        self.hf_name: str = str(cfg.hf_name)
        self.hf_split: str = str(cfg.hf_split)
        self.hf_cache_dir: str | None = getattr(cfg, "hf_cache_dir", None)
        self.max_query_length: int = int(cfg.max_query_length)

        self._corpus: list[Document] = []
        self._queries: list[Query] = []
        self._qrels: Dict[str, Dict[str, float]] = {}
        self._corpus_dataset: Dataset | None = None
        self._query_dataset: Dataset | None = None
        self._qrels_dataset: Dataset | None = None

    def __len__(self) -> int:
        return len(self._queries)

    def __getitem__(self, idx: int) -> RetrievalDataItem:
        query: Query = self._queries[idx]
        tokens: dict[str, torch.Tensor] = self.tokenizer(
            query.text,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        query_input_ids: torch.Tensor = tokens["input_ids"].squeeze(0)
        query_attention_mask: torch.Tensor = tokens["attention_mask"].squeeze(0)
        return RetrievalDataItem(
            data_idx=idx,
            qid=query.query_id,
            relevance_judgments=self.get_relevance_judgments(query.query_id),
            query_text=query.text,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
        )

    @cached_property
    def collator(self) -> RetrievalCollator:
        return RetrievalCollator(pad_token_id=self.tokenizer.pad_token_id)

    @property
    def query_dataset(self) -> Dataset:
        if self._query_dataset is None:
            self._query_dataset = self._load_hf_split(
                self.hf_name, "queries", self.hf_split, self.hf_cache_dir
            )
        return self._query_dataset

    @property
    def corpus_dataset(self) -> Dataset:
        if self._corpus_dataset is None:
            self._corpus_dataset = self._load_hf_split(
                self.hf_name, "corpus", self.hf_split, self.hf_cache_dir
            )
        return self._corpus_dataset

    @property
    def qrels_dict(self) -> Dict[str, Dict[str, float]]:
        return self._qrels

    @property
    def corpus(self) -> list[Document]:
        return self._corpus

    @property
    def queries(self) -> list[Query]:
        return self._queries

    @cached_property
    def corpus_text_column(self) -> str:
        return self._resolve_column(
            self.corpus_dataset.column_names, ("text", "contents", "body")
        )

    @cached_property
    def query_id_column(self) -> str:
        return self._resolve_column(
            self.query_dataset.column_names, ("query_id", "qid", "_id", "id")
        )

    @cached_property
    def corpus_id_column(self) -> str:
        return self._resolve_column(
            self.corpus_dataset.column_names, ("doc_id", "corpus_id", "_id", "id")
        )

    @cached_property
    def qrel_query_column(self) -> str:
        if self._qrels_dataset is None:
            raise RuntimeError("Qrels dataset is not loaded. Call setup() first.")
        return self._resolve_column(
            self._qrels_dataset.column_names, ("query-id", "query_id", "qid")
        )

    @cached_property
    def qrel_doc_column(self) -> str:
        if self._qrels_dataset is None:
            raise RuntimeError("Qrels dataset is not loaded. Call setup() first.")
        return self._resolve_column(
            self._qrels_dataset.column_names,
            ("corpus-id", "doc_id", "pid", "docid"),
        )

    @cached_property
    def qrel_score_column(self) -> str:
        if self._qrels_dataset is None:
            raise RuntimeError("Qrels dataset is not loaded. Call setup() first.")
        return self._resolve_column(
            self._qrels_dataset.column_names, ("score", "relevance", "rel")
        )

    def _resolve_column(
        self, column_names: Iterable[str], candidates: tuple[str, ...]
    ) -> str:
        for name in candidates:
            if name in column_names:
                return name
        raise ValueError(f"Unable to resolve column from {column_names}")

    def _load_hf_split(
        self, hf_name: str, config: str, split: str, cache_dir: str | None
    ) -> Dataset:
        try:
            return load_dataset(hf_name, config, split=split, cache_dir=cache_dir)
        except Exception:  # pylint: disable=broad-except
            return load_dataset(hf_name, config, split="train", cache_dir=cache_dir)

    def prepare_data(self) -> None:
        self._load_hf_split(self.hf_name, "corpus", self.hf_split, self.hf_cache_dir)
        self._load_hf_split(self.hf_name, "queries", self.hf_split, self.hf_cache_dir)
        self._load_hf_split(self.hf_name, "qrels", self.hf_split, self.hf_cache_dir)

    def setup(self) -> None:
        self._corpus_dataset = self._load_hf_split(
            self.hf_name, "corpus", self.hf_split, self.hf_cache_dir
        )
        self._query_dataset = self._load_hf_split(
            self.hf_name, "queries", self.hf_split, self.hf_cache_dir
        )
        self._qrels_dataset = self._load_hf_split(
            self.hf_name, "qrels", self.hf_split, self.hf_cache_dir
        )

        # Build an in-memory corpus for quick document lookups.
        self._corpus = []
        for raw_row in self._corpus_dataset:
            row: dict[str, Any] = raw_row
            doc_id: str = str(row.get("_id") or row.get("doc_id") or row.get("id"))
            title: str = str(row.get("title") or "")
            text: str = str(
                row.get("text") or row.get("contents") or row.get("body") or ""
            )
            full_text: str = (title + "\n" + text).strip()
            self._corpus.append(Document(doc_id=doc_id, text=full_text))

        # Build query objects for evaluation.
        self._queries = []
        for raw_row in self._query_dataset:
            row: dict[str, Any] = raw_row
            qid: str = str(row.get("_id") or row.get("query_id") or row.get("id"))
            text: str = str(row.get("text") or row.get("query") or "")
            self._queries.append(Query(query_id=qid, text=text))

        # Build qrels mapping keyed by query ID.
        self._qrels = {}
        for raw_row in self._qrels_dataset:
            row: dict[str, Any] = raw_row
            qid: str = str(
                row.get("query-id")
                or row.get("query_id")
                or row.get("qid")
                or row.get("_id")
            )
            doc_id: str = str(
                row.get("corpus-id")
                or row.get("doc_id")
                or row.get("pid")
                or row.get("docid")
            )
            score: float = float(
                row.get("score") or row.get("relevance") or row.get("rel") or 0
            )
            self._qrels.setdefault(qid, {})[doc_id] = score

    def get_relevance_judgments(self, qid: str) -> Dict[str, float]:
        return self._qrels.get(qid, {})

    @classmethod
    def from_hf(
        cls,
        hf_name: str,
        split: str,
        cache_dir: str | None,
        tokenizer: PreTrainedTokenizerBase,
        max_query_length: int,
    ) -> "RetrievalDataset":
        cfg: DictConfig = OmegaConf.create(
            {
                "name": "hf_retrieval",
                "hf_name": hf_name,
                "hf_split": split,
                "hf_cache_dir": cache_dir,
                "max_query_length": max_query_length,
            }
        )
        dataset: RetrievalDataset = cls(cfg=cfg, global_cfg=None, tokenizer=tokenizer)
        dataset.prepare_data()
        dataset.setup()
        return dataset
