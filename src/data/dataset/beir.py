from __future__ import annotations

import logging
from typing import Any, Callable

from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.dataclass import Document
from src.data.dataset.base import BaseDataset

logger: logging.Logger = logging.getLogger("BEIRDataset")


class BEIRDataset(BaseDataset):
    """Minimal BEIR dataset for evaluation-only retrieval tasks."""

    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        resolved_global_cfg: DictConfig = global_cfg or cfg
        BaseDataset.__init__(
            self, cfg=cfg, global_cfg=resolved_global_cfg, tokenizer=tokenizer
        )
        self._hf_name: str = self._resolve_hf_name()
        self._query_ids: list[str] = []
        self._query_texts: dict[str, str] = {}
        self._qrels_dict: dict[str, dict[str, int]] = {}
        self._corpus_docs: list[Document] | None = None
        self._is_loaded: bool = False

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._query_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._ensure_loaded()
        qid: str = self._query_ids[idx]
        return {
            "qid": qid,
            "query_text": self._query_texts[qid],
            "relevance_judgments": self._qrels_dict.get(qid, {}),
        }

    @property
    def corpus(self) -> list[Document]:
        """Return the full corpus as a list of Document objects."""
        if self._corpus_docs is None:
            self._corpus_docs = self._load_corpus()
        return self._corpus_docs

    @property
    def qrels_dict(self) -> dict[str, dict[str, int]]:
        """Return qrels as a dict of query_id -> {doc_id: score}."""
        self._ensure_loaded()
        return self._qrels_dict

    @property
    def collator(self) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Return a collator that preserves per-query structures."""
        return self._collate

    def _ensure_loaded(self) -> None:
        """Load queries/qrels lazily to avoid repeated work in DDP."""
        if self._is_loaded:
            return
        self._load_queries()
        self._load_qrels()
        self._is_loaded = True

    def _resolve_hf_name(self) -> str:
        hf_name: str | None = getattr(self.cfg, "hf_name", None)
        if hf_name:
            return hf_name
        beir_dataset: str | None = getattr(self.cfg, "beir_dataset", None)
        if beir_dataset:
            return f"BeIR/{beir_dataset}"
        raise ValueError("dataset.hf_name or dataset.beir_dataset must be set.")

    def _load_queries(self) -> None:
        cache_dir: str | None = getattr(self.cfg, "hf_cache_dir", None)
        query_dataset: Dataset = load_dataset(
            self._hf_name, "queries", split="train", cache_dir=cache_dir
        )
        max_samples: int | None = getattr(self.cfg, "hf_max_samples", None)

        # Build query IDs and text map.
        self._query_ids = []
        self._query_texts = {}
        for idx, raw_record in enumerate(query_dataset):
            if max_samples is not None and idx >= int(max_samples):
                break
            record: dict[str, Any] = raw_record
            qid: str = str(record.get("_id"))
            text: str = str(record.get("text", ""))
            self._query_ids.append(qid)
            self._query_texts[qid] = text

        logger.info("Loaded %d queries from %s", len(self._query_ids), self._hf_name)

    def _load_qrels(self) -> None:
        cache_dir: str | None = getattr(self.cfg, "hf_cache_dir", None)
        qrels_split: str = str(getattr(self.cfg, "hf_split", "test"))
        qrels_dataset: Dataset = load_dataset(
            self._hf_name, "qrels", split=qrels_split, cache_dir=cache_dir
        )
        allowed_queries: set[str] = set(self._query_ids)

        qrels_dict: dict[str, dict[str, int]] = {}
        for raw_record in qrels_dataset:
            record: dict[str, Any] = raw_record
            qid: str = str(record.get("query-id"))
            if qid not in allowed_queries:
                continue
            doc_id: str = str(record.get("corpus-id"))
            score: int = int(record.get("score", 0))
            qrels_dict.setdefault(qid, {})[doc_id] = score

        self._qrels_dict = qrels_dict
        logger.info("Loaded qrels for %d queries", len(self._qrels_dict))

    def _load_corpus(self) -> list[Document]:
        cache_dir: str | None = getattr(self.cfg, "hf_cache_dir", None)
        corpus_dataset: Dataset = load_dataset(
            self._hf_name, "corpus", split="train", cache_dir=cache_dir
        )
        corpus_docs: list[Document] = []
        for raw_record in corpus_dataset:
            record: dict[str, Any] = raw_record
            doc_id: str = str(record.get("_id"))
            text: str = self._build_doc_text(record)
            corpus_docs.append(Document(doc_id=doc_id, text=text))

        logger.info("Loaded %d documents from %s", len(corpus_docs), self._hf_name)
        return corpus_docs

    def _build_doc_text(self, record: dict[str, Any]) -> str:
        title: str = str(record.get("title", "") or "").strip()
        text: str = str(record.get("text", "") or "").strip()
        if title and text:
            return f"{title} {text}"
        return title or text

    def _collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        qids: list[str] = [item["qid"] for item in batch]
        query_texts: list[str] = [item["query_text"] for item in batch]
        relevance_judgments: list[dict[str, int]] = [
            item["relevance_judgments"] for item in batch
        ]
        return {
            "qid": qids,
            "query_text": query_texts,
            "relevance_judgments": relevance_judgments,
        }

    def prepare_data(self) -> None:
        self._ensure_loaded()

    def setup(self) -> None:
        self._ensure_loaded()
