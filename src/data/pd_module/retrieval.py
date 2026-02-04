from typing import Any, Dict

import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.collator import UniversalCollator
from src.data.dataclass import Document, Query, RetrievalDataItem
from src.data.pd_module import PDModule


class RetrievalPDModule(PDModule):
    """Retrieval PyTorch datasets module for evaluation/inference."""

    # --- Special methods ---
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        *,
        seed: int,
        load_teacher_scores: bool | None = None,
        require_teacher_scores: bool | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            tokenizer=tokenizer,
            seed=seed,
            load_teacher_scores=load_teacher_scores,
            require_teacher_scores=require_teacher_scores,
        )
        self.hf_name: str = (
            str(self.cfg.query_corpus_hf_name)
            if self.cfg.query_corpus_hf_name is not None
            else str(self.cfg.hf_name)
        )
        self.hf_split: str = str(self.cfg.split)
        text_cache_dir: str | None = self.cfg.query_corpus_hf_cache_dir
        self.hf_cache_dir: str | None = (
            None
            if text_cache_dir is None and self.cfg.hf_cache_dir is None
            else str(
                text_cache_dir if text_cache_dir is not None else self.cfg.hf_cache_dir
            )
        )
        self._queries: list[Query] = []
        self._corpus: list[Document] = []
        self._qrels: Dict[str, Dict[str, float]] = {}
        self._query_dataset: Dataset | None = None
        self._corpus_dataset: Dataset | None = None
        self._qrels_dataset: Dataset | None = None

    def __len__(self) -> int:
        return len(self._queries)

    def __getitem__(self, idx: int) -> RetrievalDataItem:
        query: Query = self._queries[int(idx)]
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
            data_idx=int(idx),
            qid=query.query_id,
            relevance_judgments=self.get_relevance_judgments(query.query_id),
            query_text=query.text,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
        )

    # --- Property methods ---
    @property
    def collator(self) -> UniversalCollator:
        return UniversalCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_padding=self.max_padding,
            max_query_length=self.max_query_length,
        )

    @property
    def queries(self) -> list[Query]:
        return self._queries

    @property
    def corpus(self) -> list[Document]:
        return self._corpus

    @property
    def qrels_dict(self) -> Dict[str, Dict[str, float]]:
        return self._qrels

    # --- Protected methods ---
    def _load_hf_split(
        self, hf_name: str, config: str, split: str, cache_dir: str | None
    ) -> Dataset:
        return load_dataset(hf_name, config, split=split, cache_dir=cache_dir)

    # --- Public methods ---
    def prepare_data(self) -> None:
        _ = self._load_hf_split(
            self.hf_name, "corpus", self.hf_split, self.hf_cache_dir
        )
        _ = self._load_hf_split(
            self.hf_name, "queries", self.hf_split, self.hf_cache_dir
        )
        _ = self._load_hf_split(self.hf_name, "qrels", self.hf_split, self.hf_cache_dir)

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

        max_samples: int | None = (
            None if self.cfg.hf_max_samples is None else int(self.cfg.hf_max_samples)
        )

        self._corpus = []
        if self._corpus_dataset is not None:
            for raw_row in self._corpus_dataset:
                row: dict[str, Any] = raw_row
                doc_id: str = str(row.get("_id") or row.get("doc_id") or row.get("id"))
                title: str = str(row.get("title") or "")
                text: str = str(
                    row.get("text") or row.get("contents") or row.get("body") or ""
                )
                full_text: str = (title + " " + text).strip()
                self._corpus.append(Document(doc_id=doc_id, text=full_text))

        self._queries = []
        if self._query_dataset is not None:
            for idx, raw_row in enumerate(self._query_dataset):
                if max_samples is not None and idx >= max_samples:
                    break
                row: dict[str, Any] = raw_row
                qid: str = str(row.get("_id") or row.get("query_id") or row.get("id"))
                text: str = str(row.get("text") or row.get("query") or "")
                self._queries.append(Query(query_id=qid, text=text))

        self._qrels = {}
        if self._qrels_dataset is not None:
            allowed_queries: set[str] = {query.query_id for query in self._queries}
            for raw_row in self._qrels_dataset:
                row: dict[str, Any] = raw_row
                qid: str = str(
                    row.get("query-id")
                    or row.get("query_id")
                    or row.get("qid")
                    or row.get("_id")
                )
                if qid not in allowed_queries:
                    continue
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
