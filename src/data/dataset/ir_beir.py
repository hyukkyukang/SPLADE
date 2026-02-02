from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

import ir_datasets
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.dataset.base import BaseDataset
from src.utils.logging import log_if_rank_zero

logger: logging.Logger = logging.getLogger("IRBEIRDataset")


class IRBEIRDataset(BaseDataset):
    """Evaluation dataset using ir_datasets for BEIR queries and qrels."""

    # --- Special methods ---
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
        self._ir_base_name: str = self._resolve_ir_base_name()
        self._ir_split: str = self._resolve_ir_split()
        self._ir_eval_name: str | None = None
        self._eval_dataset: Any | None = None
        self._query_ids: list[str] = []
        self._query_texts: dict[str, str] = {}
        self._qrels_dict: dict[str, dict[str, int]] = {}
        self._is_loaded: bool = False

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._query_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._ensure_loaded()
        qid: str = self._query_ids[idx]
        query_text: str = self._query_texts.get(qid, "")
        return {
            "qid": qid,
            "query_text": query_text,
            "relevance_judgments": self._qrels_dict.get(qid, {}),
        }

    # --- Property methods ---
    @property
    def collator(self) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        return self._collate

    # --- Protected methods ---
    def _ensure_loaded(self) -> None:
        if self._is_loaded:
            return
        self._load_queries()
        self._load_qrels()
        self._is_loaded = True

    def _resolve_ir_base_name(self) -> str:
        beir_dataset: str | None = self.cfg.beir_dataset
        if beir_dataset:
            return f"beir/{str(beir_dataset).lower()}"
        hf_name: str | None = self.cfg.hf_name
        if hf_name:
            normalized: str = str(hf_name)
            if normalized.lower().startswith("beir/"):
                return f"beir/{normalized.split('/', 1)[1].lower()}"
            return normalized
        raise ValueError("dataset.beir_dataset or dataset.hf_name must be set.")

    def _resolve_ir_split(self) -> str:
        split_value: str | None = getattr(self.cfg, "split", None)
        if split_value:
            return str(split_value).lower()
        hf_split: str | None = getattr(self.cfg, "hf_split", None)
        if hf_split:
            return str(hf_split).lower()
        return "test"

    def _resolve_eval_dataset(self) -> Any:
        if self._eval_dataset is not None:
            return self._eval_dataset
        candidates: list[str] = []
        if self._ir_split:
            candidates.append(f"{self._ir_base_name}/{self._ir_split}")
        candidates.append(self._ir_base_name)
        errors: list[str] = []
        for name in candidates:
            try:
                dataset: Any = ir_datasets.load(name)
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f"{name}: {exc}")
                continue
            has_queries: bool = hasattr(dataset, "queries_iter")
            has_qrels: bool = hasattr(dataset, "qrels_iter")
            if has_queries and has_qrels:
                self._eval_dataset = dataset
                self._ir_eval_name = name
                return dataset
            errors.append(f"{name}: missing queries or qrels")
        raise ValueError(
            "Unable to resolve ir_datasets BEIR queries/qrels. "
            f"Tried: {', '.join(candidates)}. Errors: {errors}"
        )

    def _extract_query_text(self, query: Any) -> str:
        text_value: Any | None = getattr(query, "text", None)
        if text_value:
            return str(text_value)
        fallback_value: Any | None = getattr(query, "query", None)
        if fallback_value:
            return str(fallback_value)
        return ""

    def _load_queries(self) -> None:
        dataset: Any = self._resolve_eval_dataset()
        queries_iter: Iterable[Any] = dataset.queries_iter()
        skip_samples: int = int(self.cfg.hf_skip_samples)
        max_samples: int | None = self.cfg.hf_max_samples
        loaded: int = 0
        for idx, query in enumerate(queries_iter):
            if idx < skip_samples:
                continue
            if max_samples is not None and loaded >= int(max_samples):
                break
            query_id: str = str(getattr(query, "query_id", idx))
            query_text: str = self._extract_query_text(query)
            self._query_ids.append(query_id)
            self._query_texts[query_id] = query_text
            loaded += 1
        log_if_rank_zero(
            logger,
            f"Loaded {len(self._query_ids)} queries from {self._ir_eval_name or self._ir_base_name}.",
        )

    def _load_qrels(self) -> None:
        dataset: Any = self._resolve_eval_dataset()
        allowed_queries: set[str] = set(self._query_ids)
        qrels_dict: dict[str, dict[str, int]] = {}
        for qrel in dataset.qrels_iter():
            qid_value: Any | None = getattr(qrel, "query_id", None)
            if qid_value is None:
                continue
            qid: str = str(qid_value)
            if qid not in allowed_queries:
                continue
            doc_value: Any | None = getattr(qrel, "doc_id", None)
            if doc_value is None:
                continue
            doc_id: str = str(doc_value)
            relevance_value: Any | None = getattr(qrel, "relevance", None)
            relevance: int = int(relevance_value) if relevance_value is not None else 0
            qrels_dict.setdefault(qid, {})[doc_id] = relevance
        self._qrels_dict = qrels_dict
        log_if_rank_zero(
            logger,
            f"Loaded qrels for {len(self._qrels_dict)} queries from {self._ir_eval_name or self._ir_base_name}.",
        )

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

    # --- Public methods ---
    def prepare_data(self) -> None:
        self._ensure_loaded()

    def setup(self) -> None:
        self._ensure_loaded()
