from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

import ir_datasets
import torch
from omegaconf import DictConfig
from torch.utils.data import IterableDataset, get_worker_info

from src.utils.logging import log_if_rank_zero

logger: logging.Logger = logging.getLogger("IRCorpusDataset")


class IRCorpusDataset(IterableDataset[dict[str, Any]]):
    """Iterable corpus dataset backed by ir_datasets."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig, global_cfg: DictConfig | None) -> None:
        super().__init__()
        resolved_global_cfg: DictConfig = global_cfg or cfg
        self.cfg: DictConfig = cfg
        self.global_cfg: DictConfig = resolved_global_cfg
        self._ir_name: str = self._resolve_ir_name()
        self._dataset: Any = ir_datasets.load(self._ir_name)
        self._docs_count: int | None = None

    def __len__(self) -> int:
        total_len: int = self._resolve_total_len()
        world_size: int = self._resolve_world_size()
        if world_size <= 1:
            return total_len
        rank: int = self._resolve_rank()
        base: int = total_len // world_size
        remainder: int = total_len % world_size
        return base + (1 if rank < remainder else 0)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        docs_iter: Iterator[Any] = self._dataset.docs_iter()
        skip: int = int(self.cfg.hf_skip_samples)
        max_samples: int | None = self.cfg.hf_max_samples
        max_index: int | None = None
        if max_samples is not None:
            max_index = skip + int(max_samples)

        rank: int = self._resolve_rank()
        world_size: int = self._resolve_world_size()
        worker_info: Any | None = get_worker_info()
        worker_id: int = int(worker_info.id) if worker_info is not None else 0
        num_workers: int = (
            int(worker_info.num_workers) if worker_info is not None else 1
        )
        shard_id: int = rank * num_workers + worker_id
        num_shards: int = world_size * num_workers

        for idx, doc in enumerate(docs_iter):
            if idx < skip:
                continue
            if max_index is not None and idx >= max_index:
                break
            # Shard the global stream across ranks and workers deterministically.
            global_idx: int = idx - skip
            if num_shards > 1 and global_idx % num_shards != shard_id:
                continue
            doc_id: str = self._extract_doc_id(doc)
            text: str = self._build_doc_text(doc)
            yield {"doc_id": doc_id, "text": text}

    # --- Public methods ---
    @property
    def collator(self) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        return self._collate

    def prepare_data(self) -> None:
        docs_iter: Iterator[Any] = self._dataset.docs_iter()
        try:
            first_doc: Any = next(docs_iter)
            _ = first_doc
        except StopIteration:
            return
        log_if_rank_zero(logger, f"Prepared ir_datasets corpus {self._ir_name}.")

    def setup(self) -> None:
        doc_count: int = self._resolve_total_len()
        log_if_rank_zero(
            logger, f"Loaded {doc_count} corpus documents from {self._ir_name}."
        )

    # --- Protected methods ---
    def _resolve_ir_name(self) -> str:
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

    def _resolve_docs_count(self) -> int:
        if self._docs_count is not None:
            return self._docs_count
        docs_count: int = int(self._dataset.docs_count())
        self._docs_count = docs_count
        return docs_count

    def _resolve_total_len(self) -> int:
        docs_count: int = self._resolve_docs_count()
        skip: int = int(self.cfg.hf_skip_samples)
        max_samples: int | None = self.cfg.hf_max_samples
        start: int = min(skip, docs_count)
        if max_samples is None:
            return max(0, docs_count - start)
        return max(0, min(docs_count - start, int(max_samples)))

    def _resolve_rank(self) -> int:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return 0
        return int(torch.distributed.get_rank())

    def _resolve_world_size(self) -> int:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return 1
        return int(torch.distributed.get_world_size())

    def _extract_doc_id(self, doc: Any) -> str:
        doc_id: Any | None = getattr(doc, "doc_id", None)
        if doc_id is None:
            doc_id = getattr(doc, "docno", None) or getattr(doc, "id", None)
        if doc_id is None:
            raise ValueError("Unable to resolve doc_id from ir_datasets record.")
        return str(doc_id)

    def _build_doc_text(self, doc: Any) -> str:
        title: str = str(getattr(doc, "title", "") or "").strip()
        text: str = str(getattr(doc, "text", "") or "").strip()
        if not text:
            text = str(getattr(doc, "body", "") or "").strip()
        if title and text:
            return f"{title} {text}"
        return title or text

    def _collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        doc_ids: list[str] = [str(item["doc_id"]) for item in batch]
        texts: list[str] = [str(item["text"]) for item in batch]
        return {"doc_ids": doc_ids, "doc_texts": texts}
