from __future__ import annotations

import itertools
import logging
import os
import random
import time
from functools import cached_property
from numbers import Number
from typing import Any, Iterable, Iterator, Mapping

import pyarrow as pa
import torch
from datasets import Dataset, IterableDataset, load_dataset, load_from_disk
from omegaconf import DictConfig
from torch.utils.data import get_worker_info
from transformers import PreTrainedTokenizerBase

from src.data.collators import RerankingCollator
from src.data.dataset.base import BaseDataset
from src.data.dataclass import RerankingDataItem
from src.data.utils import (
    build_integer_id_cache_key,
    id_to_idx,
    resolve_dataset_column,
    resolve_integer_id_cache_dir,
)
from src.utils.logging import log_if_rank_zero

logger: logging.Logger = logging.getLogger("MSMARCO")


def _normalize_optional_str(value: Any) -> str | None:
    """Normalize optional string values from configs."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized: str = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
    return str(value)


def _load_hf_dataset(
    hf_name: str,
    hf_subset: str | None,
    split: str,
    cache_dir: str | None,
    streaming: bool,
    data_files: Mapping[str, Any] | None,
) -> Any:
    """Load a Hugging Face dataset with optional data_files support."""
    if data_files:
        return load_dataset(
            hf_name,
            name=hf_subset,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
            data_files=dict(data_files),
        )
    return load_dataset(
        hf_name,
        name=hf_subset,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )


class MSMARCO(BaseDataset):

    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        load_teacher_scores: bool | None = None,
        require_teacher_scores: bool | None = None,
    ) -> None:
        BaseDataset.__init__(self, cfg=cfg, global_cfg=global_cfg, tokenizer=tokenizer)

        self.hf_name: str = str(cfg.hf_name)
        self.hf_subset: str | None = _normalize_optional_str(
            getattr(cfg, "hf_subset", None)
        )
        self.hf_split: str = str(cfg.hf_split)
        self.hf_text_name: str | None = _normalize_optional_str(
            getattr(cfg, "hf_text_name", None)
        )
        self.hf_data_files: Mapping[str, Any] | None = getattr(
            cfg, "hf_data_files", None
        )
        self.hf_cache_dir: str | None = getattr(cfg, "hf_cache_dir", None)
        self.hf_max_samples: int | None = getattr(cfg, "hf_max_samples", None)
        skip_samples_value: int = int(getattr(cfg, "hf_skip_samples", 0) or 0)
        if skip_samples_value < 0:
            raise ValueError("hf_skip_samples must be >= 0.")
        self.hf_skip_samples: int = skip_samples_value
        self.hf_teacher_name: str | None = _normalize_optional_str(
            getattr(cfg, "hf_teacher_name", None)
        )
        self.hf_teacher_subset: str | None = _normalize_optional_str(
            getattr(cfg, "hf_teacher_subset", None)
        )
        self.hf_teacher_split: str = str(getattr(cfg, "hf_teacher_split", "train"))
        self.hf_teacher_data_files: Mapping[str, Any] | None = getattr(
            cfg, "hf_teacher_data_files", None
        )
        self.hf_teacher_cache_dir: str | None = getattr(
            cfg, "hf_teacher_cache_dir", None
        )
        self.hf_teacher_max_samples: int | None = getattr(
            cfg, "hf_teacher_max_samples", None
        )
        self.use_integer_ids: bool = bool(getattr(cfg, "use_integer_ids", False))
        self.integer_id_cache_dir: str | None = getattr(
            cfg, "integer_id_cache_dir", None
        )
        self.query_idx_column: str = "query_idx"
        self.pos_idx_column: str = "positive_idx"
        self.neg_idx_column: str = "negative_idx"
        self.doc_idxs_column: str = "doc_idxs"
        self._integer_id_cache_path: str | None = None

        self.num_positives: int = int(cfg.num_positives)
        self.num_negatives: int = int(cfg.num_negatives)

        distill_cfg: Any = global_cfg.training.distill
        if load_teacher_scores is None:
            load_teacher_scores = bool(distill_cfg.enabled)
        self.load_teacher_scores: bool = bool(load_teacher_scores)
        if require_teacher_scores is None:
            self.require_teacher_scores: bool = bool(
                self.load_teacher_scores and distill_cfg.fail_on_missing
            )
        else:
            self.require_teacher_scores = bool(require_teacher_scores)
        self.teacher_score_key: str = str(distill_cfg.teacher_score_key)

        self.shuffle_buffer_size: int = int(
            getattr(cfg, "hf_shuffle_buffer_size", 0) or 0
        )
        self.seed: int = int(global_cfg.seed)

        self.dataset: Any | None = None
        self._length: int | None = None
        self.teacher_scores: dict[tuple[str, str], float] = {}
        self._collator: RerankingCollator | None = None
        self._stream_dataset: Any | None = None
        self._stream_iterator: Iterator[dict[str, Any]] | None = None
        self._stream_next_index: int = 0

    def __len__(self) -> int:
        self._assert_setup()
        if self._length is None:
            raise TypeError("Length is not known for this streaming dataset.")
        total_shards, shard_index = self._get_shard_context()
        if total_shards <= 1:
            return self._length
        base, remainder = divmod(self._length, total_shards)
        return base + (1 if shard_index < remainder else 0)

    def __iter__(self) -> Iterator[RerankingDataItem]:
        self._assert_setup()
        dataset: Any = self._prepare_stream(self.dataset)
        for index, row in enumerate(dataset):
            item: RerankingDataItem = self._row_to_item(row, index)
            # Guard against malformed samples that would collapse training.
            doc_mask: torch.Tensor = item.doc_mask
            pos_mask: torch.Tensor = item.pos_mask
            if doc_mask.numel() == 0 or not bool(doc_mask.any()):
                raise ValueError(
                    f"MSMARCO sample {index} (qid={item.qid}) has no documents."
                )
            if pos_mask.numel() == 0 or not bool(pos_mask.any()):
                raise ValueError(
                    f"MSMARCO sample {index} (qid={item.qid}) has no positives."
                )
            yield item

    def __getitem__(self, index: int) -> RerankingDataItem:
        self._assert_setup()
        row: dict[str, Any]
        if isinstance(self.dataset, IterableDataset):
            row = self._get_stream_row(index)
        else:
            # Map-style datasets allow direct indexing (faster than streaming).
            row = self.dataset[index]
        item: RerankingDataItem = self._row_to_item(row, index)
        # Guard against malformed samples that would collapse training.
        doc_mask: torch.Tensor = item.doc_mask
        pos_mask: torch.Tensor = item.pos_mask
        if doc_mask.numel() == 0 or not bool(doc_mask.any()):
            raise ValueError(
                f"MSMARCO sample {index} (qid={item.qid}) has no documents."
            )
        if pos_mask.numel() == 0 or not bool(pos_mask.any()):
            raise ValueError(
                f"MSMARCO sample {index} (qid={item.qid}) has no positives."
            )
        return item

    @property
    def collator(self) -> RerankingCollator:
        if self._collator is None:
            max_padding: bool = bool(getattr(self.cfg, "max_padding", False))
            max_query_length: int = int(self.cfg.max_query_length)
            max_doc_length: int = int(self.cfg.max_doc_length)
            max_docs: int = int(self.num_positives + self.num_negatives)
            self._collator = RerankingCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                require_teacher_scores=self.require_teacher_scores,
                max_padding=max_padding,
                max_query_length=max_query_length,
                max_doc_length=max_doc_length,
                max_docs=max_docs,
            )
        return self._collator

    @cached_property
    def query_dataset(self) -> Any:
        text_name: str = self.hf_text_name or self.hf_name
        try:
            log_if_rank_zero(
                logger,
                f"Loading HF queries dataset: name={text_name} subset=queries split=train "
                "streaming=False",
            )
            return _load_hf_dataset(
                text_name,
                "queries",
                split="train",
                cache_dir=self.hf_cache_dir,
                streaming=False,
                data_files=None,
            )
        except Exception:  # pylint: disable=broad-except
            log_if_rank_zero(
                logger,
                f"Loading HF queries dataset fallback: name={self.hf_name} "
                f"subset={self.hf_subset} split={self.hf_split} streaming=False",
            )
            return _load_hf_dataset(
                self.hf_name,
                self.hf_subset,
                split=self.hf_split,
                cache_dir=self.hf_cache_dir,
                streaming=False,
                data_files=self.hf_data_files,
            )

    @cached_property
    def corpus_dataset(self) -> Any:
        text_name: str = self.hf_text_name or self.hf_name
        try:
            log_if_rank_zero(
                logger,
                f"Loading HF corpus dataset: name={text_name} subset=corpus split=train "
                "streaming=False",
            )
            return _load_hf_dataset(
                text_name,
                "corpus",
                split="train",
                cache_dir=self.hf_cache_dir,
                streaming=False,
                data_files=None,
            )
        except Exception:  # pylint: disable=broad-except
            log_if_rank_zero(
                logger,
                f"Loading HF corpus dataset fallback: name={self.hf_name} "
                f"subset={self.hf_subset} split={self.hf_split} streaming=False",
            )
            return _load_hf_dataset(
                self.hf_name,
                self.hf_subset,
                split=self.hf_split,
                cache_dir=self.hf_cache_dir,
                streaming=False,
                data_files=self.hf_data_files,
            )

    @cached_property
    def query_id_column(self) -> str:
        return self._resolve_id_column(
            self.query_dataset.column_names, ("query_id", "qid", "_id", "id")
        )

    @cached_property
    def corpus_id_column(self) -> str:
        return self._resolve_id_column(
            self.corpus_dataset.column_names,
            ("doc_id", "corpus_id", "passage_id", "_id", "id"),
        )

    @cached_property
    def query_text_column(self) -> str:
        column_names: Iterable[str] = self.query_dataset.column_names
        return self._resolve_text_column(column_names, ("text", "query"))

    @cached_property
    def corpus_text_column(self) -> str:
        column_names: Iterable[str] = self.corpus_dataset.column_names
        return self._resolve_text_column(column_names, ("text", "passage", "contents"))

    @cached_property
    def query_id_to_idx(self) -> dict[str, int]:
        """
        Create a mapping from query IDs to their indices in the query dataset.
        """
        query_rows: int = len(self.query_dataset)
        log_if_rank_zero(
            logger,
            f"Building query id index map: column={self.query_id_column} rows={query_rows} "
            "(first run may take a while).",
        )
        # Use resolve_dataset_column() for fast PyArrow access that respects filtering.
        # Note: Direct .data.column() returns the underlying PyArrow table which may
        # contain all original rows for filtered datasets, causing index mismatches.
        column: pa.Array | pa.ChunkedArray = resolve_dataset_column(
            self.query_dataset, self.query_id_column
        )
        mapping_desc: str = "Mapping query ids to indices"
        enable_tqdm: bool = False
        query_id_to_idx_map: dict[str, int] = id_to_idx(
            column,
            mapping_desc,
            enable_tqdm,
        )
        return query_id_to_idx_map

    @cached_property
    def corpus_id_to_idx(self) -> dict[str, int]:
        """
        Create a mapping from document IDs to their indices in the corpus dataset.
        """
        corpus_rows: int = len(self.corpus_dataset)
        log_if_rank_zero(
            logger,
            f"Building corpus id index map: column={self.corpus_id_column} rows={corpus_rows} "
            "(first run may take a while).",
        )
        # Use resolve_dataset_column() for fast PyArrow access that respects filtering.
        # Note: Direct .data.column() returns the underlying PyArrow table which may
        # contain all original rows for filtered datasets, causing index mismatches.
        column: pa.Array | pa.ChunkedArray = resolve_dataset_column(
            self.corpus_dataset, self.corpus_id_column
        )
        mapping_desc: str = "Mapping corpus ids to indices"
        enable_tqdm: bool = False
        corpus_id_to_idx_map: dict[str, int] = id_to_idx(
            column,
            mapping_desc,
            enable_tqdm,
        )
        return corpus_id_to_idx_map

    def _resolve_integer_id_cache_path(self) -> str | None:
        if not self.use_integer_ids:
            return None
        if self._integer_id_cache_path is not None:
            return self._integer_id_cache_path
        cache_key: str = build_integer_id_cache_key(
            hf_name=self.hf_name,
            hf_subset=self.hf_subset or "",
            hf_split=self.hf_split,
            hf_skip_samples=self.hf_skip_samples,
            query_id_column=self.query_id_column,
            corpus_id_column=self.corpus_id_column,
            hf_max_samples=self.hf_max_samples,
        )
        cache_dir: str = resolve_integer_id_cache_dir(
            hf_cache_dir=self.hf_cache_dir,
            integer_id_cache_dir=self.integer_id_cache_dir,
        )
        # Keep integer-id artifacts grouped under a dedicated subdirectory.
        cache_path: str = os.path.join(cache_dir, "splade_integer_ids", cache_key)
        self._integer_id_cache_path = cache_path
        return cache_path

    def _ensure_integer_id_cache(self) -> None:
        if not self.use_integer_ids:
            return
        cache_path: str | None = self._resolve_integer_id_cache_path()
        if cache_path is None:
            return
        try:
            import torch.distributed as dist

            is_dist_ready: bool = bool(dist.is_available() and dist.is_initialized())
            if is_dist_ready:
                rank: int = int(dist.get_rank())
                if rank == 0 and not os.path.isdir(cache_path):
                    self.prepare_integer_id_cache()
                self._wait_for_integer_id_cache(cache_path)
            else:
                if not os.path.isdir(cache_path):
                    self.prepare_integer_id_cache()
        except Exception as exc:  # pylint: disable=broad-except
            log_if_rank_zero(
                logger,
                f"Integer-id cache synchronization failed: {exc}. "
                "Falling back to streaming dataset load.",
                level="warning",
            )

    def _build_id_to_idx_map(
        self, dataset: Dataset, id_column: str, mapping_label: str
    ) -> dict[str, int]:
        row_count: int = int(len(dataset))
        log_if_rank_zero(
            logger,
            f"Building {mapping_label} id index map: column={id_column} rows={row_count} "
            "(one-time cost).",
        )
        column: pa.Array | pa.ChunkedArray = resolve_dataset_column(dataset, id_column)
        mapping_desc: str = f"Mapping {mapping_label} ids"
        enable_tqdm: bool = False
        id_to_idx_map: dict[str, int] = id_to_idx(column, mapping_desc, enable_tqdm)
        return id_to_idx_map

    def _wait_for_integer_id_cache(
        self, cache_path: str, timeout_seconds: int = 7200, poll_seconds: int = 5
    ) -> bool:
        if os.path.isdir(cache_path):
            return True
        log_if_rank_zero(
            logger, f"Waiting for integer-id cache to appear: {cache_path}"
        )
        waited_seconds: int = 0
        while waited_seconds < timeout_seconds:
            time.sleep(poll_seconds)
            waited_seconds += poll_seconds
            if os.path.isdir(cache_path):
                return True
            if waited_seconds % 60 == 0:
                log_if_rank_zero(
                    logger,
                    f"Still waiting for integer-id cache ({waited_seconds} seconds)...",
                )
        log_if_rank_zero(
            logger,
            f"Timed out waiting for integer-id cache after {waited_seconds} seconds: "
            f"{cache_path}",
            level="warning",
        )
        return False

    def _assert_setup(self) -> None:
        if self.dataset is None:
            raise RuntimeError("Dataset is not set up. Call setup() first.")

    def _reset_stream_state(self) -> None:
        self._stream_dataset = self._prepare_stream(self.dataset)
        self._stream_iterator = iter(self._stream_dataset)
        self._stream_next_index = 0

    def _advance_stream_iterator(self) -> dict[str, Any]:
        if self._stream_iterator is None:
            raise RuntimeError("Streaming iterator is not initialized.")
        try:
            row: dict[str, Any] = next(self._stream_iterator)
        except StopIteration as exc:
            raise IndexError("Streaming dataset is exhausted.") from exc
        self._stream_next_index += 1
        return row

    def _get_stream_row(self, index: int) -> dict[str, Any]:
        if index == 0:
            self._reset_stream_state()
        if self._stream_iterator is None or self._stream_dataset is None:
            self._reset_stream_state()
        if index < self._stream_next_index:
            self._reset_stream_state()
        while self._stream_next_index < index:
            self._advance_stream_iterator()
        return self._advance_stream_iterator()

    @staticmethod
    def _has_inline_triplets(column_names: Iterable[str]) -> bool:
        columns: set[str] = set(column_names)
        has_query_text: bool = "query" in columns or "anchor" in columns
        has_doc_texts: bool = "positive" in columns and "negative" in columns
        # Inline triplets must include query/anchor and doc texts; id-only datasets
        # should return False so we load external query/corpus maps.
        return bool(has_query_text and has_doc_texts)

    @staticmethod
    def _resolve_id_column(
        column_names: Iterable[str], candidates: tuple[str, ...]
    ) -> str:
        for name in candidates:
            if name in column_names:
                return name
        raise ValueError(f"Unable to resolve ID column from {column_names}")

    @staticmethod
    def _resolve_text_column(
        column_names: Iterable[str], candidates: tuple[str, ...]
    ) -> str:
        columns: set[str] = set(column_names)
        for name in candidates:
            if name in columns:
                return name
        raise ValueError(f"Unable to resolve text column from {column_names}")

    def _parse_inline_scores(
        self, score_values: Any, doc_ids: list[str | None]
    ) -> list[float] | None:
        if score_values is None:
            return None
        if isinstance(score_values, (list, tuple)):
            if len(score_values) == len(doc_ids):
                return [float(score) for score in score_values]
            return None
        if isinstance(score_values, Number):
            if len(doc_ids) == 1:
                return [float(score_values)]
            return None
        return None

    def _get_query_text(self, qid: str, default: str | None = None) -> str:
        query_idx: int = self.query_id_to_idx[qid]
        row: dict[str, Any] = self.query_dataset[query_idx]
        text_value: Any = row.get(self.query_text_column)
        if text_value is None:
            return "" if default is None else default
        return str(text_value)

    def _get_query_text_by_idx(self, query_idx: int, default: str | None = None) -> str:
        if query_idx < 0:
            return "" if default is None else default
        row: dict[str, Any] = self.query_dataset[int(query_idx)]
        text_value: Any = row.get(self.query_text_column)
        if text_value is None:
            return "" if default is None else default
        return str(text_value)

    def _get_doc_text(self, doc_id: str, default: str = "") -> str:
        doc_idx: int = self.corpus_id_to_idx[doc_id]
        row: dict[str, Any] = self.corpus_dataset[doc_idx]
        text_value: Any = row.get(self.corpus_text_column)
        if text_value is None:
            return default
        return str(text_value)

    def _get_doc_text_by_idx(self, doc_idx: int, default: str = "") -> str:
        if doc_idx < 0:
            return default
        row: dict[str, Any] = self.corpus_dataset[int(doc_idx)]
        text_value: Any = row.get(self.corpus_text_column)
        if text_value is None:
            return default
        return str(text_value)

    def _row_to_item(self, row: dict[str, Any], index: int) -> RerankingDataItem:
        score_values: Any | None = None
        query_text: str = ""
        pos_texts: list[str] = []
        neg_texts: list[str] = []
        pos_ids: list[str | None] = []
        neg_ids: list[str | None] = []
        qid: str = ""

        if "query" in row and "positive" in row and "negative" in row:
            query_text = str(row["query"])
            pos_texts = [str(row["positive"])]
            neg_texts = [str(row["negative"])]
            pos_ids = [None]
            neg_ids = [None]
            qid = str(row.get("query_id") or row.get("qid") or index)
            score_values = row.get("score") or row.get("scores")
        elif "anchor" in row and "positive" in row and "negative" in row:
            query_text = str(row["anchor"])
            pos_texts = [str(row["positive"])]
            neg_texts = [str(row["negative"])]
            pos_ids = [None]
            neg_ids = [None]
            qid = str(row.get("query_id") or row.get("qid") or index)
            score_values = row.get("score") or row.get("scores")
        elif (
            self.use_integer_ids
            and self.query_idx_column in row
            and self.pos_idx_column in row
            and self.neg_idx_column in row
        ):
            qid = str(row.get("query_id") or row.get("qid") or index)
            query_idx: int = int(row[self.query_idx_column])
            pos_idx: int = int(row[self.pos_idx_column])
            neg_idx: int = int(row[self.neg_idx_column])
            query_text = self._get_query_text_by_idx(query_idx)
            pos_id_value: Any = row.get("positive_id")
            neg_id_value: Any = row.get("negative_id")
            pos_ids = [pos_id_value]
            neg_ids = [neg_id_value]
            pos_texts = [self._get_doc_text_by_idx(pos_idx)]
            neg_texts = [self._get_doc_text_by_idx(neg_idx)]
            score_values = row.get("score") or row.get("scores")
        elif (
            self.use_integer_ids
            and self.query_idx_column in row
            and self.doc_idxs_column in row
            and "doc_ids" in row
            and "labels" in row
        ):
            qid = str(row.get("query_id") or row.get("qid") or index)
            query_idx: int = int(row[self.query_idx_column])
            query_text = self._get_query_text_by_idx(query_idx, default="")
            row_doc_ids: list[str] = [str(doc_id) for doc_id in row["doc_ids"]]
            row_doc_idxs: list[int] = [
                int(doc_idx) for doc_idx in row[self.doc_idxs_column]
            ]
            labels: list[float] = list(row["labels"])
            pos_pairs: list[tuple[str, int]] = []
            neg_pairs: list[tuple[str, int]] = []
            for doc_id, doc_idx, label in zip(row_doc_ids, row_doc_idxs, labels):
                if label > 0:
                    pos_pairs.append((doc_id, doc_idx))
                else:
                    neg_pairs.append((doc_id, doc_idx))
            pos_pairs = self._sample_pairs(pos_pairs, self.num_positives)
            neg_pairs = self._sample_pairs(neg_pairs, self.num_negatives)
            pos_ids = [doc_id for doc_id, _ in pos_pairs]
            neg_ids = [doc_id for doc_id, _ in neg_pairs]
            pos_texts = [self._get_doc_text_by_idx(doc_idx) for _, doc_idx in pos_pairs]
            neg_texts = [self._get_doc_text_by_idx(doc_idx) for _, doc_idx in neg_pairs]
            score_values = row.get("score") or row.get("scores")
            if isinstance(score_values, (list, tuple)) and len(score_values) == len(
                row_doc_ids
            ):
                score_map: dict[str, float] = {
                    doc_id: float(score)
                    for doc_id, score in zip(row_doc_ids, score_values)
                }
                score_values = [
                    score_map.get(doc_id, float("nan"))
                    for doc_id in (pos_ids + neg_ids)
                ]
        elif "query_id" in row and "positive_id" in row:
            qid = str(row["query_id"])
            query_text = self._get_query_text(qid)
            pos_ids = [str(row["positive_id"])]
            neg_ids = [str(row["negative_id"])]
            pos_texts = [self._get_doc_text(pos_ids[0])]
            neg_texts = [self._get_doc_text(neg_ids[0])]
            score_values = row.get("score") or row.get("scores")
        elif "query_id" in row and "doc_ids" in row and "labels" in row:
            qid = str(row["query_id"])
            query_text = self._get_query_text(qid, default="")
            row_doc_ids: list[str] = [str(doc_id) for doc_id in row["doc_ids"]]
            labels: list[float] = list(row["labels"])
            pos_ids = [
                doc_id for doc_id, label in zip(row_doc_ids, labels) if label > 0
            ]
            neg_ids = [
                doc_id for doc_id, label in zip(row_doc_ids, labels) if label <= 0
            ]
            pos_ids = self._sample_ids(pos_ids, self.num_positives)
            neg_ids = self._sample_ids(neg_ids, self.num_negatives)
            pos_texts = [self._get_doc_text(doc_id) for doc_id in pos_ids]
            neg_texts = [self._get_doc_text(doc_id) for doc_id in neg_ids]
            score_values = row.get("score") or row.get("scores")
            if isinstance(score_values, (list, tuple)) and len(score_values) == len(
                row_doc_ids
            ):
                score_map: dict[str, float] = {
                    doc_id: float(score)
                    for doc_id, score in zip(row_doc_ids, score_values)
                }
                score_values = [
                    score_map.get(doc_id, float("nan"))
                    for doc_id in (pos_ids + neg_ids)
                ]
        else:
            raise ValueError(f"Unsupported MSMARCO HF row format: {row.keys()}")

        docs: list[str] = pos_texts + neg_texts
        doc_ids: list[str | None] = pos_ids + neg_ids
        row_teacher_scores: list[float] | None = self._parse_inline_scores(
            score_values, doc_ids
        )
        if (
            score_values is not None
            and row_teacher_scores is None
            and self.require_teacher_scores
        ):
            raise ValueError(
                "Inline teacher scores were provided but could not be aligned to "
                f"{len(doc_ids)} document(s)."
            )

        if row_teacher_scores is not None:
            teacher_scores: list[float] = row_teacher_scores
        else:
            teacher_scores = [
                self._get_teacher_score(qid, doc_id) for doc_id in doc_ids
            ]

        if self.require_teacher_scores and any(
            score != score for score in teacher_scores
        ):
            raise ValueError(f"Missing teacher score in HF sample for query {qid}")

        max_padding: bool = bool(getattr(self.cfg, "max_padding", False))
        # Use max_length padding to keep fixed shapes when enabled.
        query_padding: str | bool = "max_length" if max_padding else True
        doc_padding: str | bool = "max_length" if max_padding else True
        max_query_length: int = int(self.cfg.max_query_length)
        max_doc_length: int = int(self.cfg.max_doc_length)

        query_tokens: Any = self.tokenizer(
            query_text,
            padding=query_padding,
            truncation=True,
            max_length=max_query_length,
            return_tensors="pt",
        )
        query_input_ids: torch.Tensor = query_tokens["input_ids"].squeeze(0)
        query_attention_mask: torch.Tensor = query_tokens["attention_mask"].squeeze(0)

        if docs:
            doc_tokens: Any = self.tokenizer(
                docs,
                padding=doc_padding,
                truncation=True,
                max_length=max_doc_length,
                return_tensors="pt",
            )
            doc_input_ids: torch.Tensor = doc_tokens["input_ids"]
            doc_attention_mask: torch.Tensor = doc_tokens["attention_mask"]
        else:
            doc_input_ids = torch.empty((0, max_doc_length), dtype=torch.long)
            doc_attention_mask = torch.empty((0, max_doc_length), dtype=torch.long)

        doc_mask: torch.Tensor = torch.zeros(len(docs), dtype=torch.bool)
        if docs:
            doc_mask[:] = True
        pos_mask: torch.Tensor = torch.zeros(len(docs), dtype=torch.bool)
        if pos_texts:
            pos_mask[: len(pos_texts)] = True
        teacher_score_tensor: torch.Tensor = torch.tensor(
            teacher_scores, dtype=torch.float
        )

        pos_ids = ["" if doc_id is None else str(doc_id) for doc_id in pos_ids]
        neg_ids = ["" if doc_id is None else str(doc_id) for doc_id in neg_ids]

        return RerankingDataItem(
            data_idx=index,
            qid=qid,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            query_text=query_text,
            doc_texts=docs,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            doc_mask=doc_mask,
            pos_mask=pos_mask,
            teacher_scores=teacher_score_tensor,
        )

    def _sample_ids(self, items: list[str], count: int) -> list[str]:
        if count <= 0:
            return []
        if len(items) <= count:
            return items
        return random.sample(items, count)

    def _sample_pairs(
        self, items: list[tuple[str, int]], count: int
    ) -> list[tuple[str, int]]:
        if count <= 0:
            return []
        if len(items) <= count:
            return items
        return random.sample(items, count)

    def _get_teacher_score(self, qid: str, doc_id: str | None) -> float:
        if doc_id is None:
            return float("nan")
        score: float | None = self.teacher_scores.get((qid, str(doc_id)))
        if score is None:
            return float("nan")
        return float(score)

    @staticmethod
    def _load_teacher_scores(
        hf_name: str | None,
        subset: str | None,
        split: str,
        hf_cache_dir: str | None,
        max_samples: int | None,
        data_files: Mapping[str, Any] | None,
    ) -> dict[tuple[str, str], float]:
        if hf_name is None or subset is None:
            return {}
        log_if_rank_zero(
            logger,
            f"Loading HF teacher scores dataset: name={hf_name} subset={subset} "
            f"split={split} streaming=False",
        )
        teacher_ds: Any = _load_hf_dataset(
            hf_name,
            subset,
            split=split,
            cache_dir=hf_cache_dir,
            streaming=False,
            data_files=data_files,
        )
        if max_samples is not None:
            teacher_ds = teacher_ds.select(range(min(max_samples, len(teacher_ds))))
        scores: dict[tuple[str, str], float] = {}
        for row in teacher_ds:
            qid: str = str(row.get("query_id") or row.get("qid") or row.get("_id"))
            if subset == "pair":
                doc_id: str = str(
                    row.get("corpus_id")
                    or row.get("doc_id")
                    or row.get("passage_id")
                    or row.get("pid")
                    or row.get("_id")
                )
                score: Any = row.get("score")
                if score is None:
                    continue
                scores[(qid, doc_id)] = float(score)
            elif subset == "triplet":
                pos_id: str = str(
                    row.get("positive_id") or row.get("pos_id") or row.get("doc_pos_id")
                )
                neg_id: str = str(
                    row.get("negative_id") or row.get("neg_id") or row.get("doc_neg_id")
                )
                score = row.get("score") or row.get("scores")
                if isinstance(score, (list, tuple)) and len(score) == 2:
                    scores[(qid, pos_id)] = float(score[0])
                    scores[(qid, neg_id)] = float(score[1])
            elif subset == "list":
                doc_ids: list[Any] = row.get("corpus_id") or row.get("doc_ids") or []
                score_list: list[Any] = row.get("score") or row.get("scores") or []
                for doc_id, score in zip(doc_ids, score_list):
                    scores[(qid, str(doc_id))] = float(score)
        return scores

    def _prepare_stream(self, dataset: Any) -> Any:
        if self.shuffle_buffer_size > 0:
            if isinstance(dataset, IterableDataset):
                dataset = dataset.shuffle(
                    buffer_size=self.shuffle_buffer_size, seed=self.seed
                )
            else:
                # Map-style datasets do not accept buffer_size in shuffle.
                dataset = dataset.shuffle(seed=self.seed)
        return self._shard_for_workers(dataset)

    def _get_shard_context(self) -> tuple[int, int]:
        worker_info = get_worker_info()
        num_workers: int = worker_info.num_workers if worker_info else 1
        worker_id: int = worker_info.id if worker_info else 0

        world_size: int = 1
        rank: int = 0
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
        except Exception:  # pylint: disable=broad-except
            pass

        total_shards: int = world_size * num_workers
        shard_index: int = rank * num_workers + worker_id
        return total_shards, shard_index

    def _shard_for_workers(self, dataset: Any) -> Any:
        total_shards, shard_index = self._get_shard_context()
        if total_shards <= 1:
            return dataset
        return dataset.shard(
            num_shards=total_shards, index=shard_index, contiguous=True
        )

    def _apply_hf_sample_window(self, dataset: Any) -> Any:
        skip_samples: int = int(self.hf_skip_samples)
        max_samples: int | None = self.hf_max_samples
        if skip_samples <= 0 and max_samples is None:
            return dataset
        if isinstance(dataset, IterableDataset):
            if skip_samples > 0:
                dataset = dataset.skip(skip_samples)
            if max_samples is not None:
                dataset = dataset.take(int(max_samples))
            return dataset
        dataset_length: int = int(len(dataset))
        start_index: int = min(skip_samples, dataset_length)
        end_index: int = dataset_length
        if max_samples is not None:
            end_index = min(start_index + int(max_samples), dataset_length)
        indices: range = range(start_index, end_index)
        return dataset.select(indices)

    def _resolve_length(
        self, hf_max_samples: int | None, hf_skip_samples: int
    ) -> int | None:
        if hf_max_samples is not None:
            return int(hf_max_samples)
        if self.dataset is not None:
            try:
                return int(len(self.dataset))
            except TypeError:
                # Iterable datasets do not implement __len__.
                pass
        try:
            split_info: Any = self.dataset.info.splits.get(self.hf_split)
            if split_info and split_info.num_examples:
                remaining: int = int(split_info.num_examples) - int(hf_skip_samples)
                return max(0, remaining)
        except Exception:  # pylint: disable=broad-except
            return None
        return None

    def prepare_integer_id_cache(self) -> None:
        if not self.use_integer_ids:
            return
        cache_path: str | None = self._resolve_integer_id_cache_path()
        if cache_path is None:
            log_if_rank_zero(
                logger, "Integer-id preprocessing skipped: cache path unavailable."
            )
            return
        if os.path.isdir(cache_path):
            log_if_rank_zero(logger, f"Integer-id cache hit: {cache_path}")
            return

        log_if_rank_zero(
            logger,
            f"Preparing integer-id cache for dataset: name={self.hf_name} "
            f"subset={self.hf_subset} split={self.hf_split}",
        )
        if self.hf_max_samples is not None:
            max_samples: int = int(self.hf_max_samples)
            skip_samples: int = int(self.hf_skip_samples)
            streaming_dataset: IterableDataset = _load_hf_dataset(
                self.hf_name,
                self.hf_subset,
                split=self.hf_split,
                cache_dir=self.hf_cache_dir,
                streaming=True,
                data_files=self.hf_data_files,
            )
            column_names: list[str] = list(streaming_dataset.column_names)
            start_index: int = skip_samples
            stop_index: int = start_index + max_samples
            rows: list[dict[str, Any]] = list(
                itertools.islice(streaming_dataset, start_index, stop_index)
            )
            dataset: Dataset = Dataset.from_list(rows)
        else:
            dataset: Dataset = _load_hf_dataset(
                self.hf_name,
                self.hf_subset,
                split=self.hf_split,
                cache_dir=self.hf_cache_dir,
                streaming=False,
                data_files=self.hf_data_files,
            )
            dataset = self._apply_hf_sample_window(dataset)
            column_names: list[str] = list(dataset.column_names)

        has_inline_triplets: bool = self._has_inline_triplets(column_names)
        if has_inline_triplets:
            log_if_rank_zero(
                logger,
                "Integer-id preprocessing skipped: inline triplets detected in "
                f"{column_names}.",
            )
            return
        if "query_id" not in column_names:
            log_if_rank_zero(
                logger,
                "Integer-id preprocessing skipped: missing query_id column in "
                f"{column_names}.",
            )
            return

        query_id_to_idx: dict[str, int] = self._build_id_to_idx_map(
            self.query_dataset, self.query_id_column, "query"
        )
        corpus_id_to_idx: dict[str, int] = self._build_id_to_idx_map(
            self.corpus_dataset, self.corpus_id_column, "corpus"
        )

        def _lookup_idx(mapping: dict[str, int], raw_id: Any) -> int:
            if raw_id is None:
                return -1
            return int(mapping.get(str(raw_id), -1))

        def _map_triplet_batch(batch: dict[str, list[Any]]) -> dict[str, list[int]]:
            query_ids: list[Any] = batch["query_id"]
            pos_ids: list[Any] = batch["positive_id"]
            neg_ids: list[Any] = batch["negative_id"]
            query_indices: list[int] = []
            pos_indices: list[int] = []
            neg_indices: list[int] = []
            for idx in range(len(query_ids)):
                query_indices.append(_lookup_idx(query_id_to_idx, query_ids[idx]))
                pos_indices.append(_lookup_idx(corpus_id_to_idx, pos_ids[idx]))
                neg_indices.append(_lookup_idx(corpus_id_to_idx, neg_ids[idx]))
            return {
                self.query_idx_column: query_indices,
                self.pos_idx_column: pos_indices,
                self.neg_idx_column: neg_indices,
            }

        def _map_list_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            query_ids: list[Any] = batch["query_id"]
            doc_ids_batch: list[Any] = batch["doc_ids"]
            query_indices: list[int] = []
            doc_indices: list[list[int]] = []
            for idx in range(len(query_ids)):
                query_indices.append(_lookup_idx(query_id_to_idx, query_ids[idx]))
                row_doc_ids: list[Any] = list(doc_ids_batch[idx] or [])
                row_doc_indices: list[int] = [
                    _lookup_idx(corpus_id_to_idx, doc_id) for doc_id in row_doc_ids
                ]
                doc_indices.append(row_doc_indices)
            return {
                self.query_idx_column: query_indices,
                self.doc_idxs_column: doc_indices,
            }

        map_fn: Any
        if "positive_id" in column_names and "negative_id" in column_names:
            map_fn = _map_triplet_batch
        elif "doc_ids" in column_names:
            map_fn = _map_list_batch
        else:
            log_if_rank_zero(
                logger,
                "Integer-id preprocessing skipped: unsupported columns "
                f"{column_names}.",
            )
            return

        desc: str = "Preprocessing integer IDs"
        processed_dataset: Dataset = dataset.map(
            map_fn,
            batched=True,
            desc=desc,
        )
        parent_dir: str = os.path.dirname(cache_path)
        os.makedirs(parent_dir, exist_ok=True)
        processed_dataset.save_to_disk(cache_path)

    def prepare_data(self) -> None:
        log_if_rank_zero(
            logger,
            f"Loading HF training dataset (streaming): name={self.hf_name} "
            f"subset={self.hf_subset} split={self.hf_split} streaming=True",
        )
        dataset: Any = _load_hf_dataset(
            self.hf_name,
            self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
            streaming=True,
            data_files=self.hf_data_files,
        )
        has_text_triplets: bool = self._has_inline_triplets(dataset.column_names)
        if not has_text_triplets:
            text_name: str = self.hf_text_name or self.hf_name
            log_if_rank_zero(
                logger,
                f"Loading HF queries dataset: name={text_name} subset=queries split=train "
                "streaming=False",
            )
            _load_hf_dataset(
                text_name,
                "queries",
                split="train",
                cache_dir=self.hf_cache_dir,
                streaming=False,
                data_files=None,
            )
            log_if_rank_zero(
                logger,
                f"Loading HF corpus dataset: name={text_name} subset=corpus split=train "
                "streaming=False",
            )
            _load_hf_dataset(
                text_name,
                "corpus",
                split="train",
                cache_dir=self.hf_cache_dir,
                streaming=False,
                data_files=None,
            )
        if self.load_teacher_scores and self.hf_teacher_name and self.hf_teacher_subset:
            raise ValueError(
                "Streaming mode does not support separate teacher score datasets. "
                "Disable distillation or use a dataset with inline score columns."
            )

    def setup(self) -> None:
        self._stream_dataset = None
        self._stream_iterator = None
        self._stream_next_index = 0
        self._ensure_integer_id_cache()
        cache_path: str | None = self._resolve_integer_id_cache_path()
        use_cached_dataset: bool = bool(
            self.use_integer_ids and cache_path and os.path.isdir(cache_path)
        )
        if use_cached_dataset:
            log_if_rank_zero(logger, f"Loading integer-id cached dataset: {cache_path}")
            self.dataset = load_from_disk(cache_path)
        else:
            if self.use_integer_ids:
                log_if_rank_zero(
                    logger,
                    "Integer-id cache miss. Falling back to streaming dataset load.",
                )
            log_if_rank_zero(
                logger,
                f"Loading HF training dataset (streaming): name={self.hf_name} "
                f"subset={self.hf_subset} split={self.hf_split} streaming=True",
            )
            self.dataset = _load_hf_dataset(
                self.hf_name,
                self.hf_subset,
                split=self.hf_split,
                cache_dir=self.hf_cache_dir,
                streaming=True,
                data_files=self.hf_data_files,
            )
            self.dataset = self._apply_hf_sample_window(self.dataset)

        self._length = self._resolve_length(
            self.hf_max_samples,
            self.hf_skip_samples,
        )

        has_inline_scores: bool = any(
            key in self.dataset.column_names for key in ("score", "scores")
        )
        if self.load_teacher_scores and self.hf_teacher_name and self.hf_teacher_subset:
            raise ValueError(
                "Streaming mode does not support separate teacher score datasets. "
                "Disable distillation or use a dataset with inline score columns."
            )
        if self.require_teacher_scores and not has_inline_scores:
            raise ValueError(
                "Distillation requires teacher scores, but streaming dataset does not "
                "include score columns. Disable distillation or use inline scores."
            )

        self.teacher_scores = {}
