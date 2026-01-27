from __future__ import annotations

import logging
import random
from functools import cached_property
from numbers import Number
from typing import Any, Iterable, Iterator

import pyarrow as pa
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import get_worker_info
from transformers import PreTrainedTokenizerBase

from src.data.collators import RerankingCollator
from src.data.dataset.base import BaseDataset
from src.data.dataclass import RerankingDataItem
from src.data.utils import id_to_idx, resolve_dataset_column

logger: logging.Logger = logging.getLogger("MSMARCO")


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
        self.hf_subset: str = str(cfg.hf_subset)
        self.hf_split: str = str(cfg.hf_split)
        self.hf_text_name: str | None = getattr(cfg, "hf_text_name", None)
        self.hf_cache_dir: str | None = getattr(cfg, "hf_cache_dir", None)
        self.hf_max_samples: int | None = getattr(cfg, "hf_max_samples", None)
        self.hf_teacher_name: str | None = getattr(cfg, "hf_teacher_name", None)
        self.hf_teacher_subset: str | None = getattr(cfg, "hf_teacher_subset", None)
        self.hf_teacher_split: str = str(getattr(cfg, "hf_teacher_split", "train"))
        self.hf_teacher_cache_dir: str | None = getattr(
            cfg, "hf_teacher_cache_dir", None
        )
        self.hf_teacher_max_samples: int | None = getattr(
            cfg, "hf_teacher_max_samples", None
        )

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
            yield self._row_to_item(row, index)

    def __getitem__(self, index: int) -> RerankingDataItem:
        self._assert_setup()
        row: dict[str, Any] = self._get_stream_row(index)
        return self._row_to_item(row, index)

    @property
    def collator(self) -> RerankingCollator:
        if self._collator is None:
            self._collator = RerankingCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                require_teacher_scores=self.require_teacher_scores,
            )
        return self._collator

    @cached_property
    def query_dataset(self) -> Any:
        text_name: str = self.hf_text_name or self.hf_name
        try:
            logger.info(
                "Loading HF queries dataset: name=%s subset=queries split=train streaming=%s",
                text_name,
                False,
            )
            return load_dataset(
                text_name,
                "queries",
                split="train",
                cache_dir=self.hf_cache_dir,
                streaming=False,
            )
        except Exception:  # pylint: disable=broad-except
            logger.info(
                "Loading HF queries dataset fallback: name=%s subset=%s split=%s streaming=%s",
                self.hf_name,
                self.hf_subset,
                self.hf_split,
                False,
            )
            return load_dataset(
                self.hf_name,
                self.hf_subset,
                split=self.hf_split,
                cache_dir=self.hf_cache_dir,
                streaming=False,
            )

    @cached_property
    def corpus_dataset(self) -> Any:
        text_name: str = self.hf_text_name or self.hf_name
        try:
            logger.info(
                "Loading HF corpus dataset: name=%s subset=corpus split=train streaming=%s",
                text_name,
                False,
            )
            return load_dataset(
                text_name,
                "corpus",
                split="train",
                cache_dir=self.hf_cache_dir,
                streaming=False,
            )
        except Exception:  # pylint: disable=broad-except
            logger.info(
                "Loading HF corpus dataset fallback: name=%s subset=%s split=%s streaming=%s",
                self.hf_name,
                self.hf_subset,
                self.hf_split,
                False,
            )
            return load_dataset(
                self.hf_name,
                self.hf_subset,
                split=self.hf_split,
                cache_dir=self.hf_cache_dir,
                streaming=False,
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

    def _get_doc_text(self, doc_id: str, default: str = "") -> str:
        doc_idx: int = self.corpus_id_to_idx[doc_id]
        row: dict[str, Any] = self.corpus_dataset[doc_idx]
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

        query_tokens: Any = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_query_length,
            return_tensors="pt",
        )
        query_input_ids: torch.Tensor = query_tokens["input_ids"].squeeze(0)
        query_attention_mask: torch.Tensor = query_tokens["attention_mask"].squeeze(0)

        if docs:
            doc_tokens: Any = self.tokenizer(
                docs,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_doc_length,
                return_tensors="pt",
            )
            doc_input_ids: torch.Tensor = doc_tokens["input_ids"]
            doc_attention_mask: torch.Tensor = doc_tokens["attention_mask"]
        else:
            doc_input_ids = torch.empty((0, self.cfg.max_doc_length), dtype=torch.long)
            doc_attention_mask = torch.empty(
                (0, self.cfg.max_doc_length), dtype=torch.long
            )

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
    ) -> dict[tuple[str, str], float]:
        if hf_name is None or subset is None:
            return {}
        logger.info(
            "Loading HF teacher scores dataset: name=%s subset=%s split=%s streaming=%s",
            hf_name,
            subset,
            split,
            False,
        )
        teacher_ds: Any = load_dataset(
            hf_name, subset, split=split, cache_dir=hf_cache_dir
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
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer_size, seed=self.seed
            )
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

    def _resolve_length(self, hf_max_samples: int | None) -> int | None:
        if hf_max_samples is not None:
            return int(hf_max_samples)
        try:
            split_info: Any = self.dataset.info.splits.get(self.hf_split)
            if split_info and split_info.num_examples:
                return int(split_info.num_examples)
        except Exception:  # pylint: disable=broad-except
            return None
        return None

    def prepare_data(self) -> None:
        logger.info(
            "Loading HF training dataset (streaming): name=%s subset=%s split=%s streaming=%s",
            self.hf_name,
            self.hf_subset,
            self.hf_split,
            True,
        )
        dataset: Any = load_dataset(
            self.hf_name,
            self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
            streaming=True,
        )
        has_text_triplets: bool = self._has_inline_triplets(dataset.column_names)
        if not has_text_triplets:
            text_name: str = self.hf_text_name or self.hf_name
            logger.info(
                "Loading HF queries dataset: name=%s subset=queries split=train streaming=%s",
                text_name,
                False,
            )
            load_dataset(
                text_name,
                "queries",
                split="train",
                cache_dir=self.hf_cache_dir,
                streaming=False,
            )
            logger.info(
                "Loading HF corpus dataset: name=%s subset=corpus split=train streaming=%s",
                text_name,
                False,
            )
            load_dataset(
                text_name,
                "corpus",
                split="train",
                cache_dir=self.hf_cache_dir,
                streaming=False,
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
        logger.info(
            "Loading HF training dataset (streaming): name=%s subset=%s split=%s streaming=%s",
            self.hf_name,
            self.hf_subset,
            self.hf_split,
            True,
        )
        self.dataset = load_dataset(
            self.hf_name,
            self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
            streaming=True,
        )
        if self.hf_max_samples is not None:
            self.dataset = self.dataset.take(self.hf_max_samples)

        self._length = self._resolve_length(self.hf_max_samples)

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
