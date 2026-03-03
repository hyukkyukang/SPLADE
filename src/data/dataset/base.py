import abc
from dataclasses import dataclass
import logging
import math
import os
import random
from functools import cached_property
from typing import Any, ContextManager, Mapping

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from omegaconf import DictConfig
from torch.utils.data import get_worker_info

from src.data.dataclass import MetaItem
from src.data.dataset.utils import (
    normalize_optional_str,
    optional_cfg_str,
    parse_inline_scores,
    parse_triplet_line,
    require_cfg_str,
    sample_items,
)
from src.data.utils import id_to_idx, resolve_dataset_column
from src.utils.logging import get_logger, loading_status

logger = get_logger("BaseDataset")

QUERY_SUBSET_NAME_KEY: str = "query_subset_name"
QUERY_SPLIT_NAME_KEY: str = "query_split_name"
QUERY_ID_COLUMN_KEY: str = "query_id_column"
QUERY_TEXT_COLUMN_KEY: str = "query_text_column"
CORPUS_SUBSET_NAME_KEY: str = "corpus_subset_name"
CORPUS_SPLIT_NAME_KEY: str = "corpus_split_name"
CORPUS_ID_COLUMN_KEY: str = "corpus_id_column"
CORPUS_TEXT_COLUMN_KEY: str = "corpus_text_column"
CORPUS_TITLE_COLUMN_KEY: str = "corpus_title_column"


@dataclass(frozen=True)
class _MetaRowParseResult:
    qid: str
    pos_ids: list[str]
    neg_ids: list[str]
    pos_scores: list[float] | None
    neg_scores: list[float] | None
    query_text: str | None
    pos_texts: list[str] | None
    neg_texts: list[str] | None


class BaseDataset(abc.ABC):
    """Abstract base class for dataset metadata and text access."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg: DictConfig = cfg
        self.name: str = str(self.cfg.name)

        self.hf_name: str | None = normalize_optional_str(self.cfg.hf_name)
        self.hf_subset: str | None = normalize_optional_str(self.cfg.hf_subset)
        self.hf_split: str = str(self.cfg.split)
        self.hf_cache_dir: str | None = normalize_optional_str(self.cfg.hf_cache_dir)
        self.hf_max_samples: int | None = (
            None if self.cfg.hf_max_samples is None else int(self.cfg.hf_max_samples)
        )
        self.hf_skip_samples: int = int(self.cfg.hf_skip_samples)
        self.hf_data_files: Mapping[str, Any] | None = self.cfg.hf_data_files
        self.query_corpus_hf_name: str | None = normalize_optional_str(
            self.cfg.query_corpus_hf_name
        )
        self.query_corpus_hf_cache_dir: str | None = normalize_optional_str(
            self.cfg.query_corpus_hf_cache_dir
        )
        self.query_corpus_hf_data_files: Mapping[str, Any] | None = (
            self.cfg.query_corpus_hf_data_files
        )
        self.query_lookup_hf_name: str | None = normalize_optional_str(
            self.cfg.get("query_lookup_hf_name")
        )
        self.query_lookup_hf_subset: str | None = normalize_optional_str(
            self.cfg.get("query_lookup_hf_subset")
        )
        self.query_lookup_hf_split: str = str(
            self.cfg.get("query_lookup_hf_split", "train")
        )
        self.query_lookup_hf_cache_dir: str | None = normalize_optional_str(
            self.cfg.get("query_lookup_hf_cache_dir")
        )
        self.query_lookup_hf_data_files: Mapping[str, Any] | None = self.cfg.get(
            "query_lookup_hf_data_files"
        )
        self.query_lookup_id_column: str = str(
            self.cfg.get("query_lookup_id_column", "query_id")
        )
        self.query_lookup_text_column: str = str(
            self.cfg.get("query_lookup_text_column", "text")
        )
        self.use_hf: bool = bool(
            self.hf_name is not None or self.query_corpus_hf_name is not None
        )

        self.local_triplets_dir: str | None = normalize_optional_str(
            self.cfg.local_triplets_dir
        )

        negative_sampling_cfg: DictConfig = self.cfg.negative_sampling
        self.negative_sampling_strategy: str = str(
            negative_sampling_cfg.strategy
        ).lower()
        self.negative_sampling_top_k: int | None = (
            None
            if negative_sampling_cfg.top_k is None
            else int(negative_sampling_cfg.top_k)
        )
        self.negative_sampling_random_k: int | None = (
            None
            if negative_sampling_cfg.random_k is None
            else int(negative_sampling_cfg.random_k)
        )
        self.negative_sampling_random_pool: int | None = (
            None
            if negative_sampling_cfg.random_pool is None
            else int(negative_sampling_cfg.random_pool)
        )
        if self.negative_sampling_strategy not in {
            "random",
            "topk",
            "topk_plus_random",
        }:
            raise ValueError(
                "negative_sampling.strategy must be one of: random, topk, "
                f"topk_plus_random. Got: {self.negative_sampling_strategy}"
            )

        query_subset_name: str = require_cfg_str(self.cfg, "query_subset_name")
        query_split_name: str = require_cfg_str(self.cfg, "query_split_name")
        query_id_column: str = require_cfg_str(self.cfg, "query_id_column")
        query_text_column: str = require_cfg_str(self.cfg, "query_text_column")
        corpus_subset_name: str = require_cfg_str(self.cfg, "corpus_subset_name")
        corpus_split_name: str = require_cfg_str(self.cfg, "corpus_split_name")
        corpus_id_column: str = require_cfg_str(self.cfg, "corpus_id_column")
        corpus_text_column: str = require_cfg_str(self.cfg, "corpus_text_column")
        corpus_title_column: str | None = optional_cfg_str(
            self.cfg, "corpus_title_column"
        )

        self.query_column_names: dict[str, str] = {
            QUERY_SUBSET_NAME_KEY: query_subset_name,
            QUERY_SPLIT_NAME_KEY: query_split_name,
            QUERY_ID_COLUMN_KEY: query_id_column,
            QUERY_TEXT_COLUMN_KEY: query_text_column,
        }
        self.corpus_column_names: dict[str, str] = {
            CORPUS_SUBSET_NAME_KEY: corpus_subset_name,
            CORPUS_SPLIT_NAME_KEY: corpus_split_name,
            CORPUS_ID_COLUMN_KEY: corpus_id_column,
            CORPUS_TEXT_COLUMN_KEY: corpus_text_column,
        }
        if corpus_title_column is not None:
            self.corpus_column_names[CORPUS_TITLE_COLUMN_KEY] = corpus_title_column

    # --- Property methods ---
    @property
    def rank_id(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def worker_id(self) -> int:
        worker_info: Any | None = get_worker_info()
        return int(worker_info.id) if worker_info is not None else 0

    @property
    def all_qids(self) -> set[str]:
        """Get all query IDs in the dataset."""
        query_ids: list[Any] = list(self.query_dataset[self.query_id_column_name])
        # Normalize IDs to strings for consistent downstream lookups.
        return {str(qid) for qid in query_ids}

    @property
    def all_dids(self) -> set[str]:
        """Get all document IDs in the corpus."""
        doc_ids: list[Any] = list(self.corpus_dataset[self.corpus_id_column_name])
        # Normalize IDs to strings for consistent downstream lookups.
        return {str(doc_id) for doc_id in doc_ids}

    @property
    def huggingface_name(self) -> str:
        """Return the Hugging Face dataset name from config."""
        if self.query_corpus_hf_name is not None:
            return self.query_corpus_hf_name
        hf_name_value: Any | None = self.cfg.get("huggingface_name")
        if hf_name_value is None:
            hf_name_value = self.hf_name
        if hf_name_value is None:
            raise ValueError(
                "Missing dataset name in config (huggingface_name/hf_name)"
            )
        return str(hf_name_value)

    @cached_property
    def query_dataset(self) -> Dataset:
        """Get the query dataset containing all queries."""
        self._ensure_hf_enabled()
        with self._loading(
            logger, f"query dataset for {self.huggingface_name}", only_once=True
        ):
            subset_name: str = self.query_column_names[QUERY_SUBSET_NAME_KEY]
            split_name: str = self.query_column_names[QUERY_SPLIT_NAME_KEY]
            text_cache_dir: str | None = (
                self.query_corpus_hf_cache_dir
                if self.query_corpus_hf_cache_dir is not None
                else self.hf_cache_dir
            )
            dataset: Dataset = self._load_hf_dataset(
                self.huggingface_name,
                subset_name,
                split_name,
                text_cache_dir,
                self.query_corpus_hf_data_files,
            )
        return dataset

    @cached_property
    def corpus_dataset(self) -> Dataset:
        """Get the corpus dataset containing all documents/passages."""
        self._ensure_hf_enabled()
        with self._loading(
            logger, f"corpus dataset for {self.huggingface_name}", only_once=True
        ):
            subset_name: str = self.corpus_column_names[CORPUS_SUBSET_NAME_KEY]
            split_name: str = self.corpus_column_names[CORPUS_SPLIT_NAME_KEY]
            text_cache_dir: str | None = (
                self.query_corpus_hf_cache_dir
                if self.query_corpus_hf_cache_dir is not None
                else self.hf_cache_dir
            )
            dataset: Dataset = self._load_hf_dataset(
                self.huggingface_name,
                subset_name,
                split_name,
                text_cache_dir,
                self.query_corpus_hf_data_files,
            )
        return dataset

    @cached_property
    def meta_dataset(self) -> Dataset:
        """Return the dataset providing training metadata rows."""
        return self._resolve_meta_dataset()

    @cached_property
    def query_lookup_dataset(self) -> Dataset | None:
        """Optional fallback query-text dataset used for missing query ids."""
        if self.query_lookup_hf_name is None:
            return None
        with self._loading(
            logger,
            f"query lookup dataset for {self.query_lookup_hf_name}",
            only_once=True,
        ):
            return self._load_hf_dataset(
                self.query_lookup_hf_name,
                self.query_lookup_hf_subset,
                self.query_lookup_hf_split,
                self.query_lookup_hf_cache_dir,
                self.query_lookup_hf_data_files,
            )

    @property
    def query_id_column_name(self) -> str:
        """Return the column name for query IDs."""
        return self.query_column_names[QUERY_ID_COLUMN_KEY]

    @property
    def corpus_id_column_name(self) -> str:
        """Return the column name for document IDs."""
        return self.corpus_column_names[CORPUS_ID_COLUMN_KEY]

    @property
    def corpus_title_column_name(self) -> str | None:
        """Return the column name for document titles, if available."""
        return self.corpus_column_names.get(CORPUS_TITLE_COLUMN_KEY)

    @property
    def query_text_column_name(self) -> str:
        """Get the column name for query text."""
        return self.query_column_names[QUERY_TEXT_COLUMN_KEY]

    @property
    def corpus_text_column_name(self) -> str:
        """Get the column name for corpus text."""
        return self.corpus_column_names[CORPUS_TEXT_COLUMN_KEY]

    @cached_property
    def query_dataset_id_to_idx(self) -> dict[str, int]:
        """Create a mapping from query IDs to their indices in the query dataset."""
        enable_tqdm: bool = self.rank_id == 0 and self.worker_id == 0
        # Use resolve_dataset_column() for fast PyArrow access that respects filtering.
        return id_to_idx(
            resolve_dataset_column(self.query_dataset, self.query_id_column_name),
            "Mapping query ids to indices",
            enable_tqdm,
        )

    @cached_property
    def corpus_dataset_id_to_idx(self) -> dict[str, int]:
        """Create a mapping from document IDs to their indices in the corpus dataset."""
        enable_tqdm: bool = self.rank_id == 0 and self.worker_id == 0
        # Use resolve_dataset_column() for fast PyArrow access that respects filtering.
        return id_to_idx(
            resolve_dataset_column(self.corpus_dataset, self.corpus_id_column_name),
            "Mapping corpus ids to indices",
            enable_tqdm,
        )

    @cached_property
    def query_lookup_id_to_idx(self) -> dict[str, int]:
        lookup_dataset: Dataset | None = self.query_lookup_dataset
        if lookup_dataset is None:
            return {}
        enable_tqdm: bool = self.rank_id == 0 and self.worker_id == 0
        return id_to_idx(
            resolve_dataset_column(lookup_dataset, self.query_lookup_id_column),
            "Mapping query lookup ids to indices",
            enable_tqdm,
        )

    # --- Protected methods ---
    @abc.abstractmethod
    def _resolve_meta_dataset(self) -> Dataset:
        """Return the metadata dataset for this dataset type."""
        raise NotImplementedError

    def _ensure_hf_enabled(self) -> None:
        if self.hf_name is None and self.query_corpus_hf_name is None:
            raise RuntimeError("HuggingFace datasets are disabled (hf_name is null).")

    def _load_hf_dataset(
        self,
        hf_name: str,
        hf_subset: str | None,
        split: str,
        cache_dir: str | None,
        data_files: Mapping[str, Any] | None,
    ) -> Dataset:
        if data_files:
            return load_dataset(
                hf_name,
                name=hf_subset,
                split=split,
                cache_dir=cache_dir,
                data_files=dict(data_files),
            )
        return load_dataset(
            hf_name,
            name=hf_subset,
            split=split,
            cache_dir=cache_dir,
        )

    def _apply_hf_sample_window(self, dataset: Dataset) -> Dataset:
        skip_samples: int = int(self.hf_skip_samples)
        max_samples: int | None = self.hf_max_samples
        if skip_samples <= 0 and max_samples is None:
            return dataset
        dataset_length: int = int(len(dataset))
        start_index: int = min(skip_samples, dataset_length)
        end_index: int = dataset_length
        if max_samples is not None:
            end_index = min(start_index + int(max_samples), dataset_length)
        indices: range = range(start_index, end_index)
        return dataset.select(indices)

    def _load_local_triplets(self) -> Dataset:
        if self.local_triplets_dir is None:
            raise ValueError("local_triplets_dir must be set for local triplets.")
        raw_path: str = os.path.join(self.local_triplets_dir, "raw.tsv")
        if not os.path.isfile(raw_path):
            raise FileNotFoundError(f"Missing local triplets file: {raw_path}")
        rows: list[dict[str, Any]] = []
        with open(raw_path, "r", encoding="utf-8") as reader:
            for row_idx, line in enumerate(reader):
                parsed: tuple[str, str, str, str] | None = parse_triplet_line(
                    line, row_idx
                )
                if parsed is None:
                    continue
                qid: str
                query_text: str
                pos_text: str
                neg_text: str
                qid, query_text, pos_text, neg_text = parsed
                rows.append(
                    {
                        "query_id": qid,
                        "query": query_text,
                        "positive": pos_text,
                        "negative": neg_text,
                    }
                )
        return Dataset.from_list(rows)

    def _get_query_text_from_id(self, qid: str) -> str:
        return self.query_text(self.query_dataset_id_to_idx[qid])

    def _get_corpus_text_from_id(self, doc_id: str) -> str:
        return self.corpus_text(self.corpus_dataset_id_to_idx[doc_id])

    def _resolve_row_score_values(self, row: dict[str, Any]) -> Any | None:
        configured_score_column: str | None = normalize_optional_str(
            self.cfg.get("score_scores_column")
        )
        candidate_keys: list[str] = []
        if configured_score_column is not None:
            candidate_keys.append(configured_score_column)
        candidate_keys.extend(["teacher_scores", "scores", "score"])
        seen: set[str] = set()
        for key in candidate_keys:
            if not key or key in seen:
                continue
            seen.add(key)
            if key in row:
                return row.get(key)
        return None

    @staticmethod
    def _parse_inline_triplet_row(
        row: dict[str, Any],
        *,
        query_field: str,
        index: int,
    ) -> _MetaRowParseResult:
        query_text: str = str(row[query_field])
        pos_text: str = str(row["positive"])
        neg_text: str = str(row["negative"])
        qid: str = str(row.get("query_id") or row.get("qid") or index)
        return _MetaRowParseResult(
            qid=qid,
            pos_ids=[""],
            neg_ids=[""],
            pos_scores=None,
            neg_scores=None,
            query_text=query_text,
            pos_texts=[pos_text],
            neg_texts=[neg_text],
        )

    @staticmethod
    def _parse_query_positive_id_row(row: dict[str, Any]) -> _MetaRowParseResult:
        qid: str = str(row["query_id"])
        pos_id_value: Any | None = row.get("positive_id")
        neg_id_value: Any | None = row.get("negative_id")
        return _MetaRowParseResult(
            qid=qid,
            pos_ids=["" if pos_id_value is None else str(pos_id_value)],
            neg_ids=["" if neg_id_value is None else str(neg_id_value)],
            pos_scores=None,
            neg_scores=None,
            query_text=None,
            pos_texts=None,
            neg_texts=None,
        )

    @staticmethod
    def _parse_pos_neg_doc_ids_row(
        row: dict[str, Any],
        *,
        index: int,
        num_positives: int,
        num_negatives: int,
        rng: random.Random,
    ) -> _MetaRowParseResult:
        qid: str = str(
            row.get("query_id") or row.get("qid") or row.get("_id") or index
        )
        pos_ids: list[str] = [
            str(doc_id) for doc_id in row.get("pos_doc_ids") or [] if doc_id
        ]
        neg_ids: list[str] = [
            str(doc_id) for doc_id in row.get("neg_doc_ids") or [] if doc_id
        ]
        return _MetaRowParseResult(
            qid=qid,
            pos_ids=sample_items(pos_ids, num_positives, rng),
            neg_ids=sample_items(neg_ids, num_negatives, rng),
            pos_scores=None,
            neg_scores=None,
            query_text=None,
            pos_texts=None,
            neg_texts=None,
        )

    def _parse_labeled_doc_ids_row(
        self,
        row: dict[str, Any],
        *,
        num_positives: int,
        num_negatives: int,
        rng: random.Random,
        score_values: Any | None,
    ) -> _MetaRowParseResult:
        qid: str = str(row["query_id"])
        row_doc_ids: list[str] = [str(doc_id) for doc_id in row["doc_ids"]]
        labels: list[float] = [float(value) for value in row["labels"]]
        score_list: list[float] | None = None
        if isinstance(score_values, (list, tuple)) and len(score_values) == len(
            row_doc_ids
        ):
            score_list = [float(score) for score in score_values]

        pos_pairs: list[tuple[str, float | None]] = []
        neg_pairs: list[tuple[str, float | None]] = []
        for idx, (doc_id, label) in enumerate(zip(row_doc_ids, labels)):
            score: float | None = None
            if score_list is not None:
                score = score_list[idx]
            if label > 0:
                pos_pairs.append((doc_id, score))
            else:
                neg_pairs.append((doc_id, score))

        pos_ids: list[str] = sample_items(
            [doc_id for doc_id, _ in pos_pairs], num_positives, rng
        )
        neg_ids: list[str] = self._select_negative_ids(
            neg_pairs, num_negatives=num_negatives, rng=rng
        )
        pos_scores: list[float] | None = None
        neg_scores: list[float] | None = None
        if isinstance(score_values, (list, tuple)) and len(score_values) == len(
            row_doc_ids
        ):
            score_map: dict[str, float] = {
                doc_id: float(score) for doc_id, score in zip(row_doc_ids, score_values)
            }
            pos_scores = [score_map.get(doc_id, float("nan")) for doc_id in pos_ids]
            neg_scores = [score_map.get(doc_id, float("nan")) for doc_id in neg_ids]
        return _MetaRowParseResult(
            qid=qid,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            query_text=None,
            pos_texts=None,
            neg_texts=None,
        )

    def _row_to_meta_item(
        self,
        row: dict[str, Any],
        index: int,
        *,
        num_positives: int,
        num_negatives: int,
        rng: random.Random,
    ) -> MetaItem:
        score_values: Any | None = self._resolve_row_score_values(row)
        parsed: _MetaRowParseResult
        if "query" in row and "positive" in row and "negative" in row:
            parsed = self._parse_inline_triplet_row(
                row,
                query_field="query",
                index=index,
            )
        elif "anchor" in row and "positive" in row and "negative" in row:
            parsed = self._parse_inline_triplet_row(
                row,
                query_field="anchor",
                index=index,
            )
        elif "query_id" in row and "positive_id" in row:
            parsed = self._parse_query_positive_id_row(row)
        elif "pos_doc_ids" in row or "neg_doc_ids" in row:
            parsed = self._parse_pos_neg_doc_ids_row(
                row,
                index=index,
                num_positives=num_positives,
                num_negatives=num_negatives,
                rng=rng,
            )
        elif "query_id" in row and "doc_ids" in row and "labels" in row:
            parsed = self._parse_labeled_doc_ids_row(
                row,
                num_positives=num_positives,
                num_negatives=num_negatives,
                rng=rng,
                score_values=score_values,
            )
        else:
            raise ValueError(f"Unsupported dataset row format: {row.keys()}")

        pos_scores: list[float] | None = parsed.pos_scores
        neg_scores: list[float] | None = parsed.neg_scores
        if pos_scores is None or neg_scores is None:
            doc_ids: list[str] = parsed.pos_ids + parsed.neg_ids
            inline_scores: list[float] | None = parse_inline_scores(
                score_values, doc_ids
            )
            if inline_scores is not None and len(inline_scores) == len(doc_ids):
                pos_scores = inline_scores[: len(parsed.pos_ids)]
                neg_scores = inline_scores[len(parsed.pos_ids) :]

        return MetaItem(
            qid=str(parsed.qid),
            pos_ids=parsed.pos_ids,
            neg_ids=parsed.neg_ids,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            query_text=parsed.query_text,
            pos_texts=parsed.pos_texts,
            neg_texts=parsed.neg_texts,
        )

    def _select_negative_ids(
        self,
        neg_pairs: list[tuple[str, float | None]],
        *,
        num_negatives: int,
        rng: random.Random,
    ) -> list[str]:
        neg_ids: list[str] = [doc_id for doc_id, _ in neg_pairs]
        if self.negative_sampling_strategy == "random":
            return sample_items(neg_ids, num_negatives, rng)

        scores: list[float] = []
        for _, score in neg_pairs:
            if score is None:
                raise ValueError(
                    "negative_sampling.strategy requires scores, but scores are missing."
                )
            if not math.isfinite(float(score)):
                raise ValueError("negative_sampling.strategy requires finite scores.")
            scores.append(float(score))

        if self.negative_sampling_strategy == "topk":
            sorted_pairs = sorted(
                zip(neg_ids, scores), key=lambda pair: pair[1], reverse=True
            )
            return [doc_id for doc_id, _ in sorted_pairs[:num_negatives]]

        if self.negative_sampling_strategy == "topk_plus_random":
            top_k = self.negative_sampling_top_k
            random_k = self.negative_sampling_random_k
            if top_k is None or random_k is None:
                raise ValueError(
                    "negative_sampling.top_k and negative_sampling.random_k must be set "
                    "for topk_plus_random."
                )
            if num_negatives != top_k + random_k:
                raise ValueError(
                    "num_negatives must equal top_k + random_k for topk_plus_random."
                )
            sorted_pairs = sorted(
                zip(neg_ids, scores), key=lambda pair: pair[1], reverse=True
            )
            pool_size = (
                len(sorted_pairs)
                if self.negative_sampling_random_pool is None
                else min(self.negative_sampling_random_pool, len(sorted_pairs))
            )
            pool_pairs = sorted_pairs[:pool_size]
            top_pairs = pool_pairs[:top_k]
            remaining_pairs = pool_pairs[top_k:]
            if random_k > len(remaining_pairs):
                raise ValueError(
                    "Not enough negatives to sample random_k from the pool."
                )
            random_pairs = rng.sample(remaining_pairs, random_k)
            selected_pairs = list(top_pairs) + list(random_pairs)
            return [doc_id for doc_id, _ in selected_pairs]

        raise ValueError(
            f"Unsupported negative_sampling.strategy: {self.negative_sampling_strategy}"
        )

    def _loading(
        self,
        logger: logging.Logger,
        subject: str,
        *,
        loading_msg: str | None = None,
        done_msg: str | None = None,
        only_once: bool = False,
    ) -> ContextManager[None]:
        return loading_status(
            logger=logger,
            subject=subject,
            loading_msg=loading_msg,
            done_msg=done_msg,
            only_once=only_once,
            rank_id=self.rank_id,
            worker_id=self.worker_id,
        )

    # --- Public methods ---
    def _lookup_query_text_by_id(self, qid: str) -> str:
        lookup_dataset: Dataset | None = self.query_lookup_dataset
        if lookup_dataset is None:
            return ""
        qid_text: str = str(qid)
        query_idx: int = int(self.query_lookup_id_to_idx.get(qid_text, -1))
        if query_idx < 0:
            return ""
        row: dict[str, Any] = dict(lookup_dataset[int(query_idx)])
        raw_text: Any | None = row.get(self.query_lookup_text_column)
        return "" if raw_text is None else str(raw_text).strip()

    def resolve_query_text(self, meta_item: MetaItem) -> str:
        if meta_item.query_text is not None:
            query_text: str = str(meta_item.query_text)
            if query_text.strip():
                return query_text
        try:
            query_text = self._get_query_text_from_id(meta_item.qid)
            if query_text.strip():
                return query_text
        except KeyError:
            pass
        return self._lookup_query_text_by_id(meta_item.qid)

    def resolve_doc_texts(
        self, doc_ids: list[str], inline_texts: list[str] | None
    ) -> list[str]:
        if inline_texts is not None:
            return [str(text) for text in inline_texts]
        texts: list[str] = []
        for doc_id in doc_ids:
            if not doc_id:
                texts.append("")
                continue
            try:
                texts.append(self._get_corpus_text_from_id(doc_id))
            except KeyError:
                texts.append("")
        return texts

    def build_meta_item(
        self,
        row: dict[str, Any],
        index: int,
        *,
        num_positives: int,
        num_negatives: int,
        rng: random.Random,
        load_teacher_scores: bool,
        require_teacher_scores: bool,
    ) -> MetaItem:
        _ = load_teacher_scores
        meta_item: MetaItem = self._row_to_meta_item(
            row,
            index,
            num_positives=num_positives,
            num_negatives=num_negatives,
            rng=rng,
        )
        pos_scores: list[float] | None = meta_item.pos_scores
        neg_scores: list[float] | None = meta_item.neg_scores
        if require_teacher_scores:
            if pos_scores is None or neg_scores is None:
                raise ValueError(f"Missing teacher scores for query {meta_item.qid}")
            if any(score != score for score in pos_scores + neg_scores):
                raise ValueError(f"Missing teacher scores for query {meta_item.qid}")

        return MetaItem(
            qid=meta_item.qid,
            pos_ids=meta_item.pos_ids,
            neg_ids=meta_item.neg_ids,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            query_text=meta_item.query_text,
            pos_texts=meta_item.pos_texts,
            neg_texts=meta_item.neg_texts,
        )

    def prepare_meta_dataset(self) -> None:
        _ = self.meta_dataset

    def prepare_text_datasets(self) -> None:
        _ = self.query_dataset
        _ = self.corpus_dataset

    def lookup_query_texts(self, qids: list[str]) -> dict[str, str]:
        lookup_dataset: Dataset | None = self.query_lookup_dataset
        if lookup_dataset is None:
            return {}
        id_to_idx_map: dict[str, int] = self.query_lookup_id_to_idx
        query_texts: dict[str, str] = {}
        qid: str
        for qid in qids:
            qid_text: str = str(qid)
            query_idx: int = int(id_to_idx_map.get(qid_text, -1))
            if query_idx < 0:
                continue
            row: dict[str, Any] = dict(lookup_dataset[int(query_idx)])
            raw_text: Any | None = row.get(self.query_lookup_text_column)
            text: str = "" if raw_text is None else str(raw_text).strip()
            if text:
                query_texts[qid_text] = text
        return query_texts

    def query_text(self, idx: int) -> str:
        """Get the text of a query."""
        raw_value: Any = self.query_dataset[idx][self.query_text_column_name]
        return "" if raw_value is None else str(raw_value)

    def corpus_text(self, idx: int) -> str:
        """Get the text of a document in the corpus, including titles when present."""
        title_column_name: str | None = self.corpus_title_column_name
        title_value: Any | None = (
            self.corpus_dataset[idx][title_column_name]
            if title_column_name is not None
            else None
        )
        text_value: Any = self.corpus_dataset[idx][self.corpus_text_column_name]
        title: str = "" if title_value is None else str(title_value)
        text: str = "" if text_value is None else str(text_value)
        if title:
            return f"{title} {text}".strip()
        return text.strip()

    def download_data(self) -> None:
        """Download the dataset from HuggingFace Hub."""
        snapshot_download(repo_id=self.huggingface_name, repo_type="dataset")
