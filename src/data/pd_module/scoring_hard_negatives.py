import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

from src.data.pd_module.scoring import (
    ScoringItem,
    _build_doc_text_lookup_for_corpus,
    _build_id_to_idx_map,
    _column_text_value,
    _apply_hf_offline_mode,
    _load_dataset_from_config,
    _normalize_data_files,
    _resolve_download_config,
    _resolve_local_files_only,
)
from src.data.utils import resolve_dataset_column
from src.utils.logging import log_if_rank_zero
from src.utils.script_setup import normalize_optional_str

logger: logging.Logger = logging.getLogger("HardNegativesScoringPDModule")


@dataclass(frozen=True)
class HardNegativesSettings:
    """Settings for parsing hard-negatives rows."""

    neg_keys: list[str]
    max_negatives: int | None
    max_positives: int | None
    require_negatives: bool


def _parse_hard_negatives_settings(cfg: DictConfig) -> HardNegativesSettings:
    neg_keys_value: list[str] = [str(key) for key in cfg.neg_keys]
    max_negatives: int | None = (
        None if cfg.max_negatives is None else int(cfg.max_negatives)
    )
    max_positives: int | None = (
        None if cfg.max_positives is None else int(cfg.max_positives)
    )
    require_negatives: bool = bool(cfg.require_negatives)
    return HardNegativesSettings(
        neg_keys=neg_keys_value,
        max_negatives=max_negatives,
        max_positives=max_positives,
        require_negatives=require_negatives,
    )


def _resolve_configured_column(
    configured: str | None,
    column_names: Iterable[str],
    candidates: Sequence[str],
    *,
    field_name: str,
) -> str:
    normalized: str | None = normalize_optional_str(configured)
    if normalized is not None:
        if normalized in column_names:
            return normalized
        log_if_rank_zero(
            logger,
            f"Configured {field_name}={normalized} not found; falling back.",
            level="warning",
        )
    for name in candidates:
        if name in column_names:
            return name
    raise ValueError(f"Unable to resolve column from {list(column_names)}")


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _coerce_id_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values if value is not None]
    return [str(values)]


def _dedupe_with_sources(
    doc_ids: Iterable[str], sources: Iterable[str]
) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    output_ids: list[str] = []
    output_sources: list[str] = []
    for doc_id, source in zip(doc_ids, sources):
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        output_ids.append(doc_id)
        output_sources.append(source)
    return output_ids, output_sources


def _merge_negatives_with_sources(
    value: Any, neg_keys: list[str]
) -> tuple[list[str], list[str]]:
    if value is None:
        return [], []
    if isinstance(value, Mapping):
        if neg_keys:
            keys: list[str] = [key for key in neg_keys if key in value]
        else:
            keys = list(value.keys())
        merged_ids: list[str] = []
        merged_sources: list[str] = []
        for key in keys:
            for doc_id in _coerce_id_list(value.get(key)):
                merged_ids.append(doc_id)
                merged_sources.append(str(key))
        return _dedupe_with_sources(merged_ids, merged_sources)
    merged_ids = _coerce_id_list(value)
    merged_sources = ["neg"] * len(merged_ids)
    return _dedupe_with_sources(merged_ids, merged_sources)


class HardNegativesScoringPDModule(TorchDataset):
    """PyTorch dataset for scoring HF hard negatives."""

    def __init__(
        self,
        *,
        score_dataset_cfg: DictConfig,
        scoring_cfg: DictConfig,
        hard_negatives_cfg: DictConfig,
    ) -> None:
        self._score_dataset_cfg: DictConfig = score_dataset_cfg
        self._scoring_cfg: DictConfig = scoring_cfg
        self._hard_negatives_cfg: DictConfig = hard_negatives_cfg

        self._max_rows: int | None = (
            None if scoring_cfg.max_rows is None else int(scoring_cfg.max_rows)
        )
        self._doc_text_cache_size: int = int(scoring_cfg.doc_text_cache_size)
        self._settings: HardNegativesSettings = _parse_hard_negatives_settings(
            hard_negatives_cfg
        )

        self._score_dataset: Dataset | None = None
        self._query_dataset: Dataset | None = None
        self._corpus_dataset: Dataset | None = None
        self._query_id_to_idx: dict[str, int] | None = None
        self._corpus_id_to_idx: dict[str, int] | None = None
        self._query_text_column: pa.Array | pa.ChunkedArray | None = None
        self._corpus_text_column: pa.Array | pa.ChunkedArray | None = None
        self._doc_text_lookup: Any | None = None

    def prepare_data(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if int(torch.distributed.get_rank()) != 0:
                return
        local_files_only = _resolve_local_files_only(
            self._scoring_cfg, is_primary=True
        )
        _apply_hf_offline_mode(local_files_only)
        _ = self._load_score_dataset(local_files_only=local_files_only)
        _ = self._load_query_dataset(local_files_only=local_files_only)
        _ = self._load_corpus_dataset(local_files_only=local_files_only)

    def setup(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_primary = int(torch.distributed.get_rank()) == 0
            local_files_only = _resolve_local_files_only(
                self._scoring_cfg, is_primary=is_primary
            )
            if is_primary:
                self._load_all_datasets(local_files_only=local_files_only)
            torch.distributed.barrier()
            if not is_primary:
                local_files_only = _resolve_local_files_only(
                    self._scoring_cfg, is_primary=False
                )
                self._load_all_datasets(local_files_only=local_files_only)
        else:
            self._load_all_datasets(
                local_files_only=_resolve_local_files_only(
                    self._scoring_cfg, is_primary=True
                )
            )

        query_id_column: str = _resolve_configured_column(
            str(self._score_dataset_cfg.query_id_column),
            self._query_dataset.column_names,
            ("query_id", "qid", "_id", "id"),
            field_name="query_id_column",
        )
        query_text_column_name: str = _resolve_configured_column(
            str(self._score_dataset_cfg.query_text_column),
            self._query_dataset.column_names,
            ("text", "query"),
            field_name="query_text_column",
        )
        corpus_id_column: str = _resolve_configured_column(
            str(self._score_dataset_cfg.corpus_id_column),
            self._corpus_dataset.column_names,
            ("doc_id", "corpus_id", "passage_id", "_id", "id"),
            field_name="corpus_id_column",
        )
        corpus_text_column_name: str = _resolve_configured_column(
            str(self._score_dataset_cfg.corpus_text_column),
            self._corpus_dataset.column_names,
            ("text", "passage", "contents"),
            field_name="corpus_text_column",
        )

        self._query_id_to_idx = _build_id_to_idx_map(
            self._query_dataset, query_id_column
        )
        self._query_text_column = resolve_dataset_column(
            self._query_dataset, query_text_column_name
        )
        self._doc_text_lookup = _build_doc_text_lookup_for_corpus(
            corpus_dataset=self._corpus_dataset,
            corpus_id_column_name=corpus_id_column,
            corpus_text_column_name=corpus_text_column_name,
            scoring_cfg=self._scoring_cfg,
            score_dataset_cfg=self._score_dataset_cfg,
        )
        self._corpus_id_to_idx = None
        self._corpus_text_column = None

    def __len__(self) -> int:
        self._ensure_ready()
        if self._score_dataset is None:
            return 0
        row_count: int = int(len(self._score_dataset))
        if self._max_rows is None:
            return row_count
        return min(row_count, self._max_rows)

    def __getitem__(self, idx: int) -> ScoringItem | None:
        self._ensure_ready()
        if self._score_dataset is None:
            return None
        row: dict[str, Any] = dict(self._score_dataset[int(idx)])
        qid: str = str(
            row.get("qid") or row.get("query_id") or row.get("_id") or ""
        )
        if not qid:
            return None

        inline_query_text: str | None = normalize_optional_str(row.get("query_text"))
        if inline_query_text is not None:
            query_text = inline_query_text
        else:
            query_idx: int = int(self._query_id_to_idx.get(qid, -1))
            if query_idx < 0:
                return None
            query_text = _column_text_value(self._query_text_column, query_idx)
            if not query_text:
                return None

        pos_ids: list[str] = _dedupe_preserve_order(_coerce_id_list(row.get("pos")))
        neg_ids, neg_sources = _merge_negatives_with_sources(
            row.get("neg"), self._settings.neg_keys
        )

        if self._settings.max_positives is not None:
            pos_ids = pos_ids[: self._settings.max_positives]
        pos_id_set: set[str] = set(pos_ids)
        if pos_id_set:
            filtered_ids: list[str] = []
            filtered_sources: list[str] = []
            for doc_id, source in zip(neg_ids, neg_sources):
                if doc_id in pos_id_set:
                    continue
                filtered_ids.append(doc_id)
                filtered_sources.append(source)
            neg_ids = filtered_ids
            neg_sources = filtered_sources
        if self._settings.max_negatives is not None:
            neg_ids = neg_ids[: self._settings.max_negatives]
            neg_sources = neg_sources[: self._settings.max_negatives]
        if self._settings.require_negatives and not neg_ids:
            return None

        doc_ids: list[str] = pos_ids + neg_ids
        if not doc_ids:
            return None
        labels: list[float] = [1.0] * len(pos_ids) + [0.0] * len(neg_ids)
        doc_sources: list[str] = ["pos"] * len(pos_ids) + neg_sources
        doc_texts: list[str] = self._doc_text_lookup.lookup_many(doc_ids)
        return ScoringItem(
            row=row,
            qid=qid,
            doc_ids=doc_ids,
            labels=labels,
            doc_sources=doc_sources,
            query_text=query_text,
            doc_texts=doc_texts,
        )

    def _ensure_ready(self) -> None:
        if self._score_dataset is None or self._doc_text_lookup is None:
            self.setup()

    def _load_all_datasets(self, *, local_files_only: bool) -> None:
        _apply_hf_offline_mode(local_files_only)
        if self._score_dataset is None:
            self._score_dataset = self._load_score_dataset(
                local_files_only=local_files_only
            )
        if self._query_dataset is None:
            self._query_dataset = self._load_query_dataset(
                local_files_only=local_files_only
            )
        if self._corpus_dataset is None:
            self._corpus_dataset = self._load_corpus_dataset(
                local_files_only=local_files_only
            )

    def _load_score_dataset(self, *, local_files_only: bool) -> Dataset:
        download_config = _resolve_download_config(
            self._scoring_cfg, local_files_only=local_files_only
        )
        return _load_dataset_from_config(
            self._score_dataset_cfg, download_config=download_config
        )

    def _load_query_dataset(self, *, local_files_only: bool) -> Dataset:
        text_name_value: str | None = normalize_optional_str(
            self._score_dataset_cfg.query_corpus_hf_name
        )
        text_name: str = (
            str(text_name_value)
            if text_name_value is not None
            else str(self._score_dataset_cfg.hf_name)
        )
        text_cache_dir_value: str | None = normalize_optional_str(
            self._score_dataset_cfg.query_corpus_hf_cache_dir
        )
        text_cache_dir: str | None = (
            text_cache_dir_value
            if text_cache_dir_value is not None
            else self._score_dataset_cfg.hf_cache_dir
        )
        query_subset_name: str = (
            normalize_optional_str(self._score_dataset_cfg.query_subset_name)
            or "queries"
        )
        query_split_name: str = (
            normalize_optional_str(self._score_dataset_cfg.query_split_name) or "train"
        )
        query_corpus_data_files: Any | None = _normalize_data_files(
            self._score_dataset_cfg.query_corpus_hf_data_files
        )
        download_config = _resolve_download_config(
            self._scoring_cfg, local_files_only=local_files_only
        )
        return load_dataset(
            text_name,
            name=query_subset_name,
            split=query_split_name,
            cache_dir=text_cache_dir,
            data_files=query_corpus_data_files,
            download_config=download_config,
        )

    def _load_corpus_dataset(self, *, local_files_only: bool) -> Dataset:
        text_name_value: str | None = normalize_optional_str(
            self._score_dataset_cfg.query_corpus_hf_name
        )
        text_name: str = (
            str(text_name_value)
            if text_name_value is not None
            else str(self._score_dataset_cfg.hf_name)
        )
        text_cache_dir_value: str | None = normalize_optional_str(
            self._score_dataset_cfg.query_corpus_hf_cache_dir
        )
        text_cache_dir: str | None = (
            text_cache_dir_value
            if text_cache_dir_value is not None
            else self._score_dataset_cfg.hf_cache_dir
        )
        corpus_subset_name: str = (
            normalize_optional_str(self._score_dataset_cfg.corpus_subset_name)
            or "corpus"
        )
        corpus_split_name: str = (
            normalize_optional_str(self._score_dataset_cfg.corpus_split_name) or "train"
        )
        query_corpus_data_files: Any | None = _normalize_data_files(
            self._score_dataset_cfg.query_corpus_hf_data_files
        )
        download_config = _resolve_download_config(
            self._scoring_cfg, local_files_only=local_files_only
        )
        return load_dataset(
            text_name,
            name=corpus_subset_name,
            split=corpus_split_name,
            cache_dir=text_cache_dir,
            data_files=query_corpus_data_files,
            download_config=download_config,
        )
