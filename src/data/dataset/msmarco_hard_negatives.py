import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from datasets import Dataset
from omegaconf import DictConfig

from src.data.dataclass import MetaItem
from src.data.dataset.base import BaseDataset
from src.data.dataset.hard_negative_selector import (
    HardNegativeSelectionSettings,
    select_hard_negative_doc_ids,
    to_doc_id_list,
)
from src.utils.logging import get_logger, is_rank_zero, log_if_rank_zero

_DEFAULT_MODEL_PRIORITY: tuple[str, ...] = (
    "msmarco-distilbert-base-tas-b",
    "msmarco-distilbert-base-v3",
    "msmarco-MiniLM-L-6-v3",
    "distilbert-margin_mse-cls-dot-v2",
    "distilbert-margin_mse-cls-dot-v1",
    "distilbert-margin_mse-mean-dot-v1",
    "mpnet-margin_mse-mean-v1",
    "co-condenser-margin_mse-cls-v1",
    "distilbert-margin_mse-mnrl-mean-v1",
    "distilbert-margin_mse-sym_mnrl-mean-v1",
    "distilbert-margin_mse-sym_mnrl-mean-v2",
    "co-condenser-margin_mse-sym_mnrl-mean-v1",
    "bm25",
)

_CACHE_SUBDIR: str = "splade_msmarco_hard_negatives"
_FILTER_CACHE_SUFFIX: str = ".filtered.arrow"
_PRECOMPUTE_CACHE_SUFFIX: str = ".precomputed.arrow"
_CACHE_WAIT_TIMEOUT_SECONDS: float = 900.0
_CACHE_WAIT_POLL_SECONDS: float = 0.5

_PRECOMPUTED_QID_COLUMN: str = "__splade_hn_qid"
_PRECOMPUTED_POS_IDS_COLUMN: str = "__splade_hn_pos_ids"
_PRECOMPUTED_NEG_IDS_COLUMN: str = "__splade_hn_neg_ids"

logger = get_logger("MSMARCOHardNegativesDataset")


@dataclass(frozen=True)
class _HardNegativeDatasetSettings:
    selector: HardNegativeSelectionSettings
    require_negatives: bool


def _normalize_data_files(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_data_files(mapped)
            for key, mapped in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_data_files(item) for item in value]
    return str(value)


def _parse_settings(cfg: DictConfig) -> _HardNegativeDatasetSettings:
    raw_selection_cfg: DictConfig | None = cfg.get("hard_negative_selection")
    if raw_selection_cfg is None:
        model_priority: tuple[str, ...] = _DEFAULT_MODEL_PRIORITY
        deprioritized_models: tuple[str, ...] = ("bm25",)
        append_unlisted_models: bool = True
        drop_positive_overlaps: bool = True
        dedupe: bool = True
        require_negatives: bool = True
    else:
        configured_priority = raw_selection_cfg.get("model_priority")
        if configured_priority is None:
            model_priority = _DEFAULT_MODEL_PRIORITY
        else:
            model_priority = tuple(str(key) for key in configured_priority)
        configured_deprioritized = raw_selection_cfg.get("deprioritized_models")
        if configured_deprioritized is None:
            deprioritized_models = ("bm25",)
        else:
            deprioritized_models = tuple(str(key) for key in configured_deprioritized)
        append_unlisted_models = bool(
            raw_selection_cfg.get("append_unlisted_models", True)
        )
        drop_positive_overlaps = bool(
            raw_selection_cfg.get("drop_positive_overlaps", True)
        )
        dedupe = bool(raw_selection_cfg.get("dedupe", True))
        require_negatives = bool(raw_selection_cfg.get("require_negatives", True))

    selector_settings = HardNegativeSelectionSettings(
        model_priority=model_priority,
        deprioritized_models=deprioritized_models,
        append_unlisted_models=append_unlisted_models,
        drop_positive_overlaps=drop_positive_overlaps,
        dedupe=dedupe,
    )
    return _HardNegativeDatasetSettings(
        selector=selector_settings,
        require_negatives=require_negatives,
    )


class MSMARCOHardNegativesDataset(BaseDataset):
    """MS MARCO hard negatives dataset with deterministic per-model sampling."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._settings: _HardNegativeDatasetSettings = _parse_settings(cfg)
        self._required_negatives: int = max(int(self.cfg.get("num_negatives", 0)), 0)

    def _resolve_qid(self, row: dict[str, Any], index: int) -> str:
        qid_value: Any | None = row.get(_PRECOMPUTED_QID_COLUMN)
        if qid_value is None:
            qid_value = row.get("qid")
        if qid_value is None:
            qid_value = row.get("query_id")
        qid: str = str(index) if qid_value is None else str(qid_value).strip()
        if not qid:
            return str(index)
        return qid

    def _extract_positive_ids(self, row: dict[str, Any]) -> list[str]:
        precomputed_ids: list[str] = to_doc_id_list(row.get(_PRECOMPUTED_POS_IDS_COLUMN))
        if precomputed_ids:
            return precomputed_ids
        positive_ids: list[str] = to_doc_id_list(row.get("pos"))
        if positive_ids:
            return positive_ids
        return to_doc_id_list(row.get("positive_id"))

    def _require_positive_ids(self, row: dict[str, Any], *, qid: str) -> list[str]:
        positive_ids: list[str] = self._extract_positive_ids(row)
        if positive_ids:
            return positive_ids
        raise ValueError(
            "MS MARCO hard-negatives row is missing positive ids "
            f"for qid={qid!r}."
        )

    def _select_negative_ids(
        self,
        row: dict[str, Any],
        *,
        positive_ids: list[str],
        target_count: int,
    ) -> list[str]:
        resolved_target_count: int = max(int(target_count), 0)
        if resolved_target_count <= 0:
            return []
        precomputed_negatives: list[str] = to_doc_id_list(
            row.get(_PRECOMPUTED_NEG_IDS_COLUMN)
        )
        if precomputed_negatives:
            return precomputed_negatives[:resolved_target_count]
        return select_hard_negative_doc_ids(
            row.get("neg"),
            positive_doc_ids=positive_ids,
            target_count=resolved_target_count,
            settings=self._settings.selector,
        )

    def _has_positive_ids(self, row: dict[str, Any]) -> bool:
        return bool(self._extract_positive_ids(row))

    def _has_usable_hard_negative(self, row: dict[str, Any]) -> bool:
        if not self._settings.require_negatives or self._required_negatives <= 0:
            return True
        positive_ids: list[str] = self._extract_positive_ids(row)
        if not positive_ids:
            return False
        selected: list[str] = self._select_negative_ids(
            row,
            positive_ids=positive_ids,
            target_count=1,
        )
        return bool(selected)

    def _is_trainable_row(self, row: dict[str, Any]) -> bool:
        return self._has_positive_ids(row) and self._has_usable_hard_negative(row)

    def _cache_key_payload(self, source_fingerprint: str) -> dict[str, Any]:
        selector = self._settings.selector
        return {
            "dataset_name": self.hf_name,
            "dataset_subset": self.hf_subset,
            "split": self.hf_split,
            "data_files": _normalize_data_files(self.hf_data_files),
            "source_fingerprint": source_fingerprint,
            "skip_samples": int(self.hf_skip_samples),
            "max_samples": (
                None if self.hf_max_samples is None else int(self.hf_max_samples)
            ),
            "required_negatives": int(self._required_negatives),
            "require_negatives": bool(self._settings.require_negatives),
            "selector": {
                "model_priority": list(selector.model_priority),
                "deprioritized_models": list(selector.deprioritized_models),
                "append_unlisted_models": bool(selector.append_unlisted_models),
                "drop_positive_overlaps": bool(selector.drop_positive_overlaps),
                "dedupe": bool(selector.dedupe),
            },
        }

    def _resolve_cache_root(self, dataset: Dataset) -> Path | None:
        cache_dir: str | None = self.hf_cache_dir
        if cache_dir is not None:
            root_path = Path(cache_dir) / _CACHE_SUBDIR
            root_path.mkdir(parents=True, exist_ok=True)
            return root_path
        cache_files: Any = getattr(dataset, "cache_files", None)
        if isinstance(cache_files, list):
            cache_file: Any
            for cache_file in cache_files:
                if not isinstance(cache_file, Mapping):
                    continue
                cache_file_name: Any | None = cache_file.get("filename")
                if cache_file_name is None:
                    continue
                base_dir = Path(str(cache_file_name)).parent
                root_path = base_dir / _CACHE_SUBDIR
                root_path.mkdir(parents=True, exist_ok=True)
                return root_path
        return None

    def _resolve_cache_prefix(self, dataset: Dataset) -> Path | None:
        cache_root: Path | None = self._resolve_cache_root(dataset)
        if cache_root is None:
            return None
        source_fingerprint: str = str(getattr(dataset, "_fingerprint", "unknown"))
        key_payload: dict[str, Any] = self._cache_key_payload(source_fingerprint)
        serialized_payload: str = json.dumps(
            key_payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        cache_key: str = hashlib.sha1(
            serialized_payload.encode("utf-8")
        ).hexdigest()[:24]
        return cache_root / cache_key

    def _wait_for_cache_file(self, cache_file: Path) -> bool:
        deadline: float = time.monotonic() + _CACHE_WAIT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if cache_file.exists():
                return True
            time.sleep(_CACHE_WAIT_POLL_SECONDS)
        return cache_file.exists()

    @staticmethod
    def _rank_scoped_cache_file(cache_file: Path) -> Path:
        rank_value: str = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
        try:
            rank_suffix: str = str(int(rank_value))
        except ValueError:
            rank_suffix = "0"
        return cache_file.with_name(
            f"{cache_file.stem}.rank{rank_suffix}{cache_file.suffix}"
        )

    def _filter_trainable_rows(
        self, dataset: Dataset, cache_prefix: Path | None
    ) -> Dataset:
        if cache_prefix is None:
            return dataset.filter(self._is_trainable_row)
        cache_file: Path = cache_prefix.with_suffix(_FILTER_CACHE_SUFFIX)
        if is_rank_zero():
            return dataset.filter(
                self._is_trainable_row,
                cache_file_name=str(cache_file),
                load_from_cache_file=True,
            )
        if not self._wait_for_cache_file(cache_file):
            cache_file = self._rank_scoped_cache_file(cache_file)
        return dataset.filter(
            self._is_trainable_row,
            cache_file_name=str(cache_file),
            load_from_cache_file=True,
        )

    def _precompute_row_fields(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        qid: str = self._resolve_qid(row, index)
        positive_ids: list[str] = self._require_positive_ids(row, qid=qid)
        selected_negatives: list[str] = self._select_negative_ids(
            row,
            positive_ids=positive_ids,
            target_count=self._required_negatives,
        )
        return {
            _PRECOMPUTED_QID_COLUMN: qid,
            _PRECOMPUTED_POS_IDS_COLUMN: positive_ids,
            _PRECOMPUTED_NEG_IDS_COLUMN: selected_negatives,
        }

    def _materialize_precomputed_fields(
        self, dataset: Dataset, cache_prefix: Path | None
    ) -> Dataset:
        if cache_prefix is None:
            return dataset.map(self._precompute_row_fields, with_indices=True)
        cache_file: Path = cache_prefix.with_suffix(_PRECOMPUTE_CACHE_SUFFIX)
        if is_rank_zero():
            return dataset.map(
                self._precompute_row_fields,
                with_indices=True,
                cache_file_name=str(cache_file),
                load_from_cache_file=True,
            )
        if not self._wait_for_cache_file(cache_file):
            cache_file = self._rank_scoped_cache_file(cache_file)
        return dataset.map(
            self._precompute_row_fields,
            with_indices=True,
            cache_file_name=str(cache_file),
            load_from_cache_file=True,
        )

    # --- Protected methods ---
    def _resolve_meta_dataset(self) -> Dataset:
        if self.hf_name is None:
            raise ValueError("hf_name must be set for HuggingFace datasets.")
        meta_dataset: Dataset = self._load_hf_dataset(
            hf_name=self.hf_name,
            hf_subset=self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
            data_files=self.hf_data_files,
        )
        meta_dataset = self._apply_hf_sample_window(meta_dataset)
        cache_prefix: Path | None = self._resolve_cache_prefix(meta_dataset)
        original_size: int = int(len(meta_dataset))
        filtered_dataset: Dataset = self._filter_trainable_rows(meta_dataset, cache_prefix)
        dropped_rows: int = original_size - int(len(filtered_dataset))
        if dropped_rows > 0:
            log_if_rank_zero(
                logger,
                "Dropped "
                f"{dropped_rows} malformed hard-negative rows from split={self.hf_split}.",
                level="warning",
            )
        return self._materialize_precomputed_fields(filtered_dataset, cache_prefix)

    def _row_to_meta_item(
        self,
        row: dict[str, Any],
        index: int,
        *,
        num_positives: int,
        num_negatives: int,
        rng: Any,
    ) -> MetaItem:
        _ = rng
        qid: str = self._resolve_qid(row, index)
        all_positive_ids: list[str] = self._require_positive_ids(row, qid=qid)

        resolved_num_positives: int = max(int(num_positives), 0)
        resolved_num_negatives: int = max(int(num_negatives), 0)
        pos_ids: list[str] = (
            all_positive_ids[:resolved_num_positives]
            if resolved_num_positives > 0
            else []
        )
        neg_ids: list[str] = self._select_negative_ids(
            row,
            positive_ids=all_positive_ids,
            target_count=resolved_num_negatives,
        )
        if (
            self._settings.require_negatives
            and resolved_num_negatives > 0
            and not neg_ids
        ):
            raise ValueError(
                "MS MARCO hard-negatives row is missing usable negative ids "
                f"for qid={qid!r}."
            )

        return MetaItem(
            qid=qid,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            pos_scores=None,
            neg_scores=None,
            query_text=None,
            pos_texts=None,
            neg_texts=None,
        )
