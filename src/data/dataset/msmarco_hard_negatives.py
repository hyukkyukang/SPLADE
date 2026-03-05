from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from omegaconf import DictConfig

from src.data.dataclass import MetaItem
from src.data.dataset.base import BaseDataset
from src.data.dataset.hard_negative_selector import (
    HardNegativeSelectionSettings,
    select_hard_negative_doc_ids,
)

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


@dataclass(frozen=True)
class _HardNegativeDatasetSettings:
    selector: HardNegativeSelectionSettings
    require_negatives: bool


def _to_doc_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    values: list[Any]
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    doc_ids: list[str] = []
    raw_value: Any
    for raw_value in values:
        doc_id: str = str(raw_value).strip()
        if doc_id:
            doc_ids.append(doc_id)
    return doc_ids


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
            deprioritized_models = tuple(
                str(key) for key in configured_deprioritized
            )
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
        return self._apply_hf_sample_window(meta_dataset)

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
        qid_value: Any | None = row.get("qid")
        if qid_value is None:
            qid_value = row.get("query_id")
        qid: str = str(index) if qid_value is None else str(qid_value).strip()
        if not qid:
            qid = str(index)

        all_positive_ids: list[str] = _to_doc_id_list(row.get("pos"))
        if not all_positive_ids:
            positive_id_value: Any | None = row.get("positive_id")
            all_positive_ids = _to_doc_id_list(positive_id_value)
        if not all_positive_ids:
            raise ValueError(
                "MS MARCO hard-negatives row is missing positive ids "
                f"for qid={qid!r}."
            )

        resolved_num_positives: int = max(int(num_positives), 0)
        resolved_num_negatives: int = max(int(num_negatives), 0)

        pos_ids: list[str] = (
            all_positive_ids[:resolved_num_positives]
            if resolved_num_positives > 0
            else []
        )
        neg_ids: list[str] = select_hard_negative_doc_ids(
            row.get("neg"),
            positive_doc_ids=all_positive_ids,
            target_count=resolved_num_negatives,
            settings=self._settings.selector,
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
