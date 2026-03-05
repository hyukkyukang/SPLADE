from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class HardNegativeSelectionSettings:
    """Settings for deterministic hard-negative selection from model buckets."""

    model_priority: tuple[str, ...]
    deprioritized_models: tuple[str, ...]
    append_unlisted_models: bool = True
    drop_positive_overlaps: bool = True
    dedupe: bool = True


def _to_doc_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values: list[Any] = list(value)
    else:
        values = [value]
    doc_ids: list[str] = []
    raw_value: Any
    for raw_value in values:
        doc_id: str = str(raw_value).strip()
        if doc_id:
            doc_ids.append(doc_id)
    return doc_ids


def _normalize_neg_map(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): mapped for key, mapped in value.items()}
    return {"neg": value}


def resolve_model_order(
    neg_map: Mapping[str, Any], settings: HardNegativeSelectionSettings
) -> list[str]:
    """Resolve an ordered list of model keys to consume for negative selection."""
    row_keys: list[str] = list(neg_map.keys())
    if settings.model_priority:
        ordered_keys: list[str] = [
            key for key in settings.model_priority if key in neg_map
        ]
    else:
        ordered_keys = list(row_keys)

    if settings.append_unlisted_models:
        seen: set[str] = set(ordered_keys)
        ordered_keys.extend(key for key in row_keys if key not in seen)

    deprioritized: set[str] = set(settings.deprioritized_models)
    if not deprioritized:
        return ordered_keys

    head_keys: list[str] = [key for key in ordered_keys if key not in deprioritized]
    tail_keys: list[str] = [key for key in ordered_keys if key in deprioritized]
    return head_keys + tail_keys


def select_hard_negative_doc_ids(
    neg_value: Any,
    *,
    positive_doc_ids: Iterable[str],
    target_count: int,
    settings: HardNegativeSelectionSettings,
) -> list[str]:
    """
    Deterministically select up to target_count hard negatives.

    Selection proceeds model-by-model in resolved priority order and consumes each
    model list from the front. If one model does not provide enough negatives,
    selection continues to the next model.
    """
    requested_count: int = int(target_count)
    if requested_count <= 0:
        return []

    neg_map: dict[str, Any] = _normalize_neg_map(neg_value)
    if not neg_map:
        return []

    model_order: list[str] = resolve_model_order(neg_map, settings)
    positive_id_set: set[str] = {str(doc_id) for doc_id in positive_doc_ids}
    selected_doc_ids: list[str] = []
    selected_set: set[str] = set()

    model_key: str
    for model_key in model_order:
        candidate_ids: list[str] = _to_doc_id_list(neg_map.get(model_key))
        doc_id: str
        for doc_id in candidate_ids:
            if settings.drop_positive_overlaps and doc_id in positive_id_set:
                continue
            if settings.dedupe and doc_id in selected_set:
                continue
            selected_doc_ids.append(doc_id)
            if settings.dedupe:
                selected_set.add(doc_id)
            if len(selected_doc_ids) >= requested_count:
                return selected_doc_ids
    return selected_doc_ids
