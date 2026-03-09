import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class HardNegativeSelectionSettings:
    """Settings for building hard-negative candidate pools from model buckets."""

    model_priority: tuple[str, ...]
    deprioritized_models: tuple[str, ...]
    append_unlisted_models: bool = True
    drop_positive_overlaps: bool = True
    dedupe: bool = True


def to_doc_id_list(value: Any) -> list[str]:
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


def _to_doc_id_list(value: Any) -> list[str]:
    # Backwards-compatible alias used internally and by older imports.
    return to_doc_id_list(value)


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


def partition_hard_negative_doc_ids(
    neg_value: Any,
    *,
    positive_doc_ids: Iterable[str],
    settings: HardNegativeSelectionSettings,
) -> tuple[list[str], list[str]]:
    """
    Build prioritized and deprioritized hard-negative pools.

    Candidate ids are filtered for positive overlaps and deduped across model
    buckets. IDs from deprioritized models (for example BM25) are returned in a
    separate tail pool so callers can backfill from them only when needed.
    """
    neg_map: dict[str, Any] = _normalize_neg_map(neg_value)
    if not neg_map:
        return [], []

    model_order: list[str] = resolve_model_order(neg_map, settings)
    positive_id_set: set[str] = {str(doc_id) for doc_id in positive_doc_ids}
    deprioritized_models: set[str] = set(settings.deprioritized_models)
    prioritized_doc_ids: list[str] = []
    deprioritized_doc_ids: list[str] = []
    selected_set: set[str] = set()

    model_key: str
    for model_key in model_order:
        candidate_ids: list[str] = _to_doc_id_list(neg_map.get(model_key))
        target_pool: list[str] = (
            deprioritized_doc_ids
            if model_key in deprioritized_models
            else prioritized_doc_ids
        )
        doc_id: str
        for doc_id in candidate_ids:
            if settings.drop_positive_overlaps and doc_id in positive_id_set:
                continue
            if settings.dedupe and doc_id in selected_set:
                continue
            target_pool.append(doc_id)
            if settings.dedupe:
                selected_set.add(doc_id)
    return prioritized_doc_ids, deprioritized_doc_ids


def _sample_candidate_pool(
    candidate_ids: list[str],
    *,
    target_count: int,
    rng: random.Random | None,
) -> list[str]:
    requested_count: int = max(int(target_count), 0)
    if requested_count <= 0 or not candidate_ids:
        return []
    if rng is None:
        return candidate_ids[:requested_count]
    if len(candidate_ids) <= requested_count:
        sampled_ids: list[str] = list(candidate_ids)
        rng.shuffle(sampled_ids)
        return sampled_ids
    return rng.sample(candidate_ids, requested_count)


def select_hard_negative_doc_ids_from_pools(
    *,
    prioritized_doc_ids: list[str],
    deprioritized_doc_ids: list[str],
    target_count: int,
    rng: random.Random | None = None,
) -> list[str]:
    """Select negatives from candidate pools, backfilling from deprioritized ids."""
    requested_count: int = max(int(target_count), 0)
    if requested_count <= 0:
        return []
    selected_doc_ids: list[str] = _sample_candidate_pool(
        prioritized_doc_ids,
        target_count=requested_count,
        rng=rng,
    )
    remaining_count: int = requested_count - len(selected_doc_ids)
    if remaining_count <= 0:
        return selected_doc_ids
    selected_doc_ids.extend(
        _sample_candidate_pool(
            deprioritized_doc_ids,
            target_count=remaining_count,
            rng=rng,
        )
    )
    return selected_doc_ids


def select_hard_negative_doc_ids(
    neg_value: Any,
    *,
    positive_doc_ids: Iterable[str],
    target_count: int,
    settings: HardNegativeSelectionSettings,
    rng: random.Random | None = None,
) -> list[str]:
    """
    Select hard negatives from prioritized candidate pools.

    When `rng` is provided, sampling is random within the non-deprioritized pool
    and only falls back to deprioritized pools (for example BM25) when required.
    Without `rng`, selection remains deterministic from the front of each pool.
    """
    prioritized_doc_ids: list[str]
    deprioritized_doc_ids: list[str]
    prioritized_doc_ids, deprioritized_doc_ids = partition_hard_negative_doc_ids(
        neg_value,
        positive_doc_ids=positive_doc_ids,
        settings=settings,
    )
    return select_hard_negative_doc_ids_from_pools(
        prioritized_doc_ids=prioritized_doc_ids,
        deprioritized_doc_ids=deprioritized_doc_ids,
        target_count=target_count,
        rng=rng,
    )
