from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset


@dataclass(frozen=True)
class TextPair:
    query: str
    positive: str
    query_id: str | None = None
    positive_id: str | None = None


def load_hf_split(
    *,
    hf_name: str,
    hf_subset: str | None,
    split: str,
    cache_dir: str | None,
    data_files: Mapping[str, Any] | None,
) -> Dataset | IterableDataset:
    kwargs: dict[str, Any] = {
        "path": hf_name,
        "name": hf_subset,
        "split": split,
        "cache_dir": cache_dir,
    }
    if data_files is not None:
        kwargs["data_files"] = dict(data_files)
    return load_dataset(**kwargs)


def load_hf_splits(
    *,
    hf_name: str,
    hf_subset: str | None,
    splits: Iterable[str],
    cache_dir: str | None,
    data_files: Mapping[str, Any] | None,
    allow_missing_split: bool,
) -> list[Dataset | IterableDataset]:
    datasets: list[Dataset | IterableDataset] = []
    split: str
    for split in splits:
        split_name: str = str(split).strip()
        if not split_name:
            continue
        try:
            datasets.append(
                load_hf_split(
                    hf_name=hf_name,
                    hf_subset=hf_subset,
                    split=split_name,
                    cache_dir=cache_dir,
                    data_files=data_files,
                )
            )
        except Exception:
            if not allow_missing_split:
                raise
    return datasets


def maybe_concat_datasets(
    datasets: list[Dataset | IterableDataset],
) -> Dataset | IterableDataset:
    if not datasets:
        raise ValueError("No dataset splits were loaded.")
    if len(datasets) == 1:
        return datasets[0]

    first = datasets[0]
    if isinstance(first, Dataset):
        if not all(isinstance(dataset, Dataset) for dataset in datasets):
            raise ValueError("Cannot concatenate a mixed list of map and iterable datasets.")
        from datasets import concatenate_datasets

        return concatenate_datasets(list(datasets))

    # Iterable datasets cannot be concatenated via concatenate_datasets.
    # Wrap as a DatasetDict-like merged iterator.
    class _MergedIterable(IterableDataset):
        def __iter__(self):
            for dataset in datasets:
                for row in dataset:
                    yield row

    return _MergedIterable()


def resolve_first_present_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    column_set: set[str] = {str(name) for name in columns}
    candidate: str
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def lookup_texts_by_ids(
    *,
    dataset: Dataset | IterableDataset,
    id_column: str,
    text_column: str,
    wanted_ids: set[str],
) -> dict[str, str]:
    remaining: set[str] = set(wanted_ids)
    resolved: dict[str, str] = {}
    if not remaining:
        return resolved

    row: dict[str, Any]
    for row in dataset:
        row_id: Any | None = row.get(id_column)
        if row_id is None:
            continue
        row_id_str: str = str(row_id)
        if row_id_str not in remaining:
            continue
        text_value: Any | None = row.get(text_column)
        if text_value is None:
            continue
        text: str = str(text_value).strip()
        if not text:
            continue
        resolved[row_id_str] = text
        remaining.remove(row_id_str)
        if not remaining:
            break
    return resolved


def _as_text(value: Any | None) -> str | None:
    if value is None:
        return None
    text: str = str(value).strip()
    if not text:
        return None
    return text


def build_text_pairs(
    *,
    meta_dataset: Dataset | IterableDataset,
    query_text_column: str | None,
    positive_text_column: str | None,
    query_id_column: str,
    positive_id_column: str,
    query_lookup: dict[str, str] | None,
    corpus_lookup: dict[str, str] | None,
    max_pairs: int | None = None,
) -> list[TextPair]:
    pairs: list[TextPair] = []

    row: dict[str, Any]
    for row in meta_dataset:
        if query_text_column is not None and positive_text_column is not None:
            query_text: str | None = _as_text(row.get(query_text_column))
            positive_text: str | None = _as_text(row.get(positive_text_column))
            if query_text is None or positive_text is None:
                continue
            pairs.append(TextPair(query=query_text, positive=positive_text))
        else:
            query_id_value: Any | None = row.get(query_id_column)
            positive_id_value: Any | None = row.get(positive_id_column)
            if query_id_value is None or positive_id_value is None:
                continue
            query_id: str = str(query_id_value)
            positive_id: str = str(positive_id_value)
            if query_lookup is None or corpus_lookup is None:
                raise ValueError("query_lookup and corpus_lookup must be provided for ID mode.")
            query_text = query_lookup.get(query_id)
            positive_text = corpus_lookup.get(positive_id)
            if query_text is None or positive_text is None:
                continue
            pairs.append(
                TextPair(
                    query=query_text,
                    positive=positive_text,
                    query_id=query_id,
                    positive_id=positive_id,
                )
            )

        if max_pairs is not None and len(pairs) >= int(max_pairs):
            break

    return pairs


def collect_required_ids(
    *,
    meta_dataset: Dataset | IterableDataset,
    query_id_column: str,
    positive_id_column: str,
    max_rows: int | None,
) -> tuple[set[str], set[str], int]:
    query_ids: set[str] = set()
    positive_ids: set[str] = set()
    rows_seen: int = 0

    row: dict[str, Any]
    for row in meta_dataset:
        qid: Any | None = row.get(query_id_column)
        pid: Any | None = row.get(positive_id_column)
        if qid is None or pid is None:
            continue
        query_ids.add(str(qid))
        positive_ids.add(str(pid))
        rows_seen += 1
        if max_rows is not None and rows_seen >= int(max_rows):
            break

    return query_ids, positive_ids, rows_seen


def column_names_of(dataset: Dataset | IterableDataset) -> list[str]:
    if isinstance(dataset, Dataset):
        return list(dataset.column_names)
    if isinstance(dataset, DatasetDict):
        # Unused for now; normalize best-effort.
        names: list[str] = []
        for split_dataset in dataset.values():
            for col in split_dataset.column_names:
                if col not in names:
                    names.append(col)
        return names
    # IterableDataset can expose features.
    features: Any = getattr(dataset, "features", None)
    if features is None:
        return []
    return list(features.keys())
