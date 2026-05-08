"""Build train-exclusive validation JSONL for patent hard-negative datasets.

The patent hard-negative datasets contain inline query, positive, and hard-negative
texts. This script keeps validation rows whose document/chunk identity fields do
not appear in the training split, then writes the rows as JSONL so the training
Hydra config can load them through ``hf_name=json``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import Dataset, load_dataset

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is optional for CLI use.

    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        _ = args, kwargs
        return False


DEFAULT_DOCUMENT_ID_COLUMNS: tuple[str, ...] = (
    "source_document_id",
    "positive_document_id",
)
DEFAULT_NODE_ID_COLUMNS: tuple[str, ...] = (
    "query_chunk_id",
    "positive_node_id",
)
DEFAULT_HARD_NEGATIVE_DOCUMENT_COLUMNS: tuple[str, ...] = (
    "hard_negative_document_ids",
)
DEFAULT_HARD_NEGATIVE_NODE_COLUMNS: tuple[str, ...] = (
    "hard_negative_node_ids",
)
DEFAULT_OUTPUT_PATH: Path = Path(
    "data/validation/patent25k_validation_exclusive_train_doc_chunk.jsonl"
)


@dataclass(frozen=True)
class ExclusiveGroup:
    """A set of validation columns checked against a train-side blocker set."""

    name: str
    validation_columns: tuple[str, ...]
    train_columns: tuple[str, ...]


@dataclass
class FilterResult:
    rows: list[dict[str, Any]]
    dropped_rows: int
    group_conflict_counts: dict[str, int]
    column_conflict_counts: dict[str, int]


def normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    return normalized


def dedupe_preserve_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return tuple(deduped)


def build_exclusive_groups(
    *,
    document_id_columns: Sequence[str] = DEFAULT_DOCUMENT_ID_COLUMNS,
    node_id_columns: Sequence[str] = DEFAULT_NODE_ID_COLUMNS,
    train_document_id_columns: Sequence[str] | None = None,
    train_node_id_columns: Sequence[str] | None = None,
    include_train_hard_negatives: bool = False,
    include_validation_hard_negatives: bool = False,
) -> tuple[ExclusiveGroup, ...]:
    """Build role-agnostic document and node/chunk exclusivity groups.

    By default, validation source/positive document IDs are checked against the
    union of train source/positive document IDs, and validation query/positive
    node IDs are checked against the union of train query/positive node IDs.
    """

    validation_doc_columns: tuple[str, ...] = dedupe_preserve_order(document_id_columns)
    validation_node_columns: tuple[str, ...] = dedupe_preserve_order(node_id_columns)
    train_doc_columns: tuple[str, ...] = dedupe_preserve_order(
        train_document_id_columns or document_id_columns
    )
    train_node_columns: tuple[str, ...] = dedupe_preserve_order(
        train_node_id_columns or node_id_columns
    )

    if include_train_hard_negatives:
        train_doc_columns = dedupe_preserve_order(
            (*train_doc_columns, *DEFAULT_HARD_NEGATIVE_DOCUMENT_COLUMNS)
        )
        train_node_columns = dedupe_preserve_order(
            (*train_node_columns, *DEFAULT_HARD_NEGATIVE_NODE_COLUMNS)
        )
    if include_validation_hard_negatives:
        validation_doc_columns = dedupe_preserve_order(
            (*validation_doc_columns, *DEFAULT_HARD_NEGATIVE_DOCUMENT_COLUMNS)
        )
        validation_node_columns = dedupe_preserve_order(
            (*validation_node_columns, *DEFAULT_HARD_NEGATIVE_NODE_COLUMNS)
        )

    return (
        ExclusiveGroup(
            name="document_id",
            validation_columns=validation_doc_columns,
            train_columns=train_doc_columns,
        ),
        ExclusiveGroup(
            name="node_id",
            validation_columns=validation_node_columns,
            train_columns=train_node_columns,
        ),
    )


def iter_column_values(row: Mapping[str, Any], column_name: str) -> list[str]:
    value: Any | None = row.get(column_name)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = (value,)

    values: list[str] = []
    for raw_value in raw_values:
        if raw_value is None:
            continue
        normalized = str(raw_value).strip()
        if normalized:
            values.append(normalized)
    return values


def validate_required_columns(
    dataset: Dataset,
    columns: Sequence[str],
    *,
    split_name: str,
) -> None:
    missing_columns = sorted(set(columns) - set(dataset.column_names))
    if missing_columns:
        raise ValueError(
            f"Split {split_name!r} is missing required columns: {missing_columns}"
        )


def validate_group_columns(
    *,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    groups: Sequence[ExclusiveGroup],
    train_split: str,
    validation_split: str,
) -> None:
    train_columns: list[str] = []
    validation_columns: list[str] = []
    for group in groups:
        train_columns.extend(group.train_columns)
        validation_columns.extend(group.validation_columns)
    validate_required_columns(
        train_dataset,
        dedupe_preserve_order(train_columns),
        split_name=train_split,
    )
    validate_required_columns(
        validation_dataset,
        dedupe_preserve_order(validation_columns),
        split_name=validation_split,
    )


def build_blocker_sets(
    train_dataset: Dataset,
    groups: Sequence[ExclusiveGroup],
) -> dict[str, set[str]]:
    blocker_sets: dict[str, set[str]] = {group.name: set() for group in groups}
    for row in train_dataset:
        for group in groups:
            blocker_set = blocker_sets[group.name]
            for column_name in group.train_columns:
                blocker_set.update(iter_column_values(row, column_name))
    return blocker_sets


def find_row_conflicts(
    row: Mapping[str, Any],
    groups: Sequence[ExclusiveGroup],
    blocker_sets: Mapping[str, set[str]],
) -> dict[str, dict[str, set[str]]]:
    conflicts: dict[str, dict[str, set[str]]] = {}
    for group in groups:
        group_blockers = blocker_sets[group.name]
        group_conflicts: dict[str, set[str]] = {}
        for column_name in group.validation_columns:
            overlapping_values = {
                value
                for value in iter_column_values(row, column_name)
                if value in group_blockers
            }
            if overlapping_values:
                group_conflicts[column_name] = overlapping_values
        if group_conflicts:
            conflicts[group.name] = group_conflicts
    return conflicts


def filter_exclusive_rows(
    validation_dataset: Dataset,
    groups: Sequence[ExclusiveGroup],
    blocker_sets: Mapping[str, set[str]],
) -> FilterResult:
    rows: list[dict[str, Any]] = []
    group_conflict_counts: Counter[str] = Counter()
    column_conflict_counts: Counter[str] = Counter()
    dropped_rows = 0

    for row in validation_dataset:
        conflicts = find_row_conflicts(
            row=row,
            groups=groups,
            blocker_sets=blocker_sets,
        )
        if conflicts:
            dropped_rows += 1
            for group_name, column_conflicts in conflicts.items():
                group_conflict_counts[group_name] += 1
                for column_name in column_conflicts:
                    column_conflict_counts[column_name] += 1
            continue
        rows.append(dict(row))

    return FilterResult(
        rows=rows,
        dropped_rows=dropped_rows,
        group_conflict_counts=dict(group_conflict_counts),
        column_conflict_counts=dict(column_conflict_counts),
    )


def write_jsonl(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False))
            output_file.write("\n")


def write_report(report: Mapping[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_hf_token(explicit_token: str | None) -> str | None:
    token = normalize_optional_str(explicit_token)
    if token is not None:
        return token
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = normalize_optional_str(os.getenv(env_name))
        if token is not None:
            return token
    return None


def load_patent_split(
    *,
    hf_name: str,
    hf_subset: str | None,
    split: str,
    cache_dir: str | None,
    token: str | None,
) -> Dataset:
    kwargs: dict[str, Any] = {
        "path": hf_name,
        "split": split,
        "cache_dir": cache_dir,
    }
    if hf_subset is not None:
        kwargs["name"] = hf_subset
    if token is not None:
        kwargs["token"] = token
    return load_dataset(**kwargs)


def build_report_payload(
    *,
    hf_name: str,
    hf_subset: str | None,
    train_split: str,
    validation_split: str,
    output_path: Path,
    groups: Sequence[ExclusiveGroup],
    blocker_sets: Mapping[str, set[str]],
    train_rows: int,
    validation_rows: int,
    filter_result: FilterResult,
) -> dict[str, Any]:
    return {
        "hf_name": hf_name,
        "hf_subset": hf_subset,
        "train_split": train_split,
        "validation_split": validation_split,
        "output_path": output_path.as_posix(),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "kept_rows": len(filter_result.rows),
        "dropped_rows": filter_result.dropped_rows,
        "groups": [
            {
                "name": group.name,
                "validation_columns": list(group.validation_columns),
                "train_columns": list(group.train_columns),
                "train_blocker_values": len(blocker_sets[group.name]),
            }
            for group in groups
        ],
        "group_conflict_counts": filter_result.group_conflict_counts,
        "column_conflict_counts": filter_result.column_conflict_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a train-exclusive validation JSONL for patent hard-negative datasets."
    )
    parser.add_argument("--hf-name", default="Hyukkyu/patent-25k")
    parser.add_argument("--hf-subset", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Defaults to <output>.report.json.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv file for private HF datasets.",
    )
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    parser.add_argument(
        "--document-id-columns",
        nargs="+",
        default=list(DEFAULT_DOCUMENT_ID_COLUMNS),
    )
    parser.add_argument(
        "--node-id-columns",
        nargs="+",
        default=list(DEFAULT_NODE_ID_COLUMNS),
    )
    parser.add_argument(
        "--train-document-id-columns",
        nargs="+",
        default=None,
        help="Defaults to --document-id-columns.",
    )
    parser.add_argument(
        "--train-node-id-columns",
        nargs="+",
        default=None,
        help="Defaults to --node-id-columns.",
    )
    parser.add_argument(
        "--include-train-hard-negatives",
        action="store_true",
        help="Also block validation rows whose IDs appear as train hard negatives.",
    )
    parser.add_argument(
        "--include-validation-hard-negatives",
        action="store_true",
        help="Also require validation hard-negative IDs to be absent from train blockers.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow writing an empty validation file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_file is not None and args.env_file.exists():
        load_dotenv(args.env_file)
    else:
        load_dotenv()

    hf_name = str(args.hf_name)
    hf_subset = normalize_optional_str(args.hf_subset)
    cache_dir = normalize_optional_str(args.cache_dir)
    token = resolve_hf_token(args.token)
    output_path = args.output
    report_path = args.report_output or output_path.with_suffix(
        output_path.suffix + ".report.json"
    )

    groups = build_exclusive_groups(
        document_id_columns=args.document_id_columns,
        node_id_columns=args.node_id_columns,
        train_document_id_columns=args.train_document_id_columns,
        train_node_id_columns=args.train_node_id_columns,
        include_train_hard_negatives=bool(args.include_train_hard_negatives),
        include_validation_hard_negatives=bool(args.include_validation_hard_negatives),
    )

    train_dataset = load_patent_split(
        hf_name=hf_name,
        hf_subset=hf_subset,
        split=str(args.train_split),
        cache_dir=cache_dir,
        token=token,
    )
    validation_dataset = load_patent_split(
        hf_name=hf_name,
        hf_subset=hf_subset,
        split=str(args.validation_split),
        cache_dir=cache_dir,
        token=token,
    )

    validate_group_columns(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        groups=groups,
        train_split=str(args.train_split),
        validation_split=str(args.validation_split),
    )

    blocker_sets = build_blocker_sets(train_dataset=train_dataset, groups=groups)
    filter_result = filter_exclusive_rows(
        validation_dataset=validation_dataset,
        groups=groups,
        blocker_sets=blocker_sets,
    )
    if not filter_result.rows and not args.allow_empty:
        raise ValueError(
            "Exclusive validation set is empty. Pass --allow-empty if this is expected."
        )

    write_jsonl(filter_result.rows, output_path)
    report = build_report_payload(
        hf_name=hf_name,
        hf_subset=hf_subset,
        train_split=str(args.train_split),
        validation_split=str(args.validation_split),
        output_path=output_path,
        groups=groups,
        blocker_sets=blocker_sets,
        train_rows=len(train_dataset),
        validation_rows=len(validation_dataset),
        filter_result=filter_result,
    )
    write_report(report, report_path)

    print(f"train rows: {len(train_dataset)}")
    print(f"validation rows: {len(validation_dataset)}")
    print(f"kept rows: {len(filter_result.rows)}")
    print(f"dropped rows: {filter_result.dropped_rows}")
    print(f"output: {output_path.as_posix()}")
    print(f"report: {report_path.as_posix()}")


if __name__ == "__main__":
    main()
