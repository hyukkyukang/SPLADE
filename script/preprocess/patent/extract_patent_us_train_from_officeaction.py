"""Reconstruct a candidate `Hyukkyu/patent-us` train split from office-action JSONL.

This script is intentionally heuristic-driven. The original extraction script that
built `usc102103_train.json` is not available in the repository, so this pipeline
tries to recreate the train split from:

1. `officeaction_102_103_20250105-20250330_cpc.jsonl`
2. the local patent corpus parquet export

The output is an HF-style ID-only train parquet with columns:

- `question_id`
- `label_id`

It can also emit an intermediate candidate JSONL and an optional comparison report
against a reference dataset such as `Hyukkyu/patent-us`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from datasets import load_dataset


QUESTION_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("question_id", pa.string()),
        pa.field("label_id", pa.list_(pa.string())),
    ]
)

CLAIM_BLOCK_START_RE: re.Pattern[str] = re.compile(
    r"(?im)^\s*Claim(?:s|\(s\))?\s+[^\n]{0,500}?\brejected under\b[^\n]*"
)
FIRST_SENTENCE_RE: re.Pattern[str] = re.compile(r"(?s)^(.+?[.!?])(?:\s|$)")


@dataclass(slots=True)
class CandidateExample:
    source_line: int
    patent_application_number: str
    section: str
    segment_index: int
    question: str
    candidate_positive_ids: list[str]
    positive_id_field: str | None


@dataclass(slots=True)
class BuildStats:
    rows_scanned: int = 0
    sections_with_text: int = 0
    emitted_candidate_examples: int = 0
    emitted_question_rows: int = 0
    empty_question_rows: int = 0
    rows_with_empty_label_id: int = 0
    distinct_positive_ids_needed: int = 0
    resolved_positive_ids: int = 0
    missing_positive_ids: int = 0


def _sha1_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}"


def normalize_whitespace(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def compose_doc_text(
    row: dict[str, Any],
    columns: Sequence[str],
    *,
    normalize: bool,
) -> str:
    parts: list[str] = []
    for column_name in columns:
        raw_value: Any | None = row.get(column_name)
        if raw_value is None:
            continue
        value: str = str(raw_value).strip()
        if not value:
            continue
        parts.append(normalize_whitespace(value) if normalize else value)
    return " ".join(parts).strip()


def normalize_identifier(value: str) -> str:
    text = value.strip().upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def normalize_match_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def split_rejection_text(text: str, mode: str) -> list[str]:
    raw_text = text.strip()
    if not raw_text:
        return []
    if mode == "full_text":
        return [raw_text]

    matches = list(CLAIM_BLOCK_START_RE.finditer(raw_text))
    if not matches:
        if mode == "claims_blocks":
            return []
        return [raw_text]

    segments: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        segment = raw_text[start:end].strip()
        if segment:
            segments.append(segment)
    return segments


def project_question_text(text: str, *, mode: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if mode == "segment":
        return stripped
    if mode == "first_line":
        for line in stripped.splitlines():
            candidate = line.strip()
            if candidate:
                return candidate
        return stripped
    if mode == "first_sentence":
        normalized = normalize_whitespace(stripped)
        match = FIRST_SENTENCE_RE.match(normalized)
        if match:
            return match.group(1).strip()
        return normalized
    raise ValueError(f"Unsupported question text mode: {mode}")


def _section_field_candidates(section_suffix: str) -> dict[str, str]:
    return {
        "dedup_application": f"DedupApplicationCheck{section_suffix}",
        "opensearch_application": f"OpenSearchApplicationMatches{section_suffix}",
        "search_application": f"SearchApplicationNumbers{section_suffix}",
        "cited_application": f"CitedApplicationNumbers{section_suffix}",
        "dedup_publication": f"DedupPublicationCheck{section_suffix}",
        "opensearch_publication": f"OpenSearchPublicationMatches{section_suffix}",
        "search_publication": f"SearchPublicationNumbers{section_suffix}",
        "cited_publication": f"CitedPublicationNumbers{section_suffix}",
    }


def _field_priority(
    *,
    section_suffix: str,
    prefer_application_matches: bool,
) -> list[str]:
    field_candidates = _section_field_candidates(section_suffix)
    if prefer_application_matches:
        logical_priority = [
            "dedup_application",
            "opensearch_application",
            "search_application",
            "cited_application",
            "dedup_publication",
            "opensearch_publication",
            "search_publication",
            "cited_publication",
        ]
    else:
        logical_priority = [
            "dedup_publication",
            "opensearch_publication",
            "search_publication",
            "cited_publication",
            "dedup_application",
            "opensearch_application",
            "search_application",
            "cited_application",
        ]
    return [field_candidates[name] for name in logical_priority]


def build_reference_items(
    section: dict[str, Any],
    *,
    section_suffix: str,
    prefer_application_matches: bool,
) -> list[dict[str, Any]]:
    field_names = _field_priority(
        section_suffix=section_suffix,
        prefer_application_matches=prefer_application_matches,
    )
    arrays: dict[str, list[str]] = {}
    max_len = 0
    for field_name in set(field_names + list(_section_field_candidates(section_suffix).values())):
        raw_values: Any | None = section.get(field_name)
        if isinstance(raw_values, list):
            arrays[field_name] = [str(value).strip() for value in raw_values]
            max_len = max(max_len, len(arrays[field_name]))
        else:
            arrays[field_name] = []

    items: list[dict[str, Any]] = []
    for index in range(max_len):
        item: dict[str, Any] = {"index": index, "markers": []}
        chosen_id = ""
        chosen_field = None
        for field_name in field_names:
            values = arrays.get(field_name, [])
            if index < len(values):
                value = values[index].strip()
                if value and not chosen_id:
                    chosen_id = value
                    chosen_field = field_name
        item["positive_id"] = normalize_identifier(chosen_id) if chosen_id else ""
        item["positive_id_field"] = chosen_field
        marker_field_names = [
            f"CitedPublicationNumbers{section_suffix}",
            f"SearchPublicationNumbers{section_suffix}",
            f"OpenSearchPublicationMatches{section_suffix}",
            f"DedupPublicationCheck{section_suffix}",
            f"CitedApplicationNumbers{section_suffix}",
            f"SearchApplicationNumbers{section_suffix}",
            f"OpenSearchApplicationMatches{section_suffix}",
            f"DedupApplicationCheck{section_suffix}",
        ]
        markers: list[str] = []
        for marker_field_name in marker_field_names:
            values = arrays.get(marker_field_name, [])
            if index >= len(values):
                continue
            value = values[index].strip()
            normalized = normalize_match_text(value)
            if normalized:
                markers.append(normalized)
        item["markers"] = _dedupe_preserve_order(markers)
        items.append(item)
    return items


def select_segment_positive_ids(
    section: dict[str, Any],
    segment_text: str,
    *,
    section_suffix: str,
    positive_id_mode: str,
    prefer_application_matches: bool,
    selection_scope: str,
) -> tuple[list[str], str | None]:
    section_level_ids, section_field = select_positive_ids(
        section,
        section_suffix=section_suffix,
        positive_id_mode=positive_id_mode,
        prefer_application_matches=prefer_application_matches,
    )
    section_level_ids = [
        normalize_identifier(value)
        for value in section_level_ids
        if normalize_identifier(value)
    ]
    if selection_scope == "section":
        return section_level_ids, section_field

    normalized_segment = normalize_match_text(segment_text)
    filtered_ids: list[str] = []
    filtered_fields: list[str] = []
    for item in build_reference_items(
        section,
        section_suffix=section_suffix,
        prefer_application_matches=prefer_application_matches,
    ):
        positive_id = str(item.get("positive_id", "")).strip()
        if not positive_id:
            continue
        markers: list[str] = list(item.get("markers", []))
        if not markers:
            continue
        if any(marker and marker in normalized_segment for marker in markers):
            filtered_ids.append(positive_id)
            positive_field = item.get("positive_id_field")
            if isinstance(positive_field, str) and positive_field:
                filtered_fields.append(positive_field)
    filtered_ids = _dedupe_preserve_order(filtered_ids)
    if filtered_ids:
        chosen_field = filtered_fields[0] if filtered_fields else section_field
        return filtered_ids, chosen_field
    if selection_scope == "segment_filtered":
        return [], section_field
    if selection_scope == "segment_filtered_or_section":
        return section_level_ids, section_field
    raise ValueError(f"Unsupported selection scope: {selection_scope}")


def select_positive_ids(
    section: dict[str, Any],
    *,
    section_suffix: str,
    positive_id_mode: str,
    prefer_application_matches: bool,
) -> tuple[list[str], str | None]:
    field_candidates = _section_field_candidates(section_suffix)
    if prefer_application_matches:
        priority = [
            "dedup_application",
            "opensearch_application",
            "search_application",
            "cited_application",
            "dedup_publication",
            "opensearch_publication",
            "search_publication",
            "cited_publication",
        ]
    else:
        priority = [
            "dedup_publication",
            "opensearch_publication",
            "search_publication",
            "cited_publication",
            "dedup_application",
            "opensearch_application",
            "search_application",
            "cited_application",
        ]

    selected_ids: list[str] = []
    selected_field: str | None = None
    for logical_name in priority:
        field_name = field_candidates[logical_name]
        raw_values: Any | None = section.get(field_name)
        if not isinstance(raw_values, list):
            continue
        cleaned = [str(value).strip() for value in raw_values if str(value).strip()]
        if not cleaned:
            continue
        if positive_id_mode == "first_nonempty":
            return _dedupe_preserve_order(cleaned), field_name
        if selected_field is None:
            selected_field = field_name
        selected_ids.extend(cleaned)
    return _dedupe_preserve_order(selected_ids), selected_field


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def iter_candidate_examples(
    officeaction_path: Path,
    *,
    segment_mode: str,
    question_text_mode: str,
    positive_id_mode: str,
    prefer_application_matches: bool,
    positive_selection_scope: str,
) -> Iterable[CandidateExample]:
    with officeaction_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            patent_application_number = str(row.get("patentApplicationNumber", ""))
            for section_name, section_suffix in (
                ("ClaimRejections102", "102"),
                ("ClaimRejections103", "103"),
            ):
                raw_section: Any | None = row.get(section_name)
                if not isinstance(raw_section, dict):
                    continue
                question_source = str(raw_section.get("text", "") or "").strip()
                if not question_source:
                    continue
                segments = split_rejection_text(question_source, mode=segment_mode)
                if not segments:
                    continue
                for segment_index, question in enumerate(segments):
                    stripped_question = project_question_text(
                        question,
                        mode=question_text_mode,
                    ).strip()
                    if not stripped_question:
                        continue
                    positive_ids, positive_field = select_segment_positive_ids(
                        raw_section,
                        question,
                        section_suffix=section_suffix,
                        positive_id_mode=positive_id_mode,
                        prefer_application_matches=prefer_application_matches,
                        selection_scope=positive_selection_scope,
                    )
                    yield CandidateExample(
                        source_line=line_number,
                        patent_application_number=patent_application_number,
                        section=section_suffix,
                        segment_index=segment_index,
                        question=stripped_question,
                        candidate_positive_ids=positive_ids,
                        positive_id_field=positive_field,
                    )


def write_candidate_examples_jsonl(
    officeaction_path: Path,
    candidate_jsonl_path: Path,
    *,
    segment_mode: str,
    question_text_mode: str,
    positive_id_mode: str,
    prefer_application_matches: bool,
    positive_selection_scope: str,
) -> tuple[BuildStats, set[str], Counter[str]]:
    stats = BuildStats()
    needed_positive_ids: set[str] = set()
    positive_field_usage: Counter[str] = Counter()
    candidate_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with candidate_jsonl_path.open("w", encoding="utf-8") as writer:
        with officeaction_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                stats.rows_scanned += 1
                row = json.loads(raw_line)
                patent_application_number = str(row.get("patentApplicationNumber", ""))
                for section_name, section_suffix in (
                    ("ClaimRejections102", "102"),
                    ("ClaimRejections103", "103"),
                ):
                    raw_section: Any | None = row.get(section_name)
                    if not isinstance(raw_section, dict):
                        continue
                    question_source = str(raw_section.get("text", "") or "").strip()
                    if not question_source:
                        continue
                    stats.sections_with_text += 1
                    segments = split_rejection_text(question_source, mode=segment_mode)
                    if not segments:
                        continue
                    for segment_index, question in enumerate(segments):
                        stripped_question = project_question_text(
                            question,
                            mode=question_text_mode,
                        ).strip()
                        if not stripped_question:
                            continue
                        positive_ids, positive_field = select_segment_positive_ids(
                            raw_section,
                            question,
                            section_suffix=section_suffix,
                            positive_id_mode=positive_id_mode,
                            prefer_application_matches=prefer_application_matches,
                            selection_scope=positive_selection_scope,
                        )
                        normalized_ids = [
                            normalized
                            for value in positive_ids
                            if (normalized := normalize_identifier(value))
                        ]
                        needed_positive_ids.update(normalized_ids)
                        if positive_field:
                            positive_field_usage[positive_field] += 1
                        stats.emitted_candidate_examples += 1
                        example = CandidateExample(
                            source_line=line_number,
                            patent_application_number=patent_application_number,
                            section=section_suffix,
                            segment_index=segment_index,
                            question=stripped_question,
                            candidate_positive_ids=normalized_ids,
                            positive_id_field=positive_field,
                        )
                        writer.write(json.dumps(asdict(example), ensure_ascii=False))
                        writer.write("\n")

    stats.distinct_positive_ids_needed = len(needed_positive_ids)
    return stats, needed_positive_ids, positive_field_usage


def build_patent_text_lookup(
    corpus_paths: Sequence[str],
    target_ids: set[str],
    *,
    columns: Sequence[str],
    normalize_doc_whitespace: bool,
) -> dict[str, str]:
    if not target_ids:
        return {}
    dataset = ds.dataset(list(corpus_paths), format="parquet")
    table = dataset.to_table(
        columns=["doc_id", "application_id", *columns],
        filter=(
            pc.field("application_id").isin(sorted(target_ids))
            | pc.field("doc_id").isin(sorted(target_ids))
        ),
    )
    lookup: dict[str, str] = {}
    for row in table.to_pylist():
        doc_text = compose_doc_text(row, columns=columns, normalize=normalize_doc_whitespace)
        if not doc_text:
            continue
        doc_id = normalize_identifier(str(row.get("doc_id", "") or ""))
        application_id = normalize_identifier(str(row.get("application_id", "") or ""))
        if doc_id:
            lookup.setdefault(doc_id, doc_text)
        if application_id:
            lookup.setdefault(application_id, doc_text)
    return lookup


def write_hf_like_train_parquet(
    candidate_jsonl_path: Path,
    train_parquet_path: Path,
    *,
    patent_text_lookup: dict[str, str],
    normalize_question_whitespace: bool,
    keep_empty_labels: bool,
) -> BuildStats:
    stats = BuildStats()
    train_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_buffer: list[dict[str, Any]] = []
    missing_positive_ids: set[str] = set()
    resolved_positive_ids: set[str] = set()

    try:
        with candidate_jsonl_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                question = str(payload["question"])
                if normalize_question_whitespace:
                    question = normalize_whitespace(question)
                if not question:
                    stats.empty_question_rows += 1
                    continue

                label_ids: list[str] = []
                for positive_id in payload.get("candidate_positive_ids", []):
                    positive_key = normalize_identifier(str(positive_id))
                    if not positive_key:
                        continue
                    doc_text = patent_text_lookup.get(positive_key)
                    if not doc_text:
                        missing_positive_ids.add(positive_key)
                        continue
                    resolved_positive_ids.add(positive_key)
                    label_ids.append(_sha1_id("d_", doc_text))
                label_ids = _dedupe_preserve_order(label_ids)
                if not label_ids and not keep_empty_labels:
                    continue
                if not label_ids:
                    stats.rows_with_empty_label_id += 1

                rows_buffer.append(
                    {
                        "question_id": _sha1_id("q_", question),
                        "label_id": label_ids,
                    }
                )
                stats.emitted_question_rows += 1

                if len(rows_buffer) >= 10000:
                    table = pa.Table.from_pylist(rows_buffer, schema=QUESTION_SCHEMA)
                    if writer is None:
                        writer = pq.ParquetWriter(train_parquet_path.as_posix(), QUESTION_SCHEMA)
                    writer.write_table(table)
                    rows_buffer = []

        if rows_buffer:
            table = pa.Table.from_pylist(rows_buffer, schema=QUESTION_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(train_parquet_path.as_posix(), QUESTION_SCHEMA)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    stats.resolved_positive_ids = len(resolved_positive_ids)
    stats.missing_positive_ids = len(missing_positive_ids)
    return stats


def load_reference_rows(
    *,
    reference_repo: str | None,
    reference_split: str,
    reference_parquet: str | None,
) -> list[dict[str, Any]]:
    if reference_parquet:
        table = pq.read_table(reference_parquet, columns=["question_id", "label_id"])
        return table.to_pylist()
    if not reference_repo:
        return []
    dataset = load_dataset(reference_repo, split=reference_split)
    return [
        {
            "question_id": str(row["question_id"]),
            "label_id": [str(value) for value in row.get("label_id", [])],
        }
        for row in dataset
    ]


def compare_rows(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_counter = Counter(
        (str(row["question_id"]), tuple(str(v) for v in row.get("label_id", [])))
        for row in candidate_rows
    )
    reference_counter = Counter(
        (str(row["question_id"]), tuple(str(v) for v in row.get("label_id", [])))
        for row in reference_rows
    )

    exact_row_overlap = sum(
        min(count, reference_counter.get(key, 0))
        for key, count in candidate_counter.items()
    )
    candidate_question_ids = {str(row["question_id"]) for row in candidate_rows}
    reference_question_ids = {str(row["question_id"]) for row in reference_rows}
    candidate_only = sorted(candidate_question_ids - reference_question_ids)
    reference_only = sorted(reference_question_ids - candidate_question_ids)

    return {
        "candidate_rows": len(candidate_rows),
        "reference_rows": len(reference_rows),
        "shared_question_ids": len(candidate_question_ids & reference_question_ids),
        "candidate_only_question_ids": len(candidate_only),
        "reference_only_question_ids": len(reference_only),
        "exact_row_overlap": exact_row_overlap,
        "candidate_only_question_ids_sample": candidate_only[:20],
        "reference_only_question_ids_sample": reference_only[:20],
    }


def load_candidate_rows(train_parquet_path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(train_parquet_path.as_posix(), columns=["question_id", "label_id"])
    return table.to_pylist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--officeaction-path",
        type=Path,
        default=Path("officeaction_102_103_20250105-20250330_cpc.jsonl"),
        help="Path to office-action JSONL.",
    )
    parser.add_argument(
        "--corpus-glob",
        default=".cache/hf/patent-us-corpus/patent_us_docs_slice*.parquet",
        help="Glob for local patent corpus parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/patent/reconstructed_patent_us"),
        help="Output directory for candidate artifacts.",
    )
    parser.add_argument(
        "--segment-mode",
        choices=["claims_blocks", "claims_blocks_or_full_text", "full_text"],
        default="claims_blocks_or_full_text",
        help="How to segment rejection narratives into questions.",
    )
    parser.add_argument(
        "--question-text-mode",
        choices=["segment", "first_line", "first_sentence"],
        default="segment",
        help="How to project each segment into the final hashed question text.",
    )
    parser.add_argument(
        "--positive-id-mode",
        choices=["first_nonempty", "merge_nonempty"],
        default="first_nonempty",
        help="How to select positive IDs from citation-related arrays.",
    )
    parser.add_argument(
        "--prefer-publication-matches",
        action="store_true",
        help="Prefer publication-based match arrays before application-based arrays.",
    )
    parser.add_argument(
        "--positive-selection-scope",
        choices=["section", "segment_filtered", "segment_filtered_or_section"],
        default="section",
        help="Whether to use section-level positives or filter them by segment-local citation markers.",
    )
    parser.add_argument(
        "--doc-columns",
        nargs="+",
        default=["title", "abstract", "claims"],
        help="Patent corpus columns used to compose positive document text.",
    )
    parser.add_argument(
        "--normalize-question-whitespace",
        action="store_true",
        help="Collapse question whitespace before hashing.",
    )
    parser.add_argument(
        "--normalize-doc-whitespace",
        action="store_true",
        help="Collapse patent corpus whitespace before hashing positives.",
    )
    parser.add_argument(
        "--drop-empty-labels",
        action="store_true",
        help="Drop rows that resolve to no positive labels.",
    )
    parser.add_argument(
        "--reference-repo",
        default=None,
        help="Optional reference HF dataset repo, e.g. Hyukkyu/patent-us.",
    )
    parser.add_argument(
        "--reference-split",
        default="train",
        help="Reference split name when --reference-repo is set.",
    )
    parser.add_argument(
        "--reference-parquet",
        default=None,
        help="Optional local parquet path with question_id/label_id to compare against.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    officeaction_path: Path = Path(args.officeaction_path)
    if not officeaction_path.exists():
        raise FileNotFoundError(f"Missing office-action file: {officeaction_path}")

    corpus_paths = sorted(glob(str(args.corpus_glob)))
    if not corpus_paths:
        raise FileNotFoundError(f"No parquet files matched --corpus-glob={args.corpus_glob!r}")

    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    candidate_jsonl_path = output_dir / "candidate_examples.jsonl"
    train_parquet_path = data_dir / "train-00000-of-00001.parquet"
    metadata_path = output_dir / "metadata.json"
    comparison_path = output_dir / "comparison.json"

    candidate_stats, needed_positive_ids, positive_field_usage = write_candidate_examples_jsonl(
        officeaction_path=officeaction_path,
        candidate_jsonl_path=candidate_jsonl_path,
        segment_mode=args.segment_mode,
        question_text_mode=args.question_text_mode,
        positive_id_mode=args.positive_id_mode,
        prefer_application_matches=not bool(args.prefer_publication_matches),
        positive_selection_scope=str(args.positive_selection_scope),
    )

    patent_text_lookup = build_patent_text_lookup(
        corpus_paths=corpus_paths,
        target_ids=needed_positive_ids,
        columns=args.doc_columns,
        normalize_doc_whitespace=bool(args.normalize_doc_whitespace),
    )

    question_stats = write_hf_like_train_parquet(
        candidate_jsonl_path=candidate_jsonl_path,
        train_parquet_path=train_parquet_path,
        patent_text_lookup=patent_text_lookup,
        normalize_question_whitespace=bool(args.normalize_question_whitespace),
        keep_empty_labels=not bool(args.drop_empty_labels),
    )

    metadata = {
        "officeaction_path": officeaction_path.as_posix(),
        "corpus_glob": str(args.corpus_glob),
        "segment_mode": str(args.segment_mode),
        "question_text_mode": str(args.question_text_mode),
        "positive_id_mode": str(args.positive_id_mode),
        "prefer_publication_matches": bool(args.prefer_publication_matches),
        "positive_selection_scope": str(args.positive_selection_scope),
        "doc_columns": [str(value) for value in args.doc_columns],
        "normalize_question_whitespace": bool(args.normalize_question_whitespace),
        "normalize_doc_whitespace": bool(args.normalize_doc_whitespace),
        "drop_empty_labels": bool(args.drop_empty_labels),
        "candidate_examples_path": candidate_jsonl_path.as_posix(),
        "train_parquet_path": train_parquet_path.as_posix(),
        "candidate_stats": asdict(candidate_stats),
        "question_stats": asdict(question_stats),
        "positive_field_usage": dict(positive_field_usage),
        "patent_text_lookup_size": len(patent_text_lookup),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison_payload: dict[str, Any] | None = None
    if args.reference_repo or args.reference_parquet:
        try:
            reference_rows = load_reference_rows(
                reference_repo=args.reference_repo,
                reference_split=str(args.reference_split),
                reference_parquet=args.reference_parquet,
            )
            candidate_rows = load_candidate_rows(train_parquet_path)
            comparison_payload = compare_rows(candidate_rows, reference_rows)
        except Exception as exc:  # pragma: no cover - best-effort diagnostic path
            comparison_payload = {
                "error": str(exc),
                "reference_repo": args.reference_repo,
                "reference_split": args.reference_split,
                "reference_parquet": args.reference_parquet,
            }
        comparison_path.write_text(
            json.dumps(comparison_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if comparison_payload is not None:
        print(json.dumps(comparison_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
