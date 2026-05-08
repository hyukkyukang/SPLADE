"""Build local query/qrels artifacts for patent document retrieval evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset

from src.data.patent_text import format_patent_document_text
from src.data.patent_passages import clean_patent_passage_text
from src.utils.normalize import normalize_optional_str
from src.utils.huggingface import resolve_hf_token
from src.utils.script_setup import configure_script_environment

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=False,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


def _normalize_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_label_doc_ids(value: Any) -> list[str]:
    if value is None:
        return []
    raw_values: Sequence[Any]
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    elif isinstance(value, str):
        stripped_value: str = value.strip()
        if not stripped_value:
            return []
        try:
            parsed_value: Any = json.loads(stripped_value)
        except Exception:
            raw_values = [stripped_value]
        else:
            if isinstance(parsed_value, list):
                raw_values = list(parsed_value)
            else:
                raw_values = [parsed_value]
    else:
        raw_values = [value]
    normalized: list[str] = []
    raw_value: Any
    for raw_value in raw_values:
        doc_id: str = _normalize_id(raw_value)
        if not doc_id:
            continue
        normalized.append(doc_id)
    return normalized


def _resolve_tsv_cell(
    row: Sequence[str],
    *,
    column_idx: int,
    column_name: str,
    row_number: int,
    tsv_path: str | Path,
) -> str:
    if column_idx < 0:
        raise ValueError(f"{column_name} must be non-negative; got {column_idx}.")
    if column_idx >= len(row):
        raise ValueError(
            f"{tsv_path} row {row_number} is missing {column_name} at column index "
            f"{column_idx} (row has {len(row)} columns)."
        )
    return str(row[column_idx])


def _parse_tsv_labels(value: Any, *, label_separator: str | None = None) -> list[str]:
    normalized_separator: str | None = normalize_optional_str(label_separator)
    if normalized_separator is None:
        return _normalize_label_doc_ids(value)
    normalized_value: str | None = normalize_optional_str(value)
    if normalized_value is None:
        return []
    return _normalize_label_doc_ids(normalized_value.split(normalized_separator))


def collect_qa_tsv_artifacts(
    *,
    qa_tsv: str | Path,
    query_column: int = 0,
    label_column: int = 1,
    query_id_column: int | None = None,
    label_separator: str | None = None,
    has_header: bool = False,
) -> tuple[list[str], list[tuple[str, str, float]], dict[str, str], dict[str, Any]]:
    ordered_query_ids: list[str] = []
    seen_query_ids: set[str] = set()
    qrels_rows: list[tuple[str, str, float]] = []
    seen_qrel_pairs: set[tuple[str, str]] = set()
    query_text_by_id: dict[str, str] = {}

    empty_query_rows: int = 0
    empty_label_rows: int = 0
    duplicate_qrel_pairs: int = 0
    conflicting_query_text_rows: int = 0
    generated_query_ids: int = 0

    qa_path = Path(str(qa_tsv))
    with qa_path.open("r", encoding="utf-8", newline="") as qa_file:
        reader = csv.reader(qa_file, delimiter="\t")
        if has_header:
            next(reader, None)
        data_row_idx: int
        row: Sequence[str]
        for data_row_idx, row in enumerate(reader):
            row_number: int = data_row_idx + (2 if has_header else 1)
            query_text: str = clean_patent_passage_text(
                _resolve_tsv_cell(
                    row,
                    column_idx=int(query_column),
                    column_name="query_column",
                    row_number=row_number,
                    tsv_path=qa_path,
                )
            )
            if not query_text:
                empty_query_rows += 1
                continue
            label_values: list[str] = _parse_tsv_labels(
                _resolve_tsv_cell(
                    row,
                    column_idx=int(label_column),
                    column_name="label_column",
                    row_number=row_number,
                    tsv_path=qa_path,
                ),
                label_separator=label_separator,
            )
            if not label_values:
                empty_label_rows += 1
                continue

            if query_id_column is None:
                query_id = f"q{data_row_idx}"
                generated_query_ids += 1
            else:
                query_id = _normalize_id(
                    _resolve_tsv_cell(
                        row,
                        column_idx=int(query_id_column),
                        column_name="query_id_column",
                        row_number=row_number,
                        tsv_path=qa_path,
                    )
                )
                if not query_id:
                    query_id = f"q{data_row_idx}"
                    generated_query_ids += 1

            existing_query_text: str | None = query_text_by_id.get(query_id)
            if existing_query_text is None:
                query_text_by_id[query_id] = query_text
            elif existing_query_text != query_text:
                conflicting_query_text_rows += 1

            if query_id not in seen_query_ids:
                seen_query_ids.add(query_id)
                ordered_query_ids.append(query_id)

            doc_id: str
            for doc_id in label_values:
                qrel_pair: tuple[str, str] = (query_id, doc_id)
                if qrel_pair in seen_qrel_pairs:
                    duplicate_qrel_pairs += 1
                    continue
                seen_qrel_pairs.add(qrel_pair)
                qrels_rows.append((query_id, doc_id, 1.0))

    return ordered_query_ids, qrels_rows, query_text_by_id, {
        "qa_tsv": str(qa_path),
        "qa_has_header": bool(has_header),
        "qa_query_column": int(query_column),
        "qa_label_column": int(label_column),
        "qa_query_id_column": (
            None if query_id_column is None else int(query_id_column)
        ),
        "qa_label_separator": normalize_optional_str(label_separator),
        "generated_query_ids": generated_query_ids,
        "empty_query_rows": empty_query_rows,
        "empty_label_rows": empty_label_rows,
        "duplicate_qrel_pairs": duplicate_qrel_pairs,
        "conflicting_query_text_rows": conflicting_query_text_rows,
    }


def collect_benchmark_qrels(
    benchmark_rows: Iterable[Mapping[str, Any]],
    *,
    question_id_column: str = "question_id",
    label_id_column: str = "label_id",
) -> tuple[list[str], list[tuple[str, str, float]], dict[str, int]]:
    ordered_query_ids: list[str] = []
    seen_query_ids: set[str] = set()
    qrels_rows: list[tuple[str, str, float]] = []
    seen_qrel_pairs: set[tuple[str, str]] = set()
    empty_query_rows: int = 0
    empty_label_rows: int = 0
    duplicate_qrel_pairs: int = 0

    row: Mapping[str, Any]
    for row in benchmark_rows:
        query_id: str = _normalize_id(row.get(question_id_column))
        if not query_id:
            empty_query_rows += 1
            continue
        label_doc_ids: list[str] = _normalize_label_doc_ids(row.get(label_id_column))
        if not label_doc_ids:
            empty_label_rows += 1
            continue
        if query_id not in seen_query_ids:
            seen_query_ids.add(query_id)
            ordered_query_ids.append(query_id)
        doc_id: str
        for doc_id in label_doc_ids:
            qrel_pair: tuple[str, str] = (query_id, doc_id)
            if qrel_pair in seen_qrel_pairs:
                duplicate_qrel_pairs += 1
                continue
            seen_qrel_pairs.add(qrel_pair)
            qrels_rows.append((query_id, doc_id, 1.0))

    return ordered_query_ids, qrels_rows, {
        "empty_query_rows": empty_query_rows,
        "empty_label_rows": empty_label_rows,
        "duplicate_qrel_pairs": duplicate_qrel_pairs,
    }


def load_query_texts_from_parquet(
    *,
    corpus_glob: str,
    query_ids: Sequence[str],
    corpus_id_column: str = "doc_id",
    title_column: str = "title",
    abstract_column: str = "abstract",
    claims_column: str = "claims",
    description_column: str = "description",
    query_text_template: str = "patent_document_v1",
) -> dict[str, str]:
    corpus_paths: list[str] = sorted(glob(str(corpus_glob)))
    if not corpus_paths:
        raise FileNotFoundError(
            f"No corpus parquet files matched --corpus-glob={corpus_glob!r}."
        )
    query_id_set: set[str] = {str(query_id) for query_id in query_ids if str(query_id)}
    if not query_id_set:
        return {}
    corpus_dataset = ds.dataset(corpus_paths, format="parquet")
    query_table = corpus_dataset.to_table(
        columns=[
            corpus_id_column,
            title_column,
            abstract_column,
            claims_column,
            description_column,
        ],
        filter=pc.field(corpus_id_column).isin(sorted(query_id_set)),
    )
    query_text_by_id: dict[str, str] = {}
    row: dict[str, Any]
    for row in query_table.to_pylist():
        doc_id: str = _normalize_id(row.get(corpus_id_column))
        if not doc_id:
            continue
        query_text_by_id[doc_id] = _render_query_text(
            row,
            template_name=query_text_template,
            title_key=title_column,
            abstract_key=abstract_column,
            claims_key=claims_column,
            description_key=description_column,
        )
    return query_text_by_id


def load_query_texts_from_hf(
    *,
    corpus_repo: str,
    query_ids: Sequence[str],
    corpus_subset: str | None = None,
    corpus_split: str = "train",
    corpus_cache_dir: str | None = None,
    corpus_id_column: str = "doc_id",
    title_column: str = "title",
    abstract_column: str = "abstract",
    claims_column: str = "claims",
    description_column: str = "description",
    query_text_template: str = "patent_document_v1",
) -> dict[str, str]:
    token: str | None = resolve_hf_token()
    corpus_dataset = load_dataset(
        corpus_repo,
        name=corpus_subset,
        split=corpus_split,
        cache_dir=corpus_cache_dir,
        token=token,
        streaming=True,
    )
    query_id_set: set[str] = {str(query_id) for query_id in query_ids if str(query_id)}
    if not query_id_set:
        return {}
    query_text_by_id: dict[str, str] = {}
    row: Mapping[str, Any]
    for row in corpus_dataset:
        doc_id: str = _normalize_id(row.get(corpus_id_column))
        if not doc_id or doc_id not in query_id_set:
            continue
        query_text_by_id[doc_id] = _render_query_text(
            row,
            template_name=query_text_template,
            title_key=title_column,
            abstract_key=abstract_column,
            claims_key=claims_column,
            description_key=description_column,
        )
        if len(query_text_by_id) >= len(query_id_set):
            break
    return query_text_by_id


def _render_query_text(
    row: Mapping[str, Any],
    *,
    template_name: str,
    title_key: str,
    abstract_key: str,
    claims_key: str,
    description_key: str,
) -> str:
    normalized_template: str = str(template_name).strip().lower()
    if normalized_template in {"patent_document_v1", "full"}:
        return format_patent_document_text(
            row,
            title_key=title_key,
            abstract_key=abstract_key,
            claims_key=claims_key,
            description_key=description_key,
        )

    title: str = clean_patent_passage_text(row.get(title_key))
    abstract: str = clean_patent_passage_text(row.get(abstract_key))
    claims: str = clean_patent_passage_text(row.get(claims_key))
    description: str = clean_patent_passage_text(row.get(description_key))

    plain: bool = normalized_template.startswith("plain_")
    suffix: str = normalized_template[6:] if plain else normalized_template
    labeled_fields: list[str] = []
    plain_fields: list[str] = []

    def add_field(label: str, value: str) -> None:
        if not value:
            return
        labeled_fields.append(f"{label}: {value}")
        plain_fields.append(value)

    if suffix == "abstract":
        add_field("Abstract", abstract)
    elif suffix == "claims":
        add_field("Claims", claims)
    elif suffix == "title_abstract":
        add_field("Title", title)
        add_field("Abstract", abstract)
    elif suffix == "title_claims":
        add_field("Title", title)
        add_field("Claims", claims)
    elif suffix == "title_abstract_claims":
        add_field("Title", title)
        add_field("Abstract", abstract)
        add_field("Claims", claims)
    elif suffix == "title_abstract_description":
        add_field("Title", title)
        add_field("Abstract", abstract)
        add_field("Description", description)
    else:
        raise ValueError(f"Unsupported query_text_template: {template_name!r}")
    return "\n".join(plain_fields if plain else labeled_fields).strip()


def write_eval_artifacts(
    *,
    ordered_query_ids: Sequence[str],
    qrels_rows: Sequence[tuple[str, str, float]],
    query_text_by_id: Mapping[str, str],
    output_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    queries_query_ids: list[str] = []
    queries_texts: list[str] = []
    missing_query_ids: list[str] = []

    query_id: str
    for query_id in ordered_query_ids:
        query_text: str = str(query_text_by_id.get(str(query_id), "")).strip()
        if not query_text:
            missing_query_ids.append(str(query_id))
            continue
        queries_query_ids.append(str(query_id))
        queries_texts.append(query_text)

    allowed_query_ids: set[str] = set(queries_query_ids)
    filtered_qrels_query_ids: list[str] = []
    filtered_qrels_doc_ids: list[str] = []
    filtered_qrels_scores: list[float] = []
    seen_qrel_pairs: set[tuple[str, str]] = set()
    qrel_query_id: str
    doc_id: str
    score: float
    for qrel_query_id, doc_id, score in qrels_rows:
        if qrel_query_id not in allowed_query_ids:
            continue
        qrel_pair: tuple[str, str] = (str(qrel_query_id), str(doc_id))
        if qrel_pair in seen_qrel_pairs:
            continue
        seen_qrel_pairs.add(qrel_pair)
        filtered_qrels_query_ids.append(str(qrel_query_id))
        filtered_qrels_doc_ids.append(str(doc_id))
        filtered_qrels_scores.append(float(score))

    output_path = Path(str(output_dir))
    output_path.mkdir(parents=True, exist_ok=True)
    queries_path: Path = output_path / "queries.parquet"
    qrels_path: Path = output_path / "qrels.parquet"
    metadata_path: Path = output_path / "metadata.json"

    queries_table = pa.Table.from_pydict(
        {"query_id": queries_query_ids, "text": queries_texts},
        schema=pa.schema([("query_id", pa.string()), ("text", pa.string())]),
    )
    qrels_table = pa.Table.from_pydict(
        {
            "query_id": filtered_qrels_query_ids,
            "doc_id": filtered_qrels_doc_ids,
            "score": filtered_qrels_scores,
        },
        schema=pa.schema(
            [
                ("query_id", pa.string()),
                ("doc_id", pa.string()),
                ("score", pa.float32()),
            ]
        ),
    )
    pq.write_table(queries_table, queries_path)
    pq.write_table(qrels_table, qrels_path)

    metadata_dict: dict[str, Any] = dict(metadata or {})
    metadata_dict.update(
        {
            "query_count": len(queries_query_ids),
            "qrels_count": len(filtered_qrels_query_ids),
            "missing_query_count": len(missing_query_ids),
            "missing_query_ids_sample": missing_query_ids[:20],
            "queries_path": str(queries_path),
            "qrels_path": str(qrels_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
    return metadata_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-repo",
        default="Hyukkyu/patent-us",
        help="Hugging Face dataset with question_id and label_id.",
    )
    parser.add_argument(
        "--benchmark-subset",
        default=None,
        help="Optional Hugging Face dataset config/subset for the benchmark dataset.",
    )
    parser.add_argument(
        "--benchmark-split",
        default="test",
        help="Split from the benchmark dataset to convert.",
    )
    parser.add_argument(
        "--benchmark-cache-dir",
        default=None,
        help="Optional Hugging Face cache directory for the benchmark dataset.",
    )
    parser.add_argument(
        "--question-id-column",
        default="question_id",
        help="Column name for query document ids in the benchmark dataset.",
    )
    parser.add_argument(
        "--label-id-column",
        default="label_id",
        help="Column name for relevant document id lists in the benchmark dataset.",
    )
    parser.add_argument(
        "--qa-tsv",
        default=None,
        help=(
            "Optional QA TSV file to convert directly into queries/qrels. When set, "
            "benchmark and corpus query-text resolution settings are skipped."
        ),
    )
    parser.add_argument(
        "--qa-has-header",
        action="store_true",
        help="Treat --qa-tsv as a headered TSV and skip the first row.",
    )
    parser.add_argument(
        "--qa-query-column",
        type=int,
        default=0,
        help="Zero-based query text column in --qa-tsv.",
    )
    parser.add_argument(
        "--qa-label-column",
        type=int,
        default=1,
        help="Zero-based relevant-doc label column in --qa-tsv.",
    )
    parser.add_argument(
        "--qa-query-id-column",
        type=int,
        default=None,
        help=(
            "Optional zero-based query id column in --qa-tsv. When unset or empty, "
            "synthetic row-based query ids are generated."
        ),
    )
    parser.add_argument(
        "--qa-label-separator",
        default=None,
        help=(
            "Optional separator used to split multi-label --qa-tsv cells. When unset, "
            "label cells are treated as a single id unless they contain JSON arrays."
        ),
    )
    parser.add_argument(
        "--corpus-repo",
        default=None,
        help="Optional Hugging Face corpus dataset to resolve query document text.",
    )
    parser.add_argument(
        "--corpus-subset",
        default=None,
        help="Optional Hugging Face dataset config/subset for the corpus dataset.",
    )
    parser.add_argument(
        "--corpus-split",
        default="train",
        help="Split from the corpus dataset to use when --corpus-repo is set.",
    )
    parser.add_argument(
        "--corpus-cache-dir",
        default=None,
        help="Optional Hugging Face cache directory for the corpus dataset.",
    )
    parser.add_argument(
        "--corpus-glob",
        default=".cache/hf/patent-us-corpus/patent_us_docs_slice*.parquet",
        help="Glob for local patent corpus parquet files when --corpus-repo is not set.",
    )
    parser.add_argument(
        "--corpus-id-column",
        default="doc_id",
        help="Document id column in the corpus dataset.",
    )
    parser.add_argument(
        "--title-column",
        default="title",
        help="Title column in the corpus dataset.",
    )
    parser.add_argument(
        "--abstract-column",
        default="abstract",
        help="Abstract column in the corpus dataset.",
    )
    parser.add_argument(
        "--claims-column",
        default="claims",
        help="Claims column in the corpus dataset.",
    )
    parser.add_argument(
        "--description-column",
        default="description",
        help="Description column in the corpus dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval/patent_us",
        help="Directory where queries/qrels artifacts will be written.",
    )
    parser.add_argument(
        "--query-text-template",
        default="patent_document_v1",
        help=(
            "Patent query text template. Options include: patent_document_v1, "
            "title_abstract, plain_title_abstract, title_claims, title_abstract_claims, "
            "title_abstract_description, abstract, claims."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    qa_tsv: str | None = normalize_optional_str(args.qa_tsv)
    if qa_tsv is not None:
        ordered_query_ids, qrels_rows, query_text_by_id, qrels_metadata = (
            collect_qa_tsv_artifacts(
                qa_tsv=qa_tsv,
                query_column=int(args.qa_query_column),
                label_column=int(args.qa_label_column),
                query_id_column=(
                    None
                    if args.qa_query_id_column is None
                    else int(args.qa_query_id_column)
                ),
                label_separator=normalize_optional_str(args.qa_label_separator),
                has_header=bool(args.qa_has_header),
            )
        )
        metadata: dict[str, Any] = write_eval_artifacts(
            ordered_query_ids=ordered_query_ids,
            qrels_rows=qrels_rows,
            query_text_by_id=query_text_by_id,
            output_dir=str(args.output_dir),
            metadata={
                "benchmark_source": "qa_tsv",
                "benchmark_repo": None,
                "benchmark_subset": None,
                "benchmark_split": None,
                "corpus_repo": None,
                "corpus_subset": None,
                "corpus_split": None,
                "query_text_template": None,
                **qrels_metadata,
            },
        )
    else:
        token: str | None = resolve_hf_token()
        benchmark = load_dataset(
            str(args.benchmark_repo),
            name=normalize_optional_str(args.benchmark_subset),
            split=str(args.benchmark_split),
            cache_dir=normalize_optional_str(args.benchmark_cache_dir),
            token=token,
        )
        ordered_query_ids: list[str]
        qrels_rows: list[tuple[str, str, float]]
        qrels_metadata: dict[str, int]
        ordered_query_ids, qrels_rows, qrels_metadata = collect_benchmark_qrels(
            benchmark,
            question_id_column=str(args.question_id_column),
            label_id_column=str(args.label_id_column),
        )

        corpus_repo: str | None = normalize_optional_str(args.corpus_repo)
        if corpus_repo is not None:
            query_text_by_id = load_query_texts_from_hf(
                corpus_repo=corpus_repo,
                corpus_subset=normalize_optional_str(args.corpus_subset),
                corpus_split=str(args.corpus_split),
                corpus_cache_dir=normalize_optional_str(args.corpus_cache_dir),
                query_ids=ordered_query_ids,
                corpus_id_column=str(args.corpus_id_column),
                title_column=str(args.title_column),
                abstract_column=str(args.abstract_column),
                claims_column=str(args.claims_column),
                description_column=str(args.description_column),
                query_text_template=str(args.query_text_template),
            )
        else:
            query_text_by_id = load_query_texts_from_parquet(
                corpus_glob=str(args.corpus_glob),
                query_ids=ordered_query_ids,
                corpus_id_column=str(args.corpus_id_column),
                title_column=str(args.title_column),
                abstract_column=str(args.abstract_column),
                claims_column=str(args.claims_column),
                description_column=str(args.description_column),
                query_text_template=str(args.query_text_template),
            )

        metadata = write_eval_artifacts(
            ordered_query_ids=ordered_query_ids,
            qrels_rows=qrels_rows,
            query_text_by_id=query_text_by_id,
            output_dir=str(args.output_dir),
            metadata={
                "benchmark_source": "hf_dataset",
                "benchmark_repo": str(args.benchmark_repo),
                "benchmark_subset": normalize_optional_str(args.benchmark_subset),
                "benchmark_split": str(args.benchmark_split),
                "corpus_repo": corpus_repo,
                "corpus_subset": normalize_optional_str(args.corpus_subset),
                "corpus_split": str(args.corpus_split),
                "query_text_template": str(args.query_text_template),
                **qrels_metadata,
            },
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
