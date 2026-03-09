"""Build local query/qrels artifacts for patent document retrieval evaluation."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from datasets import load_dataset


def _compose_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for column_name in ("title", "abstract", "claims"):
        raw_value: Any | None = row.get(column_name)
        if raw_value is None:
            continue
        value: str = str(raw_value).strip()
        if value:
            parts.append(value)
    return " ".join(parts).strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-repo",
        default="Hyukkyu/patent-us",
        help="Hugging Face dataset with question_id and label_id.",
    )
    parser.add_argument(
        "--benchmark-split",
        default="test",
        help="Split from the benchmark dataset to convert.",
    )
    parser.add_argument(
        "--corpus-glob",
        default=".cache/hf/patent-us-corpus/patent_us_docs_slice*.parquet",
        help="Glob for local patent corpus parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval/patent_us",
        help="Directory where queries/qrels artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    corpus_paths: list[str] = sorted(glob(str(args.corpus_glob)))
    if not corpus_paths:
        raise FileNotFoundError(
            f"No corpus parquet files matched --corpus-glob={args.corpus_glob!r}."
        )

    benchmark = load_dataset(str(args.benchmark_repo), split=str(args.benchmark_split))
    ordered_query_ids: list[str] = []
    seen_query_ids: set[str] = set()
    qrels_query_ids: list[str] = []
    qrels_doc_ids: list[str] = []
    qrels_scores: list[float] = []
    empty_label_rows: int = 0

    row: dict[str, Any]
    for row in benchmark:
        query_id: str = str(row["question_id"])
        label_ids: list[Any] | None = row.get("label_id")
        if not label_ids:
            empty_label_rows += 1
            continue
        if query_id not in seen_query_ids:
            seen_query_ids.add(query_id)
            ordered_query_ids.append(query_id)
        qrels_query_ids.append(query_id)
        qrels_doc_ids.append(str(label_ids[0]))
        qrels_scores.append(1.0)

    query_id_set: set[str] = set(ordered_query_ids)
    corpus_dataset = ds.dataset(corpus_paths, format="parquet")
    query_table = corpus_dataset.to_table(
        columns=["doc_id", "title", "abstract", "claims"],
        filter=pc.field("doc_id").isin(sorted(query_id_set)),
    )

    query_text_by_id: dict[str, str] = {}
    for doc_id, title, abstract, claims in zip(
        query_table.column("doc_id").to_pylist(),
        query_table.column("title").to_pylist(),
        query_table.column("abstract").to_pylist(),
        query_table.column("claims").to_pylist(),
    ):
        query_text_by_id[str(doc_id)] = _compose_text(
            {
                "title": title,
                "abstract": abstract,
                "claims": claims,
            }
        )

    queries_query_ids: list[str] = []
    queries_texts: list[str] = []
    missing_query_ids: list[str] = []
    query_id: str
    for query_id in ordered_query_ids:
        query_text: str | None = query_text_by_id.get(query_id)
        if not query_text:
            missing_query_ids.append(query_id)
            continue
        queries_query_ids.append(query_id)
        queries_texts.append(query_text)

    allowed_query_ids: set[str] = set(queries_query_ids)
    filtered_qrels_query_ids: list[str] = []
    filtered_qrels_doc_ids: list[str] = []
    filtered_qrels_scores: list[float] = []
    seen_qrel_pairs: set[tuple[str, str]] = set()
    for query_id, doc_id, score in zip(qrels_query_ids, qrels_doc_ids, qrels_scores):
        if query_id not in allowed_query_ids:
            continue
        qrel_pair: tuple[str, str] = (query_id, doc_id)
        if qrel_pair in seen_qrel_pairs:
            continue
        seen_qrel_pairs.add(qrel_pair)
        filtered_qrels_query_ids.append(query_id)
        filtered_qrels_doc_ids.append(doc_id)
        filtered_qrels_scores.append(score)

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_path = output_dir / "queries.parquet"
    qrels_path = output_dir / "qrels.parquet"
    metadata_path = output_dir / "metadata.json"

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

    metadata = {
        "benchmark_repo": str(args.benchmark_repo),
        "benchmark_split": str(args.benchmark_split),
        "query_count": len(queries_query_ids),
        "qrels_count": len(filtered_qrels_query_ids),
        "empty_label_rows": empty_label_rows,
        "missing_query_count": len(missing_query_ids),
        "missing_query_ids_sample": missing_query_ids[:20],
        "queries_path": str(queries_path),
        "qrels_path": str(qrels_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
