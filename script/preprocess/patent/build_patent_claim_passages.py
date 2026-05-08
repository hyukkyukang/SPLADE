"""Build a DPR-style claim-passage corpus from patent parquet or HF datasets."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from datasets import load_dataset

from src.data.patent_passages import build_patent_claim_passages
from src.utils.huggingface import resolve_hf_token
from src.utils.normalize import normalize_optional_str
from src.utils.script_setup import configure_script_environment

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=False,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)

_PASSAGE_SCHEMA: pa.Schema = pa.schema(
    [
        ("passage_id", pa.string()),
        ("parent_doc_id", pa.string()),
        ("source_doc_id", pa.string()),
        ("chunk_type", pa.string()),
        ("chunk_idx", pa.int32()),
        ("text", pa.string()),
    ]
)


def _iter_local_corpus_rows(
    *,
    corpus_glob: str,
    columns: Sequence[str],
    batch_size: int,
) -> Iterator[Mapping[str, Any]]:
    corpus_paths: list[str] = sorted(glob(str(corpus_glob)))
    if not corpus_paths:
        raise FileNotFoundError(
            f"No corpus parquet files matched --corpus-glob={corpus_glob!r}."
        )
    dataset = ds.dataset(corpus_paths, format="parquet")
    scanner = dataset.scanner(columns=list(columns), batch_size=max(1, int(batch_size)))
    record_batch: pa.RecordBatch
    for record_batch in scanner.to_batches():
        row: Mapping[str, Any]
        for row in record_batch.to_pylist():
            yield row


def _iter_hf_corpus_rows(
    *,
    corpus_repo: str,
    corpus_subset: str | None,
    corpus_split: str,
    corpus_cache_dir: str | None,
    columns: Sequence[str],
) -> Iterator[Mapping[str, Any]]:
    token: str | None = resolve_hf_token()
    dataset = load_dataset(
        corpus_repo,
        name=corpus_subset,
        split=corpus_split,
        cache_dir=corpus_cache_dir,
        token=token,
        streaming=True,
    )
    row: Mapping[str, Any]
    for row in dataset:
        yield {column: row.get(column) for column in columns}


def _write_passage_batch(
    writer: pq.ParquetWriter,
    batch_rows: list[dict[str, Any]],
) -> None:
    if not batch_rows:
        return
    writer.write_table(pa.Table.from_pylist(batch_rows, schema=_PASSAGE_SCHEMA))
    batch_rows.clear()


def build_passage_corpus(
    *,
    row_iter: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    doc_id_column: str = "doc_id",
    title_column: str = "title",
    claims_column: str = "claims",
    group_id_column: str | None = None,
    max_claim_chunk_words: int = 300,
    max_title_prefixed_words: int = 100,
    write_batch_size: int = 50000,
) -> dict[str, Any]:
    output_file = Path(str(output_path))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_path: Path = output_file.with_suffix(".metadata.json")

    document_count: int = 0
    passage_count: int = 0
    skipped_empty_claim_docs: int = 0
    batch_rows: list[dict[str, Any]] = []

    with pq.ParquetWriter(output_file, _PASSAGE_SCHEMA) as writer:
        row: Mapping[str, Any]
        for row in row_iter:
            document_count += 1
            passages = build_patent_claim_passages(
                row,
                doc_id_key=doc_id_column,
                title_key=title_column,
                claims_key=claims_column,
                group_id_key=group_id_column,
                max_claim_chunk_words=max_claim_chunk_words,
                max_title_prefixed_words=max_title_prefixed_words,
            )
            if not passages:
                skipped_empty_claim_docs += 1
                continue
            batch_rows.extend(passages)
            passage_count += len(passages)
            if len(batch_rows) >= max(1, int(write_batch_size)):
                _write_passage_batch(writer, batch_rows)
        _write_passage_batch(writer, batch_rows)

    metadata: dict[str, Any] = {
        "output_path": str(output_file),
        "document_count": document_count,
        "passage_count": passage_count,
        "skipped_empty_claim_docs": skipped_empty_claim_docs,
        "doc_id_column": doc_id_column,
        "group_id_column": group_id_column,
        "title_column": title_column,
        "claims_column": claims_column,
        "max_claim_chunk_words": max_claim_chunk_words,
        "max_title_prefixed_words": max_title_prefixed_words,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-repo",
        default=None,
        help="Optional HF corpus dataset repo. If unset, --corpus-glob is used.",
    )
    parser.add_argument(
        "--corpus-subset",
        default=None,
        help="Optional HF corpus dataset config/subset.",
    )
    parser.add_argument(
        "--corpus-split",
        default="train",
        help="HF corpus split when --corpus-repo is set.",
    )
    parser.add_argument(
        "--corpus-cache-dir",
        default=None,
        help="Optional HF cache dir for the corpus dataset.",
    )
    parser.add_argument(
        "--corpus-glob",
        default=".cache/hf/patent-us-corpus-small/data/*.parquet",
        help="Local parquet glob used when --corpus-repo is unset.",
    )
    parser.add_argument("--doc-id-column", default="doc_id")
    parser.add_argument("--group-id-column", default=None)
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--claims-column", default="claims")
    parser.add_argument("--scan-batch-size", type=int, default=4096)
    parser.add_argument("--write-batch-size", type=int, default=50000)
    parser.add_argument("--max-claim-chunk-words", type=int, default=300)
    parser.add_argument("--max-title-prefixed-words", type=int, default=100)
    parser.add_argument(
        "--output-path",
        default="data/corpus/patent_us_claim_passages_small/passages.parquet",
        help="Output parquet path for the generated claim-passage corpus.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    columns: list[str] = [str(args.doc_id_column), str(args.title_column), str(args.claims_column)]
    group_id_column: str | None = normalize_optional_str(args.group_id_column)
    if group_id_column is not None and group_id_column not in columns:
        columns.append(group_id_column)

    corpus_repo: str | None = normalize_optional_str(args.corpus_repo)
    if corpus_repo is not None:
        row_iter = _iter_hf_corpus_rows(
            corpus_repo=corpus_repo,
            corpus_subset=normalize_optional_str(args.corpus_subset),
            corpus_split=str(args.corpus_split),
            corpus_cache_dir=normalize_optional_str(args.corpus_cache_dir),
            columns=columns,
        )
    else:
        row_iter = _iter_local_corpus_rows(
            corpus_glob=str(args.corpus_glob),
            columns=columns,
            batch_size=int(args.scan_batch_size),
        )

    metadata = build_passage_corpus(
        row_iter=row_iter,
        output_path=str(args.output_path),
        doc_id_column=str(args.doc_id_column),
        title_column=str(args.title_column),
        claims_column=str(args.claims_column),
        group_id_column=group_id_column,
        max_claim_chunk_words=int(args.max_claim_chunk_words),
        max_title_prefixed_words=int(args.max_title_prefixed_words),
        write_batch_size=int(args.write_batch_size),
    )
    if corpus_repo is not None:
        metadata["corpus_repo"] = corpus_repo
        metadata["corpus_subset"] = normalize_optional_str(args.corpus_subset)
        metadata["corpus_split"] = str(args.corpus_split)
    else:
        metadata["corpus_glob"] = str(args.corpus_glob)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
