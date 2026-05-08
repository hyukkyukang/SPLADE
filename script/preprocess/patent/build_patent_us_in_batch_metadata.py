"""Build stage-1 patent in-batch metadata from usc102103_train.json.

This script resolves raw DPR-style positive contexts against the local
`patent-us-corpus` parquet export and emits a compact parquet file with:

- `query_id`
- `query_text`
- `pos_doc_ids`
- `source_positive_count`
- `matched_positive_count`

Only positives that can be matched to a unique corpus `doc_id` are retained.
Rows without any matched positives are dropped by default.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from glob import glob
import hashlib
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


OUTPUT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("query_id", pa.string()),
        pa.field("query_text", pa.string()),
        pa.field("pos_doc_ids", pa.list_(pa.string())),
        pa.field("source_positive_count", pa.int32()),
        pa.field("matched_positive_count", pa.int32()),
    ]
)


@dataclass(slots=True)
class BuildStats:
    raw_rows: int = 0
    emitted_rows: int = 0
    dropped_empty_query_rows: int = 0
    dropped_rows_without_matches: int = 0
    source_positive_contexts: int = 0
    matched_positive_contexts: int = 0
    unresolved_positive_contexts: int = 0
    ambiguous_positive_contexts: int = 0
    distinct_target_keys: int = 0
    matched_target_keys: int = 0
    ambiguous_target_keys: int = 0


def _sha1_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}"


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def _normalize_match_component(text: str) -> str:
    return normalize_whitespace(text).upper()


def build_title_abstract_key(title: str, abstract: str) -> str | None:
    resolved_title: str = _normalize_match_component(title)
    resolved_abstract: str = _normalize_match_component(abstract)
    if not resolved_title or not resolved_abstract:
        return None
    return f"{resolved_title}\n{resolved_abstract}"


def build_title_abstract_hash(title: str, abstract: str) -> str | None:
    key: str | None = build_title_abstract_key(title, abstract)
    if key is None:
        return None
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def extract_ctx_title_abstract(ctx: Mapping[str, Any]) -> tuple[str, str]:
    title: str = normalize_whitespace(str(ctx.get("title", "") or ""))
    text: str = normalize_whitespace(str(ctx.get("text", "") or ""))
    if text and "[SEP]" in text:
        left, right = text.split("[SEP]", 1)
        sep_title: str = normalize_whitespace(left)
        sep_abstract: str = normalize_whitespace(right)
        if title:
            if _normalize_match_component(title) == _normalize_match_component(sep_title):
                return title, sep_abstract
            return title, text
        return sep_title, sep_abstract
    if title:
        return title, text
    return "", text


def extract_ctx_match_key(ctx: Mapping[str, Any]) -> str | None:
    title, abstract = extract_ctx_title_abstract(ctx)
    return build_title_abstract_key(title, abstract)


def extract_ctx_match_hash(ctx: Mapping[str, Any]) -> str | None:
    title, abstract = extract_ctx_title_abstract(ctx)
    return build_title_abstract_hash(title, abstract)


def iter_json_array(path: Path, *, chunk_size: int = 1 << 20) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer: str = ""
        array_started: bool = False
        reached_eof: bool = False
        while True:
            if not reached_eof and len(buffer) < chunk_size:
                chunk: str = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    reached_eof = True
            buffer = buffer.lstrip()
            if not array_started:
                if not buffer:
                    if reached_eof:
                        raise ValueError(f"Expected JSON array in {path.as_posix()}")
                    continue
                if not buffer.startswith("["):
                    raise ValueError(f"Expected JSON array in {path.as_posix()}")
                buffer = buffer[1:]
                array_started = True
                continue
            buffer = buffer.lstrip()
            if not buffer:
                if reached_eof:
                    raise ValueError(f"Unexpected EOF in {path.as_posix()}")
                continue
            if buffer.startswith("]"):
                return
            try:
                value, offset = decoder.raw_decode(buffer)
            except JSONDecodeError:
                if reached_eof:
                    raise
                chunk = handle.read(chunk_size)
                if not chunk:
                    reached_eof = True
                else:
                    buffer += chunk
                continue
            yield value
            buffer = buffer[offset:].lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                return
            if not buffer and reached_eof:
                raise ValueError(f"Unexpected EOF in {path.as_posix()}")


def collect_target_match_hashes(raw_json_path: Path) -> set[str]:
    target_hashes: set[str] = set()
    row: Any
    for row in iter_json_array(raw_json_path):
        if not isinstance(row, Mapping):
            continue
        positive_ctxs: Any = row.get("positive_ctxs") or []
        if not isinstance(positive_ctxs, list):
            continue
        ctx: Any
        for ctx in positive_ctxs:
            if not isinstance(ctx, Mapping):
                continue
            key_hash: str | None = extract_ctx_match_hash(ctx)
            if key_hash is not None:
                target_hashes.add(key_hash)
    return target_hashes


def build_corpus_doc_id_lookup(
    corpus_paths: Sequence[Path],
    target_hashes: set[str],
    *,
    batch_size: int = 8192,
) -> tuple[dict[str, str], set[str]]:
    if not target_hashes:
        return {}, set()
    corpus_dataset = ds.dataset([path.as_posix() for path in corpus_paths], format="parquet")
    scanner = corpus_dataset.scanner(
        columns=["doc_id", "title", "abstract"],
        batch_size=int(batch_size),
    )
    matched_doc_ids: dict[str, str] = {}
    ambiguous_keys: set[str] = set()
    record_batch: pa.RecordBatch
    for record_batch in scanner.to_batches():
        doc_ids: list[Any] = record_batch.column("doc_id").to_pylist()
        titles: list[Any] = record_batch.column("title").to_pylist()
        abstracts: list[Any] = record_batch.column("abstract").to_pylist()
        doc_id: Any
        title: Any
        abstract: Any
        for doc_id, title, abstract in zip(doc_ids, titles, abstracts):
            key_hash: str | None = build_title_abstract_hash(
                "" if title is None else str(title),
                "" if abstract is None else str(abstract),
            )
            if key_hash is None or key_hash not in target_hashes:
                continue
            doc_id_text: str = normalize_whitespace(str(doc_id))
            if not doc_id_text:
                continue
            existing_doc_id: str | None = matched_doc_ids.get(key_hash)
            if existing_doc_id is None:
                matched_doc_ids[key_hash] = doc_id_text
                continue
            if existing_doc_id != doc_id_text:
                ambiguous_keys.add(key_hash)
    resolved_lookup: dict[str, str] = {
        key: doc_id
        for key, doc_id in matched_doc_ids.items()
        if key not in ambiguous_keys
    }
    return resolved_lookup, ambiguous_keys


def _write_rows(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    if writer is None:
        writer = pq.ParquetWriter(output_path.as_posix(), OUTPUT_SCHEMA)
    writer.write_table(table)
    return writer


def write_patent_us_in_batch_parquet(
    *,
    raw_json_path: Path,
    corpus_paths: Sequence[Path],
    output_path: Path,
    keep_rows_without_matches: bool = False,
    write_batch_size: int = 4096,
    lookup_batch_size: int = 8192,
) -> BuildStats:
    target_hashes: set[str] = collect_target_match_hashes(raw_json_path)
    doc_id_lookup, ambiguous_keys = build_corpus_doc_id_lookup(
        corpus_paths,
        target_hashes,
        batch_size=lookup_batch_size,
    )
    stats = BuildStats(
        distinct_target_keys=len(target_hashes),
        matched_target_keys=len(doc_id_lookup),
        ambiguous_target_keys=len(ambiguous_keys),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_buffer: list[dict[str, Any]] = []
    row: Any
    for row in iter_json_array(raw_json_path):
        if not isinstance(row, Mapping):
            continue
        stats.raw_rows += 1
        query_text: str = normalize_whitespace(str(row.get("question", "") or ""))
        if not query_text:
            stats.dropped_empty_query_rows += 1
            continue
        positive_ctxs: Any = row.get("positive_ctxs") or []
        if not isinstance(positive_ctxs, list):
            positive_ctxs = []
        stats.source_positive_contexts += len(positive_ctxs)
        matched_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        ctx: Any
        for ctx in positive_ctxs:
            if not isinstance(ctx, Mapping):
                stats.unresolved_positive_contexts += 1
                continue
            key_hash: str | None = extract_ctx_match_hash(ctx)
            if key_hash is None:
                stats.unresolved_positive_contexts += 1
                continue
            if key_hash in ambiguous_keys:
                stats.ambiguous_positive_contexts += 1
                continue
            doc_id: str | None = doc_id_lookup.get(key_hash)
            if doc_id is None:
                stats.unresolved_positive_contexts += 1
                continue
            stats.matched_positive_contexts += 1
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            matched_doc_ids.append(doc_id)
        if not matched_doc_ids and not keep_rows_without_matches:
            stats.dropped_rows_without_matches += 1
            continue
        rows_buffer.append(
            {
                "query_id": _sha1_id("q_", query_text),
                "query_text": query_text,
                "pos_doc_ids": matched_doc_ids,
                "source_positive_count": len(positive_ctxs),
                "matched_positive_count": len(matched_doc_ids),
            }
        )
        stats.emitted_rows += 1
        if len(rows_buffer) >= int(write_batch_size):
            writer = _write_rows(output_path, rows_buffer, writer=writer)
            rows_buffer.clear()
    if rows_buffer:
        writer = _write_rows(output_path, rows_buffer, writer=writer)
        rows_buffer.clear()
    if writer is None:
        pq.write_table(pa.Table.from_pylist([], schema=OUTPUT_SCHEMA), output_path.as_posix())
    else:
        writer.close()
    return stats


def resolve_corpus_paths(corpus_glob: str) -> list[Path]:
    corpus_paths: list[Path] = [Path(path) for path in sorted(glob(corpus_glob))]
    if not corpus_paths:
        raise FileNotFoundError(f"No parquet files matched --corpus-glob={corpus_glob!r}")
    return corpus_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve usc102103_train.json positives to patent-us-corpus doc ids.",
    )
    parser.add_argument(
        "--raw-json-path",
        type=Path,
        default=Path("data/patent/train/usc102103_train.json"),
        help="Path to the raw DPR-style patent train JSON array.",
    )
    parser.add_argument(
        "--corpus-glob",
        type=str,
        default=".cache/hf/patent-us-corpus/patent_us_docs_slice*.parquet",
        help="Glob for local patent-us-corpus parquet shards.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/patent/train/usc102103_in_batch_metadata.parquet"),
        help="Output parquet path for stage-1 in-batch training metadata.",
    )
    parser.add_argument(
        "--keep-rows-without-matches",
        action="store_true",
        help="Keep rows even when no positive context could be resolved.",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=4096,
        help="Rows buffered before each parquet write.",
    )
    parser.add_argument(
        "--lookup-batch-size",
        type=int,
        default=8192,
        help="Corpus rows processed per pyarrow record batch while building the lookup.",
    )
    args = parser.parse_args()

    if not args.raw_json_path.exists():
        raise FileNotFoundError(f"Missing raw JSON path: {args.raw_json_path.as_posix()}")
    corpus_paths: list[Path] = resolve_corpus_paths(str(args.corpus_glob))
    stats = write_patent_us_in_batch_parquet(
        raw_json_path=args.raw_json_path,
        corpus_paths=corpus_paths,
        output_path=args.output_path,
        keep_rows_without_matches=bool(args.keep_rows_without_matches),
        write_batch_size=max(int(args.write_batch_size), 1),
        lookup_batch_size=max(int(args.lookup_batch_size), 1),
    )
    print(json.dumps(asdict(stats), indent=2, sort_keys=True))
    print(f"Wrote parquet metadata to {args.output_path.as_posix()}")


if __name__ == "__main__":
    main()
