import argparse
import json
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from opensearchpy import OpenSearch
from tqdm import tqdm


TARGET_COLUMNS = [
    "doc_id",
    "title",
    "abstract",
    "claims",
    "description",
    "application_id",
]


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def build_opensearch_client(host: str, port: int, timeout: int) -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": host, "port": str(port)}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=timeout,
        max_retries=10,
        retry_on_timeout=True,
    )


def iter_scrolled_hits(
    client: OpenSearch,
    index: str,
    query: dict,
    source_fields: list[str],
    batch_size: int,
    scroll_time: str,
    slice_spec: dict | None = None,
) -> Iterable[dict]:
    search_body: dict = {"track_total_hits": True, "query": query}
    if slice_spec is not None:
        search_body["slice"] = slice_spec

    response = client.search(
        index=index,
        scroll=scroll_time,
        size=batch_size,
        _source=source_fields,
        body=search_body,
    )
    scroll_id = response["_scroll_id"]
    try:
        hits = response["hits"]["hits"]
        while hits:
            for hit in hits:
                yield hit
            response = client.scroll(scroll_id=scroll_id, scroll=scroll_time)
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]
    finally:
        try:
            client.clear_scroll(body={"scroll_id": [scroll_id]})
        except Exception:
            pass


class ParquetShardWriter:
    def __init__(
        self,
        output_dir: Path,
        output_prefix: str,
        schema: pa.Schema,
        shard_size: int,
        compression: str | None,
    ) -> None:
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.schema = schema
        self.shard_size = shard_size
        self.compression = compression

        self.writer: pq.ParquetWriter | None = None
        self.shard_index = -1
        self.rows_in_shard = 0
        self.total_rows = 0
        self.shard_paths: list[str] = []

    def _open_next_shard(self) -> None:
        self.shard_index += 1
        self.rows_in_shard = 0
        shard_path = self.output_dir / f"{self.output_prefix}-{self.shard_index:05d}.parquet"
        self.shard_paths.append(shard_path.as_posix())
        self.writer = pq.ParquetWriter(
            where=shard_path.as_posix(),
            schema=self.schema,
            compression=self.compression,
            use_dictionary=True,
        )

    def write_rows(self, rows: list[dict[str, str]]) -> None:
        start = 0
        while start < len(rows):
            if self.writer is None or self.rows_in_shard >= self.shard_size:
                self.close()
                self._open_next_shard()

            room = self.shard_size - self.rows_in_shard
            take = min(room, len(rows) - start)
            chunk = rows[start : start + take]
            table = pa.Table.from_pylist(chunk, schema=self.schema)
            self.writer.write_table(table)
            self.rows_in_shard += take
            self.total_rows += take
            start += take

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export patent documents (title/abstract/claims/description) for HF datasets."
    )
    parser.add_argument("--host", default="10.4.43.27", help="OpenSearch host")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port")
    parser.add_argument("--index", default="patent_search", help="OpenSearch index/alias")
    parser.add_argument("--org-code", default="US", help="Patent org code (e.g., US, KR)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Scroll batch size")
    parser.add_argument("--scroll-time", default="10m", help="Scroll keep-alive")
    parser.add_argument("--timeout", type=int, default=120, help="OpenSearch timeout seconds")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional max docs to export")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/patent/hf_us_patent_docs"),
        help="Output directory for parquet shards and manifest",
    )
    parser.add_argument(
        "--output-prefix",
        default="patent_us_docs",
        help="Output parquet shard prefix",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Optional stats path (default: <output-dir>/<prefix>_stats.json)",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=20000,
        help="Rows accumulated in memory before parquet write",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=200000,
        help="Rows per parquet shard file",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "none"],
        help="Parquet compression codec",
    )
    parser.add_argument(
        "--normalize-whitespace",
        action="store_true",
        help="Collapse newlines/multiple spaces in text fields",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=None,
        help="Optional sliced-scroll parallelism: total slice count",
    )
    parser.add_argument(
        "--slice-id",
        type=int,
        default=None,
        help="Optional sliced-scroll parallelism: this worker's slice id (0-indexed)",
    )
    args = parser.parse_args()

    if (args.num_slices is None) != (args.slice_id is None):
        raise ValueError("`--num-slices` and `--slice-id` must be provided together.")
    if args.num_slices is not None and not (0 <= args.slice_id < args.num_slices):
        raise ValueError("`--slice-id` must satisfy 0 <= slice-id < num-slices.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_spec = (
        None
        if args.num_slices is None
        else {"id": int(args.slice_id), "max": int(args.num_slices)}
    )

    suffix = (
        ""
        if slice_spec is None
        else f"_slice{int(args.slice_id):02d}of{int(args.num_slices):02d}"
    )
    output_prefix = f"{args.output_prefix}{suffix}"

    stats_path = (
        args.stats_json
        if args.stats_json is not None
        else output_dir / f"{output_prefix}_stats.json"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    title_field = f"{args.org_code}_title"
    abstract_field = f"{args.org_code}_abstract"
    claims_field = f"{args.org_code}_claims"
    description_field = f"{args.org_code}_description"
    application_field = "appl_id"
    doc_id_field = "doc_id"

    source_fields = [
        doc_id_field,
        application_field,
        title_field,
        abstract_field,
        claims_field,
        description_field,
        "org_code",
    ]

    filters: list[dict] = [{"term": {"org_code": args.org_code}}]
    query = {"bool": {"filter": filters}}

    client = build_opensearch_client(host=args.host, port=args.port, timeout=args.timeout)

    total_docs = (
        None
        if slice_spec is not None
        else client.count(index=args.index, body={"query": query})["count"]
    )

    if args.max_docs is not None:
        target_docs = args.max_docs if total_docs is None else min(total_docs, args.max_docs)
    else:
        target_docs = total_docs

    schema = pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("abstract", pa.string()),
            pa.field("claims", pa.string()),
            pa.field("description", pa.string()),
            pa.field("application_id", pa.string()),
        ]
    )
    compression = None if args.compression == "none" else args.compression
    writer = ParquetShardWriter(
        output_dir=output_dir,
        output_prefix=output_prefix,
        schema=schema,
        shard_size=args.shard_size,
        compression=compression,
    )

    processed_docs = 0
    emitted_docs = 0
    missing_value_counts = {col: 0 for col in TARGET_COLUMNS}
    source_missing_counts = {
        doc_id_field: 0,
        application_field: 0,
        title_field: 0,
        abstract_field: 0,
        claims_field: 0,
        description_field: 0,
    }
    rows_buffer: list[dict[str, str]] = []

    pbar = None if args.no_progress else tqdm(total=target_docs, desc="Exporting docs")
    try:
        for hit in iter_scrolled_hits(
            client=client,
            index=args.index,
            query=query,
            source_fields=source_fields,
            batch_size=args.batch_size,
            scroll_time=args.scroll_time,
            slice_spec=slice_spec,
        ):
            if args.max_docs is not None and processed_docs >= args.max_docs:
                break

            processed_docs += 1
            if pbar is not None:
                pbar.update(1)

            source = hit.get("_source", {})
            for source_key in source_missing_counts:
                if source_key not in source:
                    source_missing_counts[source_key] += 1

            row = {
                "doc_id": as_text(source.get(doc_id_field)) or as_text(hit.get("_id", "")),
                "title": as_text(source.get(title_field)),
                "abstract": as_text(source.get(abstract_field)),
                "claims": as_text(source.get(claims_field)),
                "description": as_text(source.get(description_field)),
                "application_id": as_text(source.get(application_field)),
            }

            if args.normalize_whitespace:
                row["title"] = normalize_whitespace(row["title"])
                row["abstract"] = normalize_whitespace(row["abstract"])
                row["claims"] = normalize_whitespace(row["claims"])
                row["description"] = normalize_whitespace(row["description"])

            for col, val in row.items():
                if val == "":
                    missing_value_counts[col] += 1

            rows_buffer.append(row)
            if len(rows_buffer) >= args.write_batch_size:
                writer.write_rows(rows_buffer)
                emitted_docs += len(rows_buffer)
                rows_buffer = []

        if rows_buffer:
            writer.write_rows(rows_buffer)
            emitted_docs += len(rows_buffer)
    finally:
        writer.close()
        if pbar is not None:
            pbar.close()

    stats = {
        "source_index": args.index,
        "filters": {"org_code": args.org_code},
        "slice": slice_spec,
        "source_fields_used": {
            "doc_id": doc_id_field,
            "title": title_field,
            "abstract": abstract_field,
            "claims": claims_field,
            "description": description_field,
            "application_id": application_field,
        },
        "target_columns": TARGET_COLUMNS,
        "total_matching_docs": total_docs,
        "processed_docs": processed_docs,
        "emitted_docs": emitted_docs,
        "missing_value_counts": missing_value_counts,
        "source_missing_field_counts": source_missing_counts,
        "parquet": {
            "compression": args.compression,
            "shard_size": args.shard_size,
            "shard_count": len(writer.shard_paths),
            "shard_paths": writer.shard_paths,
        },
    }
    with stats_path.open("w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    matched_text = f"{total_docs:,}" if total_docs is not None else "N/A (slice worker)"
    print(f"Matched docs: {matched_text}")
    print(f"Processed docs: {processed_docs:,}")
    print(f"Emitted docs: {emitted_docs:,}")
    print(f"Parquet shards: {len(writer.shard_paths):,}")
    print(f"Output dir: {output_dir.resolve().as_posix()}")
    print(f"Stats manifest: {stats_path.resolve().as_posix()}")


if __name__ == "__main__":
    main()
