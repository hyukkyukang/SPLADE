"""Convert a 4-column patent claim passage TSV into a SQLite lookup database."""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import time
from pathlib import Path


DEFAULT_TABLE_NAME: str = "claim_passages"
DEFAULT_BATCH_SIZE: int = 10_000
DEFAULT_LOG_EVERY: int = 100_000


def configure_csv_field_limit() -> None:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            return
        except OverflowError:
            max_int //= 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a TSV with columns "
            "(doc_id, text, appl_id, passage_id) into a SQLite database."
        )
    )
    parser.add_argument("--input-tsv", required=True, help="Path to the source TSV.")
    parser.add_argument(
        "--output-sqlite", required=True, help="Path to the output SQLite database."
    )
    parser.add_argument(
        "--table-name",
        default=DEFAULT_TABLE_NAME,
        help=f"Destination SQLite table name. Default: {DEFAULT_TABLE_NAME}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per INSERT batch. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help=f"Progress log interval in rows. Default: {DEFAULT_LOG_EVERY}.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the output database if it already exists.",
    )
    return parser.parse_args()


def quote_identifier(identifier: str) -> str:
    if not identifier:
        raise ValueError("SQLite identifier must be non-empty.")
    return '"' + identifier.replace('"', '""') + '"'


def configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA cache_size=-200000")


def create_schema(conn: sqlite3.Connection, *, table_name: str) -> None:
    quoted_table = quote_identifier(table_name)
    conn.execute(
        f"""
        CREATE TABLE {quoted_table} (
            passage_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            appl_id TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        """
        CREATE TABLE import_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )


def flush_rows(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    rows: list[tuple[str, str, str, str]],
) -> None:
    if not rows:
        return
    quoted_table = quote_identifier(table_name)
    conn.executemany(
        f"""
        INSERT INTO {quoted_table} (passage_id, doc_id, text, appl_id)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    rows.clear()


def insert_metadata(
    conn: sqlite3.Connection,
    *,
    input_tsv: Path,
    row_count: int,
    malformed_row_count: int,
    table_name: str,
    started_at_s: float,
) -> None:
    elapsed_s = time.time() - started_at_s
    values = [
        ("input_tsv", str(input_tsv)),
        ("table_name", str(table_name)),
        ("row_count", str(row_count)),
        ("malformed_row_count", str(malformed_row_count)),
        ("elapsed_seconds", f"{elapsed_s:.6f}"),
        ("imported_at_epoch_s", str(time.time())),
    ]
    conn.executemany(
        "INSERT INTO import_metadata (key, value) VALUES (?, ?)",
        values,
    )
    conn.commit()


def create_indexes(conn: sqlite3.Connection, *, table_name: str) -> None:
    quoted_table = quote_identifier(table_name)
    conn.execute(
        f"CREATE INDEX idx_{table_name}_doc_id ON {quoted_table} (doc_id)"
    )
    conn.execute(
        f"CREATE INDEX idx_{table_name}_appl_id ON {quoted_table} (appl_id)"
    )
    conn.execute(
        f"CREATE INDEX idx_{table_name}_doc_id_passage_id "
        f"ON {quoted_table} (doc_id, passage_id)"
    )
    conn.commit()


def import_tsv(
    *,
    input_tsv: Path,
    output_sqlite: Path,
    table_name: str,
    batch_size: int,
    log_every: int,
) -> tuple[int, int]:
    tmp_path = output_sqlite.with_suffix(output_sqlite.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    started_at_s = time.time()
    row_count = 0
    malformed_row_count = 0
    pending_rows: list[tuple[str, str, str, str]] = []

    conn = sqlite3.connect(str(tmp_path))
    try:
        configure_connection(conn)
        create_schema(conn, table_name=table_name)

        with input_tsv.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) != 4:
                    malformed_row_count += 1
                    continue
                doc_id, text, appl_id, passage_id = row
                pending_rows.append((passage_id, doc_id, text, appl_id))
                row_count += 1
                if len(pending_rows) >= batch_size:
                    flush_rows(conn, table_name=table_name, rows=pending_rows)
                if log_every > 0 and row_count % log_every == 0:
                    elapsed_s = time.time() - started_at_s
                    rate = row_count / elapsed_s if elapsed_s > 0 else 0.0
                    print(
                        (
                            f"[import] rows={row_count:,} "
                            f"malformed={malformed_row_count:,} "
                            f"elapsed={elapsed_s:,.1f}s "
                            f"rate={rate:,.1f} rows/s"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
            flush_rows(conn, table_name=table_name, rows=pending_rows)

        insert_metadata(
            conn,
            input_tsv=input_tsv,
            row_count=row_count,
            malformed_row_count=malformed_row_count,
            table_name=table_name,
            started_at_s=started_at_s,
        )
        create_indexes(conn, table_name=table_name)
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()

    os.replace(tmp_path, output_sqlite)
    return row_count, malformed_row_count


def main() -> None:
    args = parse_args()
    configure_csv_field_limit()

    input_tsv = Path(str(args.input_tsv)).expanduser().resolve()
    output_sqlite = Path(str(args.output_sqlite)).expanduser().resolve()
    table_name = str(args.table_name).strip()
    batch_size = max(1, int(args.batch_size))
    log_every = max(0, int(args.log_every))

    if not input_tsv.exists():
        raise FileNotFoundError(f"Input TSV does not exist: {input_tsv}")

    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if output_sqlite.exists():
        if not bool(args.replace):
            raise FileExistsError(
                f"Output SQLite already exists: {output_sqlite}. "
                "Pass --replace to overwrite it."
            )
        output_sqlite.unlink()

    row_count, malformed_row_count = import_tsv(
        input_tsv=input_tsv,
        output_sqlite=output_sqlite,
        table_name=table_name,
        batch_size=batch_size,
        log_every=log_every,
    )
    print(
        (
            f"[done] sqlite={output_sqlite} table={table_name} "
            f"rows={row_count:,} malformed={malformed_row_count:,}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
