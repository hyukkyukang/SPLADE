import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

MANIFEST_FILENAME: str = "manifest.json"
LOCK_FILENAME: str = "build.lock"
DONE_FILENAME: str = "build.done"
CACHE_VERSION: int = 2
INDEX_FILENAME_SUFFIX: str = ".index.sqlite"
ROW_INDEX_FILENAME_SUFFIX: str = ".row_index.npy"
INPUT_IDS_SIDECAR_SUFFIX: str = ".input_ids.npy"
ATTENTION_MASK_SIDECAR_SUFFIX: str = ".attention_mask.npy"
STORAGE_FORMAT_HYBRID: str = "hybrid"
STORAGE_FORMAT_SIDECAR_ONLY: str = "sidecar_only"


def make_id_key(identifier: str) -> str:
    return f"id:{identifier}"


def make_text_key(text: str) -> str:
    digest: str = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"text:{digest}"


def build_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any] = {"cache_version": CACHE_VERSION}
    manifest.update(payload)
    return manifest


def load_manifest(cache_dir: Path) -> dict[str, Any] | None:
    manifest_path: Path = cache_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        data: Any = json.load(manifest_file)
    if not isinstance(data, dict):
        return None
    return dict(data)


def write_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path: Path = cache_dir / MANIFEST_FILENAME
    tmp_path: Path = cache_dir / f"{MANIFEST_FILENAME}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=True, sort_keys=True, indent=2)
    tmp_path.replace(manifest_path)


def manifests_compatible(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    key: str
    for key in expected:
        if existing.get(key) != expected.get(key):
            return False
    return True


def resolve_lock_path(cache_dir: Path) -> Path:
    return cache_dir / LOCK_FILENAME


def resolve_done_path(cache_dir: Path) -> Path:
    return cache_dir / DONE_FILENAME


def _read_lock_pid(lock_path: Path) -> int | None:
    """Read a lock owner pid if available."""
    try:
        raw_pid: str = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw_pid:
        return None
    try:
        return int(raw_pid)
    except ValueError:
        return None


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Another user's process is assumed alive.
        return True
    return True


def _clear_stale_lock(lock_path: Path) -> bool:
    """Best-effort stale lock recovery."""
    lock_pid: int | None = _read_lock_pid(lock_path)
    if lock_pid is not None and _pid_exists(lock_pid):
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def acquire_build_lock(lock_path: Path) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    attempt: int
    for attempt in range(2):
        try:
            fd: int = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Another builder is active. If lock owner is dead, clear stale lock
            # once and retry lock acquisition.
            if attempt == 0 and _clear_stale_lock(lock_path):
                continue
            return False
        try:
            os.write(fd, str(os.getpid()).encode("utf-8"))
        finally:
            os.close(fd)
        return True
    return False


def release_build_lock(lock_path: Path) -> None:
    if lock_path.exists():
        lock_path.unlink()


def wait_for_done(done_path: Path, *, timeout_s: int = 3600, poll_s: float = 2.0) -> None:
    start_time: float = time.time()
    while not done_path.exists():
        if (time.time() - start_time) > timeout_s:
            raise TimeoutError(f"Timed out waiting for pretokenize completion: {done_path}")
        time.sleep(poll_s)


def mark_done(done_path: Path) -> None:
    done_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path = done_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as done_file:
        done_file.write("done\n")
    tmp_path.replace(done_path)


def clear_done(done_path: Path) -> None:
    if done_path.exists():
        done_path.unlink()


def normalize_storage_format(storage_format: str) -> str:
    normalized: str = str(storage_format).strip().lower()
    if normalized in {STORAGE_FORMAT_HYBRID, STORAGE_FORMAT_SIDECAR_ONLY}:
        return normalized
    raise ValueError(
        "Unsupported pretokenize.storage_format: "
        f"{storage_format!r}. Supported values: "
        f"{STORAGE_FORMAT_HYBRID!r}, {STORAGE_FORMAT_SIDECAR_ONLY!r}."
    )


def resolve_shard_path(*, cache_dir: Path, prefix: str, shard_id: int) -> Path:
    return cache_dir / f"{prefix}-{int(shard_id):05d}.parquet"


def parse_shard_id(shard_path: Path, prefix: str) -> int | None:
    filename: str = shard_path.name
    expected_prefix: str = f"{prefix}-"
    expected_suffix: str = ".parquet"
    if not filename.startswith(expected_prefix) or not filename.endswith(expected_suffix):
        return None
    shard_token: str = filename[len(expected_prefix) : -len(expected_suffix)]
    if not shard_token.isdigit():
        return None
    return int(shard_token)


def _parse_shard_id_from_sidecar_name(
    *,
    filename: str,
    prefix: str,
    sidecar_suffix: str,
) -> int | None:
    expected_prefix: str = f"{prefix}-"
    if not filename.startswith(expected_prefix) or not filename.endswith(sidecar_suffix):
        return None
    shard_token: str = filename[len(expected_prefix) : -len(sidecar_suffix)]
    if not shard_token.isdigit():
        return None
    return int(shard_token)


def iter_shard_paths(cache_dir: Path, prefix: str) -> list[Path]:
    shard_ids: set[int] = set()
    shard_path: Path
    for shard_path in cache_dir.glob(f"{prefix}-*.parquet"):
        shard_id: int | None = parse_shard_id(shard_path, prefix)
        if shard_id is not None:
            shard_ids.add(int(shard_id))
    sidecar_path: Path
    for sidecar_path in cache_dir.glob(f"{prefix}-*{INPUT_IDS_SIDECAR_SUFFIX}"):
        shard_id = _parse_shard_id_from_sidecar_name(
            filename=sidecar_path.name,
            prefix=prefix,
            sidecar_suffix=INPUT_IDS_SIDECAR_SUFFIX,
        )
        if shard_id is not None:
            shard_ids.add(int(shard_id))
    for sidecar_path in cache_dir.glob(f"{prefix}-*{ATTENTION_MASK_SIDECAR_SUFFIX}"):
        shard_id = _parse_shard_id_from_sidecar_name(
            filename=sidecar_path.name,
            prefix=prefix,
            sidecar_suffix=ATTENTION_MASK_SIDECAR_SUFFIX,
        )
        if shard_id is not None:
            shard_ids.add(int(shard_id))
    return [
        resolve_shard_path(cache_dir=cache_dir, prefix=prefix, shard_id=shard_id)
        for shard_id in sorted(shard_ids)
    ]


def resolve_input_ids_sidecar_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(INPUT_IDS_SIDECAR_SUFFIX)


def resolve_attention_mask_sidecar_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(ATTENTION_MASK_SIDECAR_SUFFIX)


def sidecar_exists(shard_path: Path) -> bool:
    return (
        resolve_input_ids_sidecar_path(shard_path).is_file()
        and resolve_attention_mask_sidecar_path(shard_path).is_file()
    )


def resolve_index_path(cache_dir: Path, prefix: str) -> Path:
    return cache_dir / f"{prefix}{INDEX_FILENAME_SUFFIX}"


def index_exists(cache_dir: Path, prefix: str) -> bool:
    return resolve_index_path(cache_dir, prefix).is_file()


def resolve_row_index_path(cache_dir: Path, prefix: str) -> Path:
    return cache_dir / f"{prefix}{ROW_INDEX_FILENAME_SUFFIX}"


def row_index_exists(cache_dir: Path, prefix: str) -> bool:
    return resolve_row_index_path(cache_dir, prefix).is_file()


def remove_shards(cache_dir: Path, prefix: str) -> None:
    shard_path: Path
    for shard_path in iter_shard_paths(cache_dir, prefix):
        remove_sidecars(shard_path)
        if shard_path.is_file():
            shard_path.unlink()


def remove_index(cache_dir: Path, prefix: str) -> None:
    index_path: Path = resolve_index_path(cache_dir, prefix)
    if index_path.is_file():
        index_path.unlink()


def remove_row_index(cache_dir: Path, prefix: str) -> None:
    row_index_path: Path = resolve_row_index_path(cache_dir, prefix)
    if row_index_path.is_file():
        row_index_path.unlink()


def remove_sidecars(shard_path: Path) -> None:
    input_ids_sidecar: Path = resolve_input_ids_sidecar_path(shard_path)
    attention_mask_sidecar: Path = resolve_attention_mask_sidecar_path(shard_path)
    if input_ids_sidecar.is_file():
        input_ids_sidecar.unlink()
    if attention_mask_sidecar.is_file():
        attention_mask_sidecar.unlink()


def write_row_index(cache_dir: Path, prefix: str, row_index: np.ndarray) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = resolve_row_index_path(cache_dir, prefix)
    tmp_path: Path = output_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as output_file:
        np.save(output_file, row_index, allow_pickle=False)
    tmp_path.replace(output_path)
    return output_path


def _resolve_id_arrow_type(write_dtype: str) -> pa.DataType:
    normalized: str = str(write_dtype).strip().lower()
    if normalized == "int64":
        return pa.int64()
    if normalized == "int32":
        return pa.int32()
    raise ValueError(f"Unsupported pretokenize.write_dtype: {write_dtype}")


def _rows_to_matrix(rows: list[list[int]], *, dtype: np.dtype) -> np.ndarray | None:
    if not rows:
        return np.empty((0, 0), dtype=dtype)
    row_width: int = len(rows[0])
    row: list[int]
    for row in rows:
        if len(row) != row_width:
            return None
    matrix: np.ndarray = np.asarray(rows, dtype=dtype)
    if matrix.ndim != 2:
        return None
    return matrix


def _write_numpy_sidecars(
    *,
    shard_path: Path,
    input_ids: list[list[int]],
    attention_masks: list[list[int]],
) -> bool:
    input_ids_matrix: np.ndarray | None = _rows_to_matrix(
        input_ids, dtype=np.int32
    )
    attention_mask_matrix: np.ndarray | None = _rows_to_matrix(
        attention_masks, dtype=np.int8
    )
    if input_ids_matrix is None or attention_mask_matrix is None:
        remove_sidecars(shard_path)
        return False
    if input_ids_matrix.shape != attention_mask_matrix.shape:
        remove_sidecars(shard_path)
        return False
    input_ids_sidecar: Path = resolve_input_ids_sidecar_path(shard_path)
    attention_mask_sidecar: Path = resolve_attention_mask_sidecar_path(shard_path)
    input_ids_tmp_path: Path = input_ids_sidecar.with_suffix(".tmp")
    attention_mask_tmp_path: Path = attention_mask_sidecar.with_suffix(".tmp")
    with open(input_ids_tmp_path, "wb") as input_ids_file:
        np.save(input_ids_file, input_ids_matrix, allow_pickle=False)
    with open(attention_mask_tmp_path, "wb") as attention_mask_file:
        np.save(attention_mask_file, attention_mask_matrix, allow_pickle=False)
    input_ids_tmp_path.replace(input_ids_sidecar)
    attention_mask_tmp_path.replace(attention_mask_sidecar)
    return True


def _chunked_list_column_to_fixed_matrix(
    column: pa.ChunkedArray,
) -> np.ndarray | None:
    combined: pa.Array = column.combine_chunks()
    if combined.null_count > 0:
        return None
    if not (pa.types.is_list(combined.type) or pa.types.is_large_list(combined.type)):
        return None
    offsets: np.ndarray = combined.offsets.to_numpy(zero_copy_only=False)
    if offsets.ndim != 1 or offsets.shape[0] != (int(len(combined)) + 1):
        return None
    if int(len(combined)) == 0:
        return np.empty((0, 0), dtype=np.int64)
    row_lengths: np.ndarray = np.diff(offsets)
    if row_lengths.size == 0:
        return np.empty((int(len(combined)), 0), dtype=np.int64)
    row_width: int = int(row_lengths[0])
    if not bool(np.all(row_lengths == row_width)):
        return None
    values: np.ndarray = combined.values.to_numpy(zero_copy_only=False)
    expected_values: int = int(len(combined)) * int(row_width)
    if int(values.size) != expected_values:
        return None
    return values.reshape(int(len(combined)), int(row_width))


def write_numpy_sidecar_from_parquet_shard(shard_path: Path) -> bool:
    if not shard_path.is_file():
        return False
    table: pa.Table = pq.read_table(
        shard_path,
        columns=["input_ids", "attention_mask"],
    )
    input_ids_matrix: np.ndarray | None = _chunked_list_column_to_fixed_matrix(
        table.column("input_ids")
    )
    attention_mask_matrix: np.ndarray | None = _chunked_list_column_to_fixed_matrix(
        table.column("attention_mask")
    )
    if input_ids_matrix is None or attention_mask_matrix is None:
        return False
    if input_ids_matrix.shape != attention_mask_matrix.shape:
        return False
    input_ids_sidecar: Path = resolve_input_ids_sidecar_path(shard_path)
    attention_mask_sidecar: Path = resolve_attention_mask_sidecar_path(shard_path)
    input_ids_tmp_path: Path = input_ids_sidecar.with_suffix(".tmp")
    attention_mask_tmp_path: Path = attention_mask_sidecar.with_suffix(".tmp")
    with open(input_ids_tmp_path, "wb") as input_ids_file:
        np.save(input_ids_file, input_ids_matrix.astype(np.int32, copy=False), allow_pickle=False)
    with open(attention_mask_tmp_path, "wb") as attention_mask_file:
        np.save(
            attention_mask_file,
            attention_mask_matrix.astype(np.int8, copy=False),
            allow_pickle=False,
        )
    input_ids_tmp_path.replace(input_ids_sidecar)
    attention_mask_tmp_path.replace(attention_mask_sidecar)
    return True


def _write_shard(
    *,
    shard_path: Path,
    keys: list[str],
    input_ids: list[list[int]],
    attention_masks: list[list[int]],
    id_arrow_type: pa.DataType,
    parquet_row_group_size: int | None,
    write_numpy_sidecar: bool,
    storage_format: str,
) -> None:
    normalized_storage_format: str = normalize_storage_format(storage_format)
    if normalized_storage_format == STORAGE_FORMAT_SIDECAR_ONLY:
        sidecar_written: bool = _write_numpy_sidecars(
            shard_path=shard_path,
            input_ids=input_ids,
            attention_masks=attention_masks,
        )
        if not sidecar_written:
            raise ValueError(
                "pretokenize.storage_format='sidecar_only' requires fixed-width "
                "token rows. Use max_padding=true or switch to storage_format='hybrid'."
            )
        if shard_path.is_file():
            shard_path.unlink()
        return

    schema: pa.Schema = pa.schema(
        [
            ("key", pa.string()),
            ("input_ids", pa.list_(id_arrow_type)),
            ("attention_mask", pa.list_(pa.int8())),
        ]
    )
    table: pa.Table = pa.Table.from_pydict(
        {
            "key": keys,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        },
        schema=schema,
    )
    tmp_path: Path = shard_path.with_suffix(".tmp")
    pq.write_table(
        table,
        tmp_path,
        row_group_size=(
            int(parquet_row_group_size)
            if parquet_row_group_size is not None
            else None
        ),
    )
    tmp_path.replace(shard_path)
    if write_numpy_sidecar:
        _write_numpy_sidecars(
            shard_path=shard_path,
            input_ids=input_ids,
            attention_masks=attention_masks,
        )


def write_token_shards(
    *,
    cache_dir: Path,
    prefix: str,
    rows: Iterable[tuple[str, Sequence[int], Sequence[int]]],
    shard_size: int,
    write_dtype: str,
    index_backend: str = "sqlite",
    parquet_row_group_size: int | None = None,
    write_numpy_sidecar: bool = False,
    storage_format: str = STORAGE_FORMAT_HYBRID,
) -> int:
    if shard_size <= 0:
        raise ValueError("shard_size must be a positive integer.")
    normalized_index_backend: str = str(index_backend).strip().lower()
    if normalized_index_backend != "sqlite":
        raise ValueError(
            f"Unsupported pretokenize.streaming_index_backend: {index_backend}"
        )
    resolved_row_group_size: int | None = None
    if parquet_row_group_size is not None:
        resolved_row_group_size = int(parquet_row_group_size)
        if resolved_row_group_size <= 0:
            raise ValueError(
                "parquet_row_group_size must be a positive integer when provided."
            )
    normalized_storage_format: str = normalize_storage_format(storage_format)
    if normalized_storage_format == STORAGE_FORMAT_SIDECAR_ONLY:
        write_numpy_sidecar = True
    id_arrow_type: pa.DataType = _resolve_id_arrow_type(write_dtype)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path: Path = resolve_index_path(cache_dir, prefix)
    index_tmp_path: Path = index_path.with_suffix(index_path.suffix + ".tmp")
    if index_tmp_path.exists():
        index_tmp_path.unlink()

    shard_idx: int = 0
    row_count: int = 0
    keys: list[str] = []
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    conn: sqlite3.Connection = sqlite3.connect(str(index_tmp_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS token_index (
            key TEXT PRIMARY KEY,
            shard_id INTEGER NOT NULL,
            row_idx INTEGER NOT NULL
        )
        """
    )

    def flush() -> None:
        nonlocal shard_idx
        if not keys:
            return
        shard_path: Path = resolve_shard_path(
            cache_dir=cache_dir,
            prefix=prefix,
            shard_id=shard_idx,
        )
        _write_shard(
            shard_path=shard_path,
            keys=keys,
            input_ids=input_ids,
            attention_masks=attention_masks,
            id_arrow_type=id_arrow_type,
            parquet_row_group_size=resolved_row_group_size,
            write_numpy_sidecar=bool(write_numpy_sidecar),
            storage_format=normalized_storage_format,
        )
        index_rows: list[tuple[str, int, int]] = [
            (key, int(shard_idx), int(row_idx))
            for row_idx, key in enumerate(keys)
        ]
        conn.executemany(
            (
                "INSERT OR REPLACE INTO token_index "
                "(key, shard_id, row_idx) VALUES (?, ?, ?)"
            ),
            index_rows,
        )
        keys.clear()
        input_ids.clear()
        attention_masks.clear()
        shard_idx += 1

    key: str
    ids: Sequence[int]
    mask: Sequence[int]
    for key, ids, mask in rows:
        keys.append(str(key))
        input_ids.append([int(token_id) for token_id in ids])
        attention_masks.append([int(value) for value in mask])
        row_count += 1
        if len(keys) >= shard_size:
            flush()
    try:
        flush()
        conn.commit()
    finally:
        conn.close()
    index_tmp_path.replace(index_path)
    return row_count


def load_token_cache(
    *, cache_dir: Path, prefix: str
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    shard_paths: list[Path] = iter_shard_paths(cache_dir, prefix)
    index_path: Path = resolve_index_path(cache_dir, prefix)
    index_conn: sqlite3.Connection | None = (
        sqlite3.connect(str(index_path)) if index_path.is_file() else None
    )
    shard_path: Path
    try:
        for shard_path in shard_paths:
            if shard_path.is_file():
                table: pa.Table = pq.read_table(
                    shard_path,
                    columns=["key", "input_ids", "attention_mask"],
                )
                keys: list[str | None] = table.column("key").to_pylist()
                ids_rows: list[list[int] | None] = table.column("input_ids").to_pylist()
                mask_rows: list[list[int] | None] = table.column("attention_mask").to_pylist()
                key: str | None
                ids: list[int] | None
                mask: list[int] | None
                for key, ids, mask in zip(keys, ids_rows, mask_rows):
                    if key is None:
                        continue
                    input_ids_tensor: torch.Tensor = torch.tensor(
                        [] if ids is None else ids,
                        dtype=torch.long,
                    )
                    attention_mask_tensor: torch.Tensor = torch.tensor(
                        [] if mask is None else mask,
                        dtype=torch.long,
                    )
                    cache[str(key)] = (input_ids_tensor, attention_mask_tensor)
                continue

            if not sidecar_exists(shard_path):
                raise FileNotFoundError(
                    "Missing token shard: expected parquet or sidecars at "
                    f"{shard_path.as_posix()}"
                )
            if index_conn is None:
                raise FileNotFoundError(
                    "Missing token index for sidecar-only cache: "
                    f"{index_path.as_posix()}"
                )
            shard_id: int | None = parse_shard_id(shard_path, prefix)
            if shard_id is None:
                raise ValueError(
                    "Failed to parse shard id from sidecar shard path: "
                    f"{shard_path.as_posix()}"
                )
            input_ids_matrix: np.ndarray = np.load(
                str(resolve_input_ids_sidecar_path(shard_path)),
                mmap_mode="r",
                allow_pickle=False,
            )
            attention_mask_matrix: np.ndarray = np.load(
                str(resolve_attention_mask_sidecar_path(shard_path)),
                mmap_mode="r",
                allow_pickle=False,
            )
            cursor: sqlite3.Cursor = index_conn.execute(
                (
                    "SELECT key, row_idx FROM token_index "
                    "WHERE shard_id = ? ORDER BY row_idx ASC"
                ),
                (int(shard_id),),
            )
            key: str
            row_idx: int
            for key, row_idx in cursor:
                row_idx_value: int = int(row_idx)
                if row_idx_value < 0 or row_idx_value >= int(input_ids_matrix.shape[0]):
                    raise IndexError(
                        "Pretokenize index row pointer is out of sidecar bounds: "
                        f"prefix={prefix!r} shard={shard_id} row_idx={row_idx_value}"
                    )
                cache[str(key)] = (
                    torch.tensor(input_ids_matrix[row_idx_value], dtype=torch.long),
                    torch.tensor(
                        attention_mask_matrix[row_idx_value], dtype=torch.long
                    ),
                )
    finally:
        if index_conn is not None:
            index_conn.close()
    return cache
