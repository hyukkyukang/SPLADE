from __future__ import annotations

import sqlite3
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.data.pd_module.pretokenize import (
    resolve_attention_mask_sidecar_path,
    resolve_index_path,
    resolve_input_ids_sidecar_path,
)


@dataclass(frozen=True)
class _CachedShard:
    num_rows: int
    input_ids: pa.ChunkedArray | None
    attention_mask: pa.ChunkedArray | None
    input_ids_matrix: np.ndarray | None
    attention_mask_matrix: np.ndarray | None


class StreamingTokenStore(Mapping[str, tuple[torch.Tensor, torch.Tensor]]):
    """Read pretokenized shards lazily via a SQLite key index."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        prefix: str,
        max_cached_shards: int = 2,
        max_cached_rows: int = 200000,
        max_cached_index_rows: int | None = None,
        sqlite_cache_size_kib: int = 131072,
        sqlite_mmap_size: int = 1073741824,
        id_to_dataset_idx: Mapping[str, int] | None = None,
        dataset_idx_to_global_row_path: Path | None = None,
        shard_size: int | None = None,
    ) -> None:
        self.cache_dir: Path = cache_dir
        self.prefix: str = str(prefix)
        self.max_cached_shards: int = max(1, int(max_cached_shards))
        self.max_cached_rows: int = max(0, int(max_cached_rows))
        self.max_cached_index_rows: int = max(
            0,
            int(
                max_cached_rows
                if max_cached_index_rows is None
                else max_cached_index_rows
            ),
        )
        self.sqlite_cache_size_kib: int = max(0, int(sqlite_cache_size_kib))
        self.sqlite_mmap_size: int = max(0, int(sqlite_mmap_size))
        self.id_to_dataset_idx: Mapping[str, int] | None = id_to_dataset_idx
        self.shard_size: int | None = (
            None if shard_size is None else max(1, int(shard_size))
        )
        self._dataset_idx_to_global_row: np.ndarray | None = None
        if dataset_idx_to_global_row_path is not None and dataset_idx_to_global_row_path.is_file():
            loaded_row_index: np.ndarray = np.load(
                str(dataset_idx_to_global_row_path),
                mmap_mode="r",
                allow_pickle=False,
            )
            self._dataset_idx_to_global_row = loaded_row_index

        self._index_path: Path = resolve_index_path(cache_dir, prefix)
        self._index_conn: sqlite3.Connection | None = None
        self._lookup_cursor: sqlite3.Cursor | None = None
        self._shard_cache: OrderedDict[int, _CachedShard] = OrderedDict()
        self._row_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = (
            OrderedDict()
        )
        self._index_cache: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._count: int | None = None

    def _cache_row_tokens(
        self,
        *,
        cache_key: str,
        tokens: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        if self.max_cached_rows <= 0:
            return
        self._row_cache[cache_key] = tokens
        if len(self._row_cache) > self.max_cached_rows:
            self._row_cache.popitem(last=False)

    def _connect(self) -> sqlite3.Connection:
        if self._index_conn is not None:
            return self._index_conn
        if not self._index_path.is_file():
            raise FileNotFoundError(
                f"Missing streaming token index: {self._index_path.as_posix()}"
            )
        uri: str = f"file:{self._index_path.as_posix()}?mode=ro"
        self._index_conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn: sqlite3.Connection = self._index_conn
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        if self.sqlite_cache_size_kib > 0:
            conn.execute(f"PRAGMA cache_size=-{self.sqlite_cache_size_kib}")
        if self.sqlite_mmap_size > 0:
            conn.execute(f"PRAGMA mmap_size={self.sqlite_mmap_size}")
        self._lookup_cursor = conn.cursor()
        return self._index_conn

    def _shard_path(self, shard_id: int) -> Path:
        return self.cache_dir / f"{self.prefix}-{int(shard_id):05d}.parquet"

    @staticmethod
    def _load_tokens_from_shard_row(
        shard: _CachedShard, row_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if row_idx < 0 or row_idx >= int(shard.num_rows):
            raise IndexError(f"row_idx out of bounds for shard: {row_idx}")
        if (
            shard.input_ids_matrix is not None
            and shard.attention_mask_matrix is not None
        ):
            return (
                torch.from_numpy(shard.input_ids_matrix[row_idx]),
                torch.from_numpy(shard.attention_mask_matrix[row_idx]),
            )
        input_ids_column: pa.ChunkedArray | None = shard.input_ids
        attention_mask_column: pa.ChunkedArray | None = shard.attention_mask
        if input_ids_column is None or attention_mask_column is None:
            raise RuntimeError(
                "Invalid token shard cache state: missing Arrow columns and sidecars."
            )
        ids_raw: list[int] | None = input_ids_column[row_idx].as_py()
        mask_raw: list[int] | None = attention_mask_column[row_idx].as_py()
        return (
            torch.as_tensor(
                [] if ids_raw is None else ids_raw,
                dtype=torch.long,
            ),
            torch.as_tensor(
                [] if mask_raw is None else mask_raw,
                dtype=torch.long,
            ),
        )

    def _load_shard(self, shard_id: int) -> _CachedShard:
        cached: _CachedShard | None = self._shard_cache.get(int(shard_id))
        if cached is not None:
            self._shard_cache.move_to_end(int(shard_id))
            return cached

        shard_path: Path = self._shard_path(shard_id)
        input_ids_sidecar_path: Path = resolve_input_ids_sidecar_path(shard_path)
        attention_mask_sidecar_path: Path = resolve_attention_mask_sidecar_path(
            shard_path
        )
        if input_ids_sidecar_path.is_file() and attention_mask_sidecar_path.is_file():
            input_ids_matrix: np.ndarray = np.load(
                str(input_ids_sidecar_path),
                mmap_mode="c",
                allow_pickle=False,
            )
            attention_mask_matrix: np.ndarray = np.load(
                str(attention_mask_sidecar_path),
                mmap_mode="c",
                allow_pickle=False,
            )
            if (
                input_ids_matrix.ndim == 2
                and attention_mask_matrix.ndim == 2
                and input_ids_matrix.shape == attention_mask_matrix.shape
            ):
                shard = _CachedShard(
                    num_rows=int(input_ids_matrix.shape[0]),
                    input_ids=None,
                    attention_mask=None,
                    input_ids_matrix=input_ids_matrix,
                    attention_mask_matrix=attention_mask_matrix,
                )
                self._shard_cache[int(shard_id)] = shard
                if len(self._shard_cache) > self.max_cached_shards:
                    self._shard_cache.popitem(last=False)
                return shard
        if not shard_path.is_file():
            raise FileNotFoundError(
                "Missing streaming token shard: expected parquet or sidecars at "
                f"{shard_path.as_posix()}"
            )
        table: pa.Table = pq.read_table(
            shard_path,
            columns=["input_ids", "attention_mask"],
        )
        input_ids_column: pa.ChunkedArray = table.column("input_ids")
        attention_mask_column: pa.ChunkedArray = table.column("attention_mask")
        shard: _CachedShard = _CachedShard(
            num_rows=int(table.num_rows),
            input_ids=input_ids_column,
            attention_mask=attention_mask_column,
            input_ids_matrix=None,
            attention_mask_matrix=None,
        )
        self._shard_cache[int(shard_id)] = shard
        if len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return shard

    def _cache_index_row(self, key: str, shard_id: int, row_idx: int) -> None:
        if self.max_cached_index_rows <= 0:
            return
        self._index_cache[key] = (int(shard_id), int(row_idx))
        if len(self._index_cache) > self.max_cached_index_rows:
            self._index_cache.popitem(last=False)

    def _load_index_row(self, key: str) -> tuple[int, int] | None:
        cached: tuple[int, int] | None = self._index_cache.get(key)
        if cached is not None:
            self._index_cache.move_to_end(key)
            return cached
        conn: sqlite3.Connection = self._connect()
        cursor: sqlite3.Cursor = (
            self._lookup_cursor if self._lookup_cursor is not None else conn.cursor()
        )
        cursor.execute(
            "SELECT shard_id, row_idx FROM token_index WHERE key = ?",
            (str(key),),
        )
        index_row: tuple[int, int] | None = cursor.fetchone()
        if index_row is None:
            return None
        shard_id: int = int(index_row[0])
        row_idx: int = int(index_row[1])
        self._cache_index_row(key, shard_id, row_idx)
        return shard_id, row_idx

    def _load_row(
        self, key: str
    ) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, bool]:
        if key in self._row_cache:
            self._row_cache.move_to_end(key)
            return self._row_cache[key], True

        if key.startswith("id:"):
            id_value: str = key[3:]
            maybe_tokens: tuple[torch.Tensor, torch.Tensor] | None = (
                self._load_row_via_dataset_idx(id_value, key)
            )
            if maybe_tokens is not None:
                return maybe_tokens, True

        index_row: tuple[int, int] | None = self._load_index_row(str(key))
        if index_row is None:
            return None, False
        shard_id: int = int(index_row[0])
        row_idx: int = int(index_row[1])
        shard: _CachedShard = self._load_shard(shard_id)
        if row_idx < 0 or row_idx >= int(shard.num_rows):
            raise IndexError(
                "Pretokenize index row pointer is out of shard bounds: "
                f"prefix={self.prefix!r} key={key!r} shard={shard_id} row_idx={row_idx}"
            )
        tokens: tuple[torch.Tensor, torch.Tensor] = self._load_tokens_from_shard_row(
            shard, row_idx
        )
        self._cache_row_tokens(cache_key=key, tokens=tokens)
        return tokens, True

    def _load_row_via_dataset_idx(
        self, id_value: str, cache_key: str
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        id_map: Mapping[str, int] | None = self.id_to_dataset_idx
        global_row_map: np.ndarray | None = self._dataset_idx_to_global_row
        shard_size: int | None = self.shard_size
        if id_map is None or global_row_map is None or shard_size is None:
            return None
        dataset_idx_value: int | None = id_map.get(str(id_value))
        if dataset_idx_value is None:
            return None
        dataset_idx: int = int(dataset_idx_value)
        if dataset_idx < 0 or dataset_idx >= int(global_row_map.shape[0]):
            return None
        global_row_value: int = int(global_row_map[dataset_idx])
        if global_row_value < 0:
            return None
        tokens: tuple[torch.Tensor, torch.Tensor] | None = self.get_by_global_row(
            int(global_row_value),
            default=None,
        )
        if tokens is None:
            return None
        self._cache_row_tokens(cache_key=cache_key, tokens=tokens)
        return tokens

    def get_by_global_row(
        self,
        global_row: int,
        default: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        shard_size: int | None = self.shard_size
        if shard_size is None:
            return default
        global_row_value: int = int(global_row)
        if global_row_value < 0:
            return default
        cache_key: str = f"__global__:{global_row_value}"
        if cache_key in self._row_cache:
            self._row_cache.move_to_end(cache_key)
            return self._row_cache[cache_key]
        shard_id: int = int(global_row_value // shard_size)
        row_idx: int = int(global_row_value % shard_size)
        try:
            shard: _CachedShard = self._load_shard(shard_id)
        except FileNotFoundError:
            return default
        if row_idx < 0 or row_idx >= int(shard.num_rows):
            return default
        tokens: tuple[torch.Tensor, torch.Tensor] = self._load_tokens_from_shard_row(
            shard, row_idx
        )
        self._cache_row_tokens(cache_key=cache_key, tokens=tokens)
        return tokens

    def get_many_by_global_rows(
        self,
        global_rows: Iterable[int],
        default: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        shard_size: int | None = self.shard_size
        row_values: list[int] = [int(global_row) for global_row in global_rows]
        if shard_size is None:
            return [default for _ in row_values]

        results: list[tuple[torch.Tensor, torch.Tensor] | None] = [default] * len(
            row_values
        )
        row_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = self._row_cache
        pending_by_shard: dict[int, list[tuple[int, int, str]]] = {}
        output_idx: int
        global_row: int
        for output_idx, global_row in enumerate(row_values):
            if global_row < 0:
                continue
            cache_key: str = f"__global__:{global_row}"
            cached_tokens: tuple[torch.Tensor, torch.Tensor] | None = (
                row_cache.get(cache_key)
            )
            if cached_tokens is not None:
                row_cache.move_to_end(cache_key)
                results[output_idx] = cached_tokens
                continue
            shard_id: int = int(global_row // shard_size)
            row_idx: int = int(global_row % shard_size)
            shard_rows: list[tuple[int, int, str]] | None = pending_by_shard.get(shard_id)
            if shard_rows is None:
                pending_by_shard[shard_id] = [(output_idx, row_idx, cache_key)]
            else:
                shard_rows.append((output_idx, row_idx, cache_key))

        load_tokens_from_shard_row = self._load_tokens_from_shard_row
        cache_row_tokens = self._cache_row_tokens
        shard_id: int
        shard_rows: list[tuple[int, int, str]]
        for shard_id, shard_rows in pending_by_shard.items():
            try:
                shard: _CachedShard = self._load_shard(shard_id)
            except FileNotFoundError:
                continue
            max_rows: int = int(shard.num_rows)
            output_idx_value: int
            row_idx_value: int
            cache_key_value: str
            for output_idx_value, row_idx_value, cache_key_value in shard_rows:
                if row_idx_value < 0 or row_idx_value >= max_rows:
                    continue
                tokens = load_tokens_from_shard_row(shard, row_idx_value)
                results[output_idx_value] = tokens
                cache_row_tokens(cache_key=cache_key_value, tokens=tokens)
        return results

    def close(self) -> None:
        self._row_cache.clear()
        self._index_cache.clear()
        self._shard_cache.clear()
        conn: sqlite3.Connection | None = self._index_conn
        self._lookup_cursor = None
        self._index_conn = None
        self._count = None
        if conn is not None:
            conn.close()

    def __getitem__(self, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens: tuple[torch.Tensor, torch.Tensor] | None
        found: bool
        tokens, found = self._load_row(str(key))
        if not found or tokens is None:
            raise KeyError(key)
        return tokens

    def __iter__(self) -> Iterator[str]:
        conn: sqlite3.Connection = self._connect()
        cursor: sqlite3.Cursor = conn.execute(
            "SELECT key FROM token_index ORDER BY key ASC"
        )
        row: tuple[str]
        for row in cursor:
            yield str(row[0])

    def __len__(self) -> int:
        if self._count is None:
            conn: sqlite3.Connection = self._connect()
            cursor: sqlite3.Cursor = conn.execute(
                "SELECT COUNT(*) FROM token_index"
            )
            result: tuple[int] | None = cursor.fetchone()
            self._count = 0 if result is None else int(result[0])
        return int(self._count)

    def get(
        self, key: str, default: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        tokens: tuple[torch.Tensor, torch.Tensor] | None
        found: bool
        tokens, found = self._load_row(str(key))
        if not found:
            return default
        return tokens

    def __del__(self) -> None:
        self.close()
