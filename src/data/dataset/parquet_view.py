from __future__ import annotations

import bisect
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import ListConfig


@dataclass(frozen=True)
class _RowGroupEntry:
    path: str
    row_group_idx: int
    start_idx: int
    num_rows: int


def _normalize_split_data_files(
    data_files: Mapping[str, Any] | Sequence[str] | str,
    *,
    split: str,
) -> list[str]:
    if isinstance(data_files, str):
        patterns: list[str] = [data_files]
    elif isinstance(data_files, Mapping):
        split_value: Any | None = data_files.get(split)
        if split_value is None:
            raise ValueError(
                f"Missing parquet data_files entry for split={split!r}."
            )
        if isinstance(split_value, str):
            patterns = [split_value]
        elif isinstance(split_value, (list, tuple, ListConfig)):
            patterns = [str(value) for value in split_value]
        else:
            raise ValueError(
                "Parquet split data_files must be a string or list of strings."
            )
    elif isinstance(data_files, (list, tuple, ListConfig)):
        patterns = [str(value) for value in data_files]
    else:
        raise ValueError("Unsupported parquet data_files specification.")

    resolved_paths: list[str] = []
    pattern: str
    for pattern in patterns:
        matches: list[str] = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No parquet files matched pattern: {pattern}")
        resolved_paths.extend(matches)
    if not resolved_paths:
        raise FileNotFoundError("No parquet files resolved from data_files.")
    # Remove duplicates while preserving order.
    unique_paths: list[str] = []
    seen_paths: set[str] = set()
    path: str
    for path in resolved_paths:
        normalized: str = str(Path(path).resolve())
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        unique_paths.append(normalized)
    return unique_paths


class ProjectedParquetDataset:
    """Row-addressable local parquet view with row-group caching and column projection."""

    def __init__(
        self,
        *,
        data_files: Mapping[str, Any] | Sequence[str] | str,
        split: str,
        columns: Sequence[str],
    ) -> None:
        self._paths: list[str] = _normalize_split_data_files(data_files, split=split)
        self._columns: tuple[str, ...] = tuple(dict.fromkeys(str(col) for col in columns))
        self._entries: list[_RowGroupEntry] = []
        self._row_group_starts: list[int] = []
        self._total_rows: int = 0
        self._parquet_files: dict[str, pq.ParquetFile] = {}
        self._cached_key: tuple[str, int] | None = None
        self._cached_table: pa.Table | None = None
        self._projected_column_names: list[str] = list(self._columns)

        path: str
        for path in self._paths:
            parquet_file: pq.ParquetFile = pq.ParquetFile(path)
            self._parquet_files[path] = parquet_file
            row_group_idx: int
            for row_group_idx in range(parquet_file.num_row_groups):
                num_rows: int = int(parquet_file.metadata.row_group(row_group_idx).num_rows)
                self._row_group_starts.append(self._total_rows)
                self._entries.append(
                    _RowGroupEntry(
                        path=path,
                        row_group_idx=row_group_idx,
                        start_idx=self._total_rows,
                        num_rows=num_rows,
                    )
                )
                self._total_rows += num_rows

    def __len__(self) -> int:
        return self._total_rows

    def __getstate__(self) -> dict[str, Any]:
        state: dict[str, Any] = dict(self.__dict__)
        state["_parquet_files"] = {}
        state["_cached_key"] = None
        state["_cached_table"] = None
        return state

    def __getitem__(self, idx: int | str) -> dict[str, Any] | list[Any]:
        if isinstance(idx, str):
            return self.get_column(idx).to_pylist()
        index: int = int(idx)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entry_idx: int = bisect.bisect_right(self._row_group_starts, index) - 1
        entry: _RowGroupEntry = self._entries[entry_idx]
        row_offset: int = index - entry.start_idx
        table: pa.Table = self._get_row_group_table(entry.path, entry.row_group_idx)
        return {
            column_name: table.column(column_idx)[row_offset].as_py()
            for column_idx, column_name in enumerate(self._columns)
        }

    @property
    def column_names(self) -> list[str]:
        return list(self._projected_column_names)

    def get_column(self, column_name: str) -> pa.ChunkedArray:
        requested_column: str = str(column_name)
        if requested_column not in self._columns:
            raise KeyError(requested_column)
        chunks: list[pa.Array] = []
        path: str
        for path in self._paths:
            parquet_file: pq.ParquetFile | None = self._parquet_files.get(path)
            if parquet_file is None:
                parquet_file = pq.ParquetFile(path)
                self._parquet_files[path] = parquet_file
            table: pa.Table = parquet_file.read(
                columns=[requested_column],
                use_threads=True,
            )
            chunks.extend(table.column(0).chunks)
        return pa.chunked_array(chunks)

    def _get_row_group_table(self, path: str, row_group_idx: int) -> pa.Table:
        cache_key: tuple[str, int] = (path, int(row_group_idx))
        if self._cached_key == cache_key and self._cached_table is not None:
            return self._cached_table
        parquet_file: pq.ParquetFile | None = self._parquet_files.get(path)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(path)
            self._parquet_files[path] = parquet_file
        table: pa.Table = parquet_file.read_row_group(
            int(row_group_idx),
            columns=list(self._columns),
            use_threads=True,
        )
        self._cached_key = cache_key
        self._cached_table = table
        return table


__all__ = ["ProjectedParquetDataset"]
