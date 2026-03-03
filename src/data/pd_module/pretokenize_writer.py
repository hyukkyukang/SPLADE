import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.data.pd_module.pretokenize import (
    make_id_key,
    resolve_index_path,
    write_row_index,
    write_token_shards,
)
from src.utils.logging import log_if_rank_zero


class PretokenizeCacheStorageWriter:
    """Handle pretokenized cache writing and row-index generation."""

    def __init__(self, *, owner: Any, logger: logging.Logger) -> None:
        self._owner: Any = owner
        self._logger: logging.Logger = logger

    @staticmethod
    def build_dataset_row_index(
        *,
        keys: Iterable[str],
        id_to_idx: Mapping[str, int],
        dataset_size: int,
    ) -> np.ndarray:
        row_index: np.ndarray = np.full(int(dataset_size), -1, dtype=np.int64)
        global_row_idx: int = 0
        key: str
        for key in keys:
            if key.startswith(make_id_key("")):
                raw_id: str = key[3:]
                dataset_idx_value: int | None = id_to_idx.get(raw_id)
                if dataset_idx_value is not None:
                    dataset_idx: int = int(dataset_idx_value)
                    if 0 <= dataset_idx < int(dataset_size):
                        row_index[dataset_idx] = int(global_row_idx)
            global_row_idx += 1
        return row_index

    def build_dataset_row_index_from_sqlite(
        self,
        *,
        prefix: str,
        id_to_idx: Mapping[str, int],
        dataset_size: int,
        shard_size: int,
    ) -> np.ndarray:
        owner: Any = self._owner
        row_index: np.ndarray = np.full(int(dataset_size), -1, dtype=np.int64)
        index_path: Path = resolve_index_path(owner._cache_dir, prefix)
        conn: sqlite3.Connection = sqlite3.connect(str(index_path))
        try:
            cursor: sqlite3.Cursor = conn.execute(
                "SELECT key, shard_id, row_idx FROM token_index WHERE key LIKE 'id:%'"
            )
            key: str
            shard_id: int
            row_idx_local: int
            for key, shard_id, row_idx_local in cursor:
                raw_id: str = str(key)[3:]
                dataset_idx_value: int | None = id_to_idx.get(raw_id)
                if dataset_idx_value is None:
                    continue
                dataset_idx: int = int(dataset_idx_value)
                if dataset_idx < 0 or dataset_idx >= int(dataset_size):
                    continue
                global_row_idx: int = int(shard_id) * int(shard_size) + int(
                    row_idx_local
                )
                row_index[dataset_idx] = int(global_row_idx)
        finally:
            conn.close()
        return row_index

    def write_cache_entries(
        self,
        *,
        query_items: dict[str, str],
        doc_items: dict[str, str],
    ) -> tuple[int, int, np.ndarray | None, np.ndarray | None]:
        owner: Any = self._owner
        query_row_index: np.ndarray | None = None
        doc_row_index: np.ndarray | None = None
        if owner._streaming_use_dataset_row_index:
            query_row_index = self.build_dataset_row_index(
                keys=query_items.keys(),
                id_to_idx=owner.dataset.query_dataset_id_to_idx,
                dataset_size=len(owner.dataset.query_dataset),
            )
            doc_row_index = self.build_dataset_row_index(
                keys=doc_items.keys(),
                id_to_idx=owner.dataset.corpus_dataset_id_to_idx,
                dataset_size=len(owner.dataset.corpus_dataset),
            )
        if owner._enable_pretokenize_tokenizers_parallelism:
            log_if_rank_zero(
                self._logger,
                "Pretokenize tokenizers parallelism enabled "
                "(TOKENIZERS_PARALLELISM=true).",
            )
        with owner._tokenizers_parallelism_context():
            query_count: int = write_token_shards(
                cache_dir=owner._cache_dir,
                prefix="queries",
                rows=owner._tokenize_rows(
                    items=query_items,
                    max_length=owner.max_query_length,
                    phase_name="queries",
                ),
                shard_size=owner._query_shard_size,
                write_dtype=owner._write_dtype,
                index_backend=owner._streaming_index_backend,
                parquet_row_group_size=owner._parquet_row_group_size,
                write_numpy_sidecar=owner._streaming_numpy_sidecar,
                storage_format=owner._pretokenize_storage_format,
            )
            doc_count: int = write_token_shards(
                cache_dir=owner._cache_dir,
                prefix="docs",
                rows=owner._tokenize_rows(
                    items=doc_items,
                    max_length=owner.max_doc_length,
                    phase_name="docs",
                ),
                shard_size=owner._doc_shard_size,
                write_dtype=owner._write_dtype,
                index_backend=owner._streaming_index_backend,
                parquet_row_group_size=owner._parquet_row_group_size,
                write_numpy_sidecar=owner._streaming_numpy_sidecar,
                storage_format=owner._pretokenize_storage_format,
            )
        if (
            owner._streaming_use_dataset_row_index
            and query_row_index is not None
            and doc_row_index is not None
        ):
            write_row_index(
                cache_dir=owner._cache_dir,
                prefix="queries",
                row_index=query_row_index,
            )
            write_row_index(
                cache_dir=owner._cache_dir,
                prefix="docs",
                row_index=doc_row_index,
            )
        return query_count, doc_count, query_row_index, doc_row_index
