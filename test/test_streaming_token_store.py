import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.pd_module.pretokenize import (
    STORAGE_FORMAT_SIDECAR_ONLY,
    write_row_index,
    write_token_shards,
)
from src.data.pd_module.token_store import StreamingTokenStore


class StreamingTokenStoreTest(unittest.TestCase):
    def test_store_reads_rows_from_sqlite_index_and_parquet_shards(self) -> None:
        with tempfile.TemporaryDirectory(prefix="streaming_store_") as tmp_dir:
            cache_dir = Path(tmp_dir)
            rows = [
                ("id:q1", [1, 2, 3], [1, 1, 1]),
                ("id:q2", [7, 8], [1, 1]),
                ("id:q3", [9], [1]),
            ]
            written = write_token_shards(
                cache_dir=cache_dir,
                prefix="queries",
                rows=rows,
                shard_size=2,
                write_dtype="int32",
            )
            self.assertEqual(written, 3)
            self.assertTrue((cache_dir / "queries.index.sqlite").is_file())

            store = StreamingTokenStore(
                cache_dir=cache_dir,
                prefix="queries",
                max_cached_shards=1,
                max_cached_rows=1,
            )
            try:
                self.assertEqual(len(store), 3)
                q1_tokens = store.get("id:q1")
                self.assertIsNotNone(q1_tokens)
                q1_ids, q1_mask = q1_tokens
                self.assertEqual(q1_ids.tolist(), [1, 2, 3])
                self.assertEqual(q1_mask.tolist(), [1, 1, 1])

                q2_ids, q2_mask = store["id:q2"]
                self.assertEqual(q2_ids.tolist(), [7, 8])
                self.assertEqual(q2_mask.tolist(), [1, 1])
                self.assertIsNone(store.get("id:missing"))
            finally:
                store.close()

    def test_store_supports_fast_id_lookup_via_dataset_row_index(self) -> None:
        with tempfile.TemporaryDirectory(prefix="streaming_store_rowidx_") as tmp_dir:
            cache_dir = Path(tmp_dir)
            rows = [
                ("id:q1", [1, 2, 3], [1, 1, 1]),
                ("id:q2", [7, 8], [1, 1]),
                ("id:q3", [9], [1]),
            ]
            write_token_shards(
                cache_dir=cache_dir,
                prefix="queries",
                rows=rows,
                shard_size=2,
                write_dtype="int32",
            )
            row_index = np.array([0, 1, 2], dtype=np.int64)
            row_index_path = write_row_index(
                cache_dir=cache_dir,
                prefix="queries",
                row_index=row_index,
            )

            store = StreamingTokenStore(
                cache_dir=cache_dir,
                prefix="queries",
                max_cached_shards=1,
                max_cached_rows=1,
                id_to_dataset_idx={"q1": 0, "q2": 1, "q3": 2},
                dataset_idx_to_global_row_path=row_index_path,
                shard_size=2,
            )
            try:
                q3_ids, q3_mask = store["id:q3"]
                self.assertEqual(q3_ids.tolist(), [9])
                self.assertEqual(q3_mask.tolist(), [1])
                by_global = store.get_by_global_row(2)
                self.assertIsNotNone(by_global)
                by_global_ids, by_global_mask = by_global
                self.assertEqual(by_global_ids.tolist(), [9])
                self.assertEqual(by_global_mask.tolist(), [1])
            finally:
                store.close()

    def test_store_reads_sidecar_only_shards(self) -> None:
        with tempfile.TemporaryDirectory(prefix="streaming_store_sidecar_only_") as tmp_dir:
            cache_dir = Path(tmp_dir)
            rows = [
                ("id:q1", [10, 11], [1, 1]),
                ("id:q2", [20, 21], [1, 1]),
                ("id:q3", [30, 31], [1, 1]),
            ]
            written = write_token_shards(
                cache_dir=cache_dir,
                prefix="queries",
                rows=rows,
                shard_size=2,
                write_dtype="int32",
                storage_format=STORAGE_FORMAT_SIDECAR_ONLY,
            )
            self.assertEqual(written, 3)
            self.assertTrue((cache_dir / "queries.index.sqlite").is_file())
            self.assertFalse(any(cache_dir.glob("queries-*.parquet")))
            self.assertTrue(any(cache_dir.glob("queries-*.input_ids.npy")))
            self.assertTrue(any(cache_dir.glob("queries-*.attention_mask.npy")))

            store = StreamingTokenStore(
                cache_dir=cache_dir,
                prefix="queries",
                max_cached_shards=1,
                max_cached_rows=8,
                shard_size=2,
            )
            try:
                q2_ids, q2_mask = store["id:q2"]
                self.assertEqual(q2_ids.tolist(), [20, 21])
                self.assertEqual(q2_mask.tolist(), [1, 1])

                row0 = store.get_by_global_row(0)
                self.assertIsNotNone(row0)
                row0_ids, row0_mask = row0
                self.assertEqual(row0_ids.tolist(), [10, 11])
                self.assertEqual(row0_mask.tolist(), [1, 1])
            finally:
                store.close()

    def test_store_batch_lookup_by_global_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="streaming_store_batch_global_") as tmp_dir:
            cache_dir = Path(tmp_dir)
            rows = [
                ("id:q1", [10, 11], [1, 1]),
                ("id:q2", [20, 21], [1, 1]),
                ("id:q3", [30, 31], [1, 1]),
            ]
            write_token_shards(
                cache_dir=cache_dir,
                prefix="queries",
                rows=rows,
                shard_size=2,
                write_dtype="int32",
                storage_format=STORAGE_FORMAT_SIDECAR_ONLY,
            )
            store = StreamingTokenStore(
                cache_dir=cache_dir,
                prefix="queries",
                max_cached_shards=2,
                max_cached_rows=8,
                shard_size=2,
            )
            try:
                batch = store.get_many_by_global_rows([0, 2, 99], default=None)
                self.assertEqual(len(batch), 3)
                self.assertIsNotNone(batch[0])
                self.assertIsNotNone(batch[1])
                self.assertIsNone(batch[2])

                row0_ids, row0_mask = batch[0]  # type: ignore[misc]
                row2_ids, row2_mask = batch[1]  # type: ignore[misc]
                self.assertEqual(row0_ids.tolist(), [10, 11])
                self.assertEqual(row0_mask.tolist(), [1, 1])
                self.assertEqual(row2_ids.tolist(), [30, 31])
                self.assertEqual(row2_mask.tolist(), [1, 1])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
