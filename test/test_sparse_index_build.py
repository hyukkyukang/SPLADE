import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.index.sparse import ShardInfo, build_inverted_index_from_shards


class SparseIndexBuildTest(unittest.TestCase):
    def test_build_inverted_index_casts_float16_shard_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            indptr_path = root / "indptr.npy"
            indices_path = root / "indices.npy"
            values_path = root / "values.npy"
            doc_ids_path = root / "doc_ids.json"

            np.save(indptr_path, np.array([0, 2], dtype=np.int64))
            np.save(indices_path, np.array([1, 3], dtype=np.int32))
            np.save(values_path, np.array([0.5, 1.5], dtype=np.float16))
            with doc_ids_path.open("w", encoding="utf-8") as handle:
                json.dump(["doc-1"], handle)

            shard = ShardInfo(
                rank=0,
                shard_id=0,
                doc_count=1,
                nnz=2,
                indptr_path=indptr_path,
                indices_path=indices_path,
                values_path=values_path,
                doc_ids_path=doc_ids_path,
            )
            term_ptr, post_doc_ids, post_weights, doc_ids = (
                build_inverted_index_from_shards(
                    [shard],
                    vocab_size=5,
                    value_dtype=np.dtype("float32"),
                )
            )

            self.assertEqual(post_weights.dtype, np.float32)
            np.testing.assert_array_equal(term_ptr, np.array([0, 0, 1, 1, 2, 2]))
            np.testing.assert_array_equal(post_doc_ids, np.array([0, 0], dtype=np.int32))
            np.testing.assert_allclose(
                post_weights, np.array([0.5, 1.5], dtype=np.float32)
            )
            self.assertEqual(doc_ids, ["doc-1"])


if __name__ == "__main__":
    unittest.main()
