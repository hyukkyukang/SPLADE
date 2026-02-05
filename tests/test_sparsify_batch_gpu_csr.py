import unittest

import numpy as np
import torch

from src.indexing.sparse_index import sparsify_batch_gpu_csr


class SparsifyBatchGpuCsrTest(unittest.TestCase):
    def test_topk_path(self) -> None:
        vectors = torch.tensor(
            [
                [0.0, 1.0, 0.5, 0.0],
                [0.2, 0.0, 3.0, 0.1],
            ],
            dtype=torch.float32,
        )
        indptr, indices, values = sparsify_batch_gpu_csr(
            vectors,
            exclude_token_ids=None,
            min_weight=0.15,
            top_k=2,
            value_dtype=np.float32,
        )
        np.testing.assert_array_equal(indptr.numpy(), np.array([0, 2, 4], dtype=np.int64))
        np.testing.assert_array_equal(
            indices.numpy(), np.array([1, 2, 0, 2], dtype=np.int32)
        )
        np.testing.assert_allclose(
            values.numpy(), np.array([1.0, 0.5, 0.2, 3.0], dtype=np.float32)
        )

    def test_threshold_with_exclude(self) -> None:
        vectors = torch.tensor(
            [
                [0.0, 1.0, 0.5],
                [0.2, 0.0, 3.0],
            ],
            dtype=torch.float32,
        )
        exclude_ids = torch.tensor([2], dtype=torch.long)
        indptr, indices, values = sparsify_batch_gpu_csr(
            vectors,
            exclude_token_ids=exclude_ids,
            min_weight=0.15,
            top_k=None,
            value_dtype=np.float32,
        )
        np.testing.assert_array_equal(indptr.numpy(), np.array([0, 1, 2], dtype=np.int64))
        np.testing.assert_array_equal(
            indices.numpy(), np.array([1, 0], dtype=np.int32)
        )
        np.testing.assert_allclose(
            values.numpy(), np.array([1.0, 0.2], dtype=np.float32)
        )

    def test_empty_batch(self) -> None:
        vectors = torch.empty((0, 4), dtype=torch.float32)
        indptr, indices, values = sparsify_batch_gpu_csr(
            vectors,
            exclude_token_ids=None,
            min_weight=0.0,
            top_k=2,
            value_dtype=np.float32,
        )
        np.testing.assert_array_equal(indptr.numpy(), np.array([0], dtype=np.int64))
        np.testing.assert_array_equal(indices.numpy(), np.array([], dtype=np.int32))
        np.testing.assert_array_equal(values.numpy(), np.array([], dtype=np.float32))

    def test_all_zero_threshold(self) -> None:
        vectors = torch.zeros((2, 3), dtype=torch.float32)
        indptr, indices, values = sparsify_batch_gpu_csr(
            vectors,
            exclude_token_ids=None,
            min_weight=0.0,
            top_k=None,
            value_dtype=np.float32,
        )
        np.testing.assert_array_equal(
            indptr.numpy(), np.array([0, 0, 0], dtype=np.int64)
        )
        np.testing.assert_array_equal(indices.numpy(), np.array([], dtype=np.int32))
        np.testing.assert_array_equal(values.numpy(), np.array([], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
