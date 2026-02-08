import unittest

import numpy as np

from src.search.scoring import (
    score_query_postings,
    score_query_postings_bmw,
    score_query_postings_wand,
)


def _build_bounds(
    term_ptr: np.ndarray, post_weights: np.ndarray, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    term_count: int = int(term_ptr.shape[0] - 1)
    term_max = np.zeros((term_count,), dtype=np.float32)
    block_ptr = np.zeros((term_count,), dtype=np.int64)
    block_max_list: list[float] = []
    for term_id in range(term_count):
        start = int(term_ptr[term_id])
        end = int(term_ptr[term_id + 1])
        term_weights = post_weights[start:end]
        if term_weights.size > 0:
            term_max[term_id] = float(term_weights.max())
        block_ptr[term_id] = len(block_max_list)
        if term_weights.size == 0:
            continue
        block_count = (term_weights.size + block_size - 1) // block_size
        for block_idx in range(block_count):
            block_start = start + block_idx * block_size
            block_end = min(start + (block_idx + 1) * block_size, end)
            block_max_list.append(float(post_weights[block_start:block_end].max()))
    block_max = np.asarray(block_max_list, dtype=np.float32)
    return term_max, block_ptr, block_max


class WandBmwScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.term_ptr = np.array([0, 3, 6, 9, 9], dtype=np.int64)
        self.post_doc_ids = np.array([0, 2, 4, 1, 2, 3, 0, 3, 5], dtype=np.int32)
        self.post_weights = np.array(
            [0.5, 1.0, 0.7, 0.9, 0.3, 0.4, 0.2, 0.8, 0.6], dtype=np.float32
        )
        self.block_size = 2
        self.term_max, self.block_ptr, self.block_max = _build_bounds(
            self.term_ptr, self.post_weights, self.block_size
        )
        self.doc_count = 6

    def _score_full(
        self, q_indices: np.ndarray, q_values: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros((self.doc_count,), dtype=np.float32)
        seen = np.zeros((self.doc_count,), dtype=np.uint8)
        return score_query_postings(
            self.term_ptr,
            self.post_doc_ids,
            self.post_weights,
            q_indices,
            q_values,
            scores=scores,
            seen=seen,
            top_k=top_k,
        )

    def _score_wand(
        self, q_indices: np.ndarray, q_values: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros((self.doc_count,), dtype=np.float32)
        seen = np.zeros((self.doc_count,), dtype=np.uint8)
        return score_query_postings_wand(
            self.term_ptr,
            self.post_doc_ids,
            self.post_weights,
            self.term_max,
            q_indices,
            q_values,
            scores=scores,
            seen=seen,
            top_k=top_k,
        )

    def _score_bmw(
        self, q_indices: np.ndarray, q_values: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros((self.doc_count,), dtype=np.float32)
        seen = np.zeros((self.doc_count,), dtype=np.uint8)
        return score_query_postings_bmw(
            self.term_ptr,
            self.post_doc_ids,
            self.post_weights,
            self.term_max,
            self.block_max,
            self.block_ptr,
            q_indices,
            q_values,
            scores=scores,
            seen=seen,
            top_k=top_k,
            block_size=self.block_size,
        )

    def _assert_match_full(
        self, q_indices: np.ndarray, q_values: np.ndarray, top_k: int
    ) -> None:
        full_docs, full_scores = self._score_full(q_indices, q_values, top_k)
        wand_docs, wand_scores = self._score_wand(q_indices, q_values, top_k)
        bmw_docs, bmw_scores = self._score_bmw(q_indices, q_values, top_k)
        np.testing.assert_array_equal(wand_docs, full_docs)
        np.testing.assert_allclose(wand_scores, full_scores, rtol=0, atol=0)
        np.testing.assert_array_equal(bmw_docs, full_docs)
        np.testing.assert_allclose(bmw_scores, full_scores, rtol=0, atol=0)

    def test_small_queries_match_full(self) -> None:
        queries = [
            (np.array([0, 1, 2], dtype=np.int32), np.array([0.7, 1.1, 0.4], dtype=np.float32)),
            (np.array([1, 2], dtype=np.int32), np.array([1.5, 0.3], dtype=np.float32)),
            (np.array([0, 2, 3], dtype=np.int32), np.array([0.5, 0.9, 2.0], dtype=np.float32)),
            (np.array([2], dtype=np.int32), np.array([0.8], dtype=np.float32)),
        ]
        for q_indices, q_values in queries:
            self._assert_match_full(q_indices, q_values, top_k=3)

    def test_random_queries_match_full(self) -> None:
        rng = np.random.default_rng(123)
        for _ in range(10):
            term_count = rng.integers(1, 4)
            q_indices = np.sort(rng.choice(4, size=int(term_count), replace=False)).astype(
                np.int32
            )
            q_values = rng.uniform(0.1, 2.0, size=q_indices.shape[0]).astype(np.float32)
            self._assert_match_full(q_indices, q_values, top_k=3)

    def test_empty_query(self) -> None:
        q_indices = np.zeros((0,), dtype=np.int32)
        q_values = np.zeros((0,), dtype=np.float32)
        self._assert_match_full(q_indices, q_values, top_k=3)

    def test_zero_topk(self) -> None:
        q_indices = np.array([0, 1], dtype=np.int32)
        q_values = np.array([0.5, 0.9], dtype=np.float32)
        self._assert_match_full(q_indices, q_values, top_k=0)


if __name__ == "__main__":
    unittest.main()
