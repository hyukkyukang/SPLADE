import unittest

import numpy as np

from src.index.sparse import (
    compute_term_and_block_max,
)
from src.search.scoring import (
    score_query_postings,
    score_query_postings_bmw,
    score_query_postings_wand,
)
from src.search.buffers import prepare_score_buffers


class WandScoringTest(unittest.TestCase):
    def _build_index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        term_ptr = np.array([0, 3, 5, 7], dtype=np.int64)
        post_doc_ids = np.array([0, 2, 3, 1, 2, 0, 3], dtype=np.int32)
        post_weights = np.array(
            [0.5, 0.2, 0.9, 0.4, 0.3, 0.1, 0.8], dtype=np.float32
        )
        return term_ptr, post_doc_ids, post_weights

    def test_wand_and_bmw_match_full(self) -> None:
        term_ptr, post_doc_ids, post_weights = self._build_index()
        term_max, block_max, block_ptr = compute_term_and_block_max(
            term_ptr, post_weights, block_size=2
        )
        q_indices = np.array([0, 1, 2], dtype=np.int32)
        q_values = np.array([1.0, 0.5, 0.2], dtype=np.float32)

        scores, seen = prepare_score_buffers(doc_count=4)
        full_docs, full_scores = score_query_postings(
            term_ptr,
            post_doc_ids,
            post_weights,
            q_indices,
            q_values,
            scores=scores,
            seen=seen,
            top_k=3,
        )

        wand_buffer_scores, wand_buffer_seen = prepare_score_buffers(doc_count=4)
        wand_docs, wand_scores = score_query_postings_wand(
            term_ptr,
            post_doc_ids,
            post_weights,
            term_max,
            q_indices,
            q_values,
            scores=wand_buffer_scores,
            seen=wand_buffer_seen,
            top_k=3,
        )

        bmw_buffer_scores, bmw_buffer_seen = prepare_score_buffers(doc_count=4)
        bmw_docs, bmw_scores = score_query_postings_bmw(
            term_ptr,
            post_doc_ids,
            post_weights,
            term_max,
            block_max,
            block_ptr,
            q_indices,
            q_values,
            scores=bmw_buffer_scores,
            seen=bmw_buffer_seen,
            top_k=3,
            block_size=2,
        )

        np.testing.assert_array_equal(wand_docs, full_docs)
        np.testing.assert_allclose(wand_scores, full_scores)
        np.testing.assert_array_equal(bmw_docs, full_docs)
        np.testing.assert_allclose(bmw_scores, full_scores)


if __name__ == "__main__":
    unittest.main()
