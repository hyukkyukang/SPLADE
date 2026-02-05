import unittest

import numpy as np

from src.indexing.sparse_index import (
    compute_term_and_block_max,
    score_query_postings,
    score_query_postings_wand,
)
from src.model.pl_module.utils import prepare_score_buffers


class WandScoringTest(unittest.TestCase):
    def _build_index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        term_ptr = np.array([0, 3, 5, 7], dtype=np.int64)
        post_doc_ids = np.array([0, 2, 3, 1, 2, 0, 3], dtype=np.int32)
        post_weights = np.array(
            [0.5, 0.2, 0.9, 0.4, 0.3, 0.1, 0.8], dtype=np.float32
        )
        return term_ptr, post_doc_ids, post_weights

    def test_wand_matches_exact(self) -> None:
        term_ptr, post_doc_ids, post_weights = self._build_index()
        term_max, block_max, block_ptr = compute_term_and_block_max(
            term_ptr, post_weights, block_size=2
        )
        q_indices = np.array([0, 1, 2], dtype=np.int32)
        q_values = np.array([1.0, 0.5, 0.2], dtype=np.float32)

        scores, seen = prepare_score_buffers(doc_count=4)
        exact_docs, exact_scores = score_query_postings(
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
            block_max,
            block_ptr,
            q_indices,
            q_values,
            scores=wand_buffer_scores,
            seen=wand_buffer_seen,
            top_k=3,
            block_size=2,
        )

        np.testing.assert_array_equal(wand_docs, exact_docs)
        np.testing.assert_allclose(wand_scores, exact_scores)


if __name__ == "__main__":
    unittest.main()
