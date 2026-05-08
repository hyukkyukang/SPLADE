import unittest
from unittest.mock import Mock, patch

import torch

from import_stubs import (
    install_fake_pytorch_lightning_utilities,
    install_fake_sentence_transformers,
)

install_fake_pytorch_lightning_utilities()
install_fake_sentence_transformers()

from src.utils.mteb_search import (
    MTEBSparseRetrievalAdapter,
    _HeartbeatTracker,
    _concat_sparse_csr_rows,
    _merge_topk_state,
)


class MTEBSearchStreamingTest(unittest.TestCase):
    def test_heartbeat_tracker_throttles_periodic_logs(self) -> None:
        logger = Mock()
        tracker = _HeartbeatTracker(logger=logger, interval_seconds=120.0)

        with patch(
            "src.utils.mteb_search.time.monotonic",
            side_effect=[0.0, 30.0, 121.0, 150.0],
        ):
            tracker.start(
                phase="index",
                task_name="NQ",
                label="docs",
                total=100,
                extras={"chunk_size": 50},
            )
            tracker.progress(
                phase="index",
                task_name="NQ",
                label="docs",
                processed=10,
                total=100,
                extras={"chunks": 0},
            )
            tracker.progress(
                phase="index",
                task_name="NQ",
                label="docs",
                processed=60,
                total=100,
                extras={"chunks": 1},
            )
            tracker.progress(
                phase="index",
                task_name="NQ",
                label="docs",
                processed=100,
                total=100,
                extras={"chunks": 2, "status": "complete"},
                force=True,
            )

        self.assertEqual(logger.info.call_count, 3)
        start_message = logger.info.call_args_list[0].args[0]
        periodic_message = logger.info.call_args_list[1].args[0]
        complete_message = logger.info.call_args_list[2].args[0]
        self.assertIn("Heartbeat[%s][%s] start", start_message)
        self.assertIn("Heartbeat[%s][%s] %s elapsed=%ss%s", periodic_message)
        self.assertIn("Heartbeat[%s][%s] %s elapsed=%ss%s", complete_message)
        self.assertEqual(tracker._phase_started_at, {})
        self.assertEqual(tracker._phase_last_logged_at, {})

    def test_concat_sparse_csr_rows_preserves_row_order(self) -> None:
        first = torch.tensor(
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
            dtype=torch.float32,
        ).to_sparse_csr()
        second = torch.tensor(
            [[0.0, 0.0, 4.0]],
            dtype=torch.float32,
        ).to_sparse_csr()

        merged = _concat_sparse_csr_rows([first, second])

        self.assertEqual(merged.layout, torch.sparse_csr)
        self.assertTrue(
            torch.equal(
                merged.to_dense(),
                torch.tensor(
                    [
                        [1.0, 0.0, 2.0],
                        [0.0, 3.0, 0.0],
                        [0.0, 0.0, 4.0],
                    ],
                    dtype=torch.float32,
                ),
            )
        )

    def test_merge_topk_state_keeps_global_best_scores(self) -> None:
        best_scores = torch.tensor(
            [[0.9, 0.7, 0.1], [0.8, 0.3, -float("inf")]],
            dtype=torch.float32,
        )
        best_positions = torch.tensor([[2, 4, 9], [1, 3, -1]], dtype=torch.long)
        chunk_scores = torch.tensor(
            [[0.95, 0.6, 0.2], [0.85, 0.4, 0.1]],
            dtype=torch.float32,
        )
        chunk_positions = torch.tensor([[5, 6, 7], [8, 10, 11]], dtype=torch.long)

        merged_scores, merged_positions = _merge_topk_state(
            best_scores,
            best_positions,
            chunk_scores,
            chunk_positions,
            top_k=3,
        )

        self.assertTrue(
            torch.equal(
                merged_scores,
                torch.tensor(
                    [[0.95, 0.9, 0.7], [0.85, 0.8, 0.4]],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                merged_positions,
                torch.tensor([[5, 2, 4], [8, 1, 10]], dtype=torch.long),
            )
        )

    def test_search_streams_corpus_chunks_and_returns_exact_topk(self) -> None:
        adapter = object.__new__(MTEBSparseRetrievalAdapter)
        adapter._corpus_embeddings = [
            torch.tensor(
                [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]],
                dtype=torch.float32,
            ).to_sparse_csr(),
            torch.tensor(
                [[0.0, 1.0, 4.0], [5.0, 0.0, 0.0]],
                dtype=torch.float32,
            ).to_sparse_csr(),
        ]
        adapter._corpus_chunk_sizes = [2, 2]
        adapter._flat_corpus_ids = ["doc-0", "doc-1", "doc-2", "doc-3"]
        adapter.score_device = torch.device("cpu")

        def _clear_index() -> None:
            adapter._corpus_embeddings.clear()
            adapter._corpus_chunk_sizes.clear()
            adapter._flat_corpus_ids.clear()

        adapter.clear_index = _clear_index
        def _encode_query_blocks(
            queries,
            *,
            task_metadata,
            encode_kwargs,
            num_proc,
        ):
            _ = queries, task_metadata, encode_kwargs, num_proc
            return [
                (
                    ["query-0", "query-1"],
                    torch.tensor(
                        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                        dtype=torch.float32,
                    ),
                )
            ]

        adapter._encode_query_blocks = _encode_query_blocks

        results = MTEBSparseRetrievalAdapter.search(
            adapter,
            queries=["unused"],
            task_metadata=None,
            hf_split="test",
            hf_subset="default",
            top_k=2,
            encode_kwargs={},
            num_proc=None,
        )

        self.assertEqual(
            results,
            {
                "query-0": {"doc-3": 5.0, "doc-2": 4.0},
                "query-1": {"doc-2": 5.0, "doc-1": 3.0},
            },
        )
        self.assertEqual(adapter._corpus_embeddings, [])
        self.assertEqual(adapter._corpus_chunk_sizes, [])
        self.assertEqual(adapter._flat_corpus_ids, [])


if __name__ == "__main__":
    unittest.main()
