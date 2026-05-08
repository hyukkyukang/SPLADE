import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import lightning as L
import torch

from src.model.pl_module.dense_eval import (
    DenseRetrievalEvalLightningModule,
    _merge_gathered_dense_query_payloads,
)


class DenseEvalRankZeroSearchTest(unittest.TestCase):
    def test_merge_gathered_dense_query_payloads_concatenates_rank_payloads(self) -> None:
        query_reps, query_ids, relevance_judgments = _merge_gathered_dense_query_payloads(
            [
                {
                    "query_reps": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                    "query_ids": ["q1"],
                    "relevance_judgments": [{"d1": 1}],
                },
                {
                    "query_reps": torch.tensor([[3.0, 4.0]], dtype=torch.float32),
                    "query_ids": ["q2"],
                    "relevance_judgments": [{"d2": 1, "d3": 0}],
                },
            ],
            embedding_dim=2,
        )

        self.assertTrue(
            torch.equal(
                query_reps,
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            )
        )
        self.assertEqual(query_ids, ["q1", "q2"])
        self.assertEqual(relevance_judgments, [{"d1": 1}, {"d2": 1, "d3": 0}])

    def test_merge_gathered_dense_query_payloads_rejects_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "ids length does not match"):
            _merge_gathered_dense_query_payloads(
                [
                    {
                        "query_reps": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                        "query_ids": [],
                        "relevance_judgments": [],
                    }
                ],
                embedding_dim=2,
            )

    def test_on_test_start_skips_index_setup_for_nonzero_rank(self) -> None:
        module = DenseRetrievalEvalLightningModule.__new__(DenseRetrievalEvalLightningModule)
        L.LightningModule.__init__(module)
        module.metric_collection = MagicMock()
        module.model = MagicMock()
        module._retrieval_helper = MagicMock()
        module._pending_query_reps = [torch.ones((1, 2), dtype=torch.float32)]
        module._pending_query_ids = ["q1"]
        module._pending_relevance_judgments = [{"d1": 1}]
        module._local_query_offset = 5
        module._trainer = SimpleNamespace(world_size=8, is_global_zero=False)

        module.on_test_start()

        self.assertEqual(module._local_query_offset, 0)
        self.assertEqual(module._pending_query_reps, [])
        self.assertEqual(module._pending_query_ids, [])
        self.assertEqual(module._pending_relevance_judgments, [])
        module.metric_collection.reset.assert_called_once_with()
        module.metric_collection.to.assert_called_once_with(torch.device("cpu"))
        module.model.eval.assert_called_once_with()
        module._retrieval_helper.setup.assert_not_called()

    def test_on_test_start_sets_up_index_on_global_zero(self) -> None:
        module = DenseRetrievalEvalLightningModule.__new__(DenseRetrievalEvalLightningModule)
        L.LightningModule.__init__(module)
        module.metric_collection = MagicMock()
        module.model = MagicMock()
        module._retrieval_helper = MagicMock()
        module._pending_query_reps = []
        module._pending_query_ids = []
        module._pending_relevance_judgments = []
        module._local_query_offset = 0
        module._trainer = SimpleNamespace(world_size=8, is_global_zero=True)

        module.on_test_start()

        module._retrieval_helper.setup.assert_called_once_with(device_index=None)


if __name__ == "__main__":
    unittest.main()
