import unittest

import torch
from omegaconf import OmegaConf

from src.metric.validation_retrieval import ValidationRetrievalMetrics
from src.model.pl_module.validation_service import ValidationMetricsAccumulator


class ValidationRetrievalMetricsTest(unittest.TestCase):
    def test_tied_scores_use_tie_break_not_original_doc_index(self) -> None:
        metric = ValidationRetrievalMetrics(k_list=[1, 5, 10, 100], tie_break_seed=0)
        scores = torch.zeros(100, dtype=torch.float32)
        targets = torch.zeros(100, dtype=torch.float32)
        targets[0] = 1.0
        indexes = torch.zeros(100, dtype=torch.long)

        # Place positive doc at the end of tied ordering.
        tie_break = torch.arange(100, dtype=torch.long)
        tie_break[0] = 10_000

        metric.append(scores, targets, indexes, tie_break)
        self.assertTrue(metric.gather(world_size=1, all_gather_fn=None))
        values = metric.compute()

        self.assertAlmostEqual(float(values["MRR_1"]), 0.0, places=6)
        self.assertAlmostEqual(float(values["MRR_5"]), 0.0, places=6)
        self.assertAlmostEqual(float(values["MRR_10"]), 0.0, places=6)
        self.assertAlmostEqual(float(values["MRR_100"]), 0.01, places=6)
        self.assertAlmostEqual(float(values["Recall_1"]), 0.0, places=6)
        self.assertAlmostEqual(float(values["Recall_100"]), 1.0, places=6)

    def test_mrr_and_recall_are_monotonic_in_k(self) -> None:
        metric = ValidationRetrievalMetrics(k_list=[1, 3, 5, 10], tie_break_seed=0)
        scores = torch.zeros(10, dtype=torch.float32)
        targets = torch.zeros(10, dtype=torch.float32)
        targets[7] = 1.0
        indexes = torch.zeros(10, dtype=torch.long)

        # Positive appears at rank 4 in tied ordering.
        tie_break = torch.tensor([0, 1, 2, 7, 3, 4, 5, 6, 8, 9], dtype=torch.long)

        metric.append(scores, targets, indexes, tie_break)
        self.assertTrue(metric.gather(world_size=1, all_gather_fn=None))
        values = metric.compute()

        mrr_values = [float(values[f"MRR_{k}"]) for k in [1, 3, 5, 10]]
        recall_values = [float(values[f"Recall_{k}"]) for k in [1, 3, 5, 10]]

        self.assertLessEqual(mrr_values[0], mrr_values[1])
        self.assertLessEqual(mrr_values[1], mrr_values[2])
        self.assertLessEqual(mrr_values[2], mrr_values[3])
        self.assertLessEqual(recall_values[0], recall_values[1])
        self.assertLessEqual(recall_values[1], recall_values[2])
        self.assertLessEqual(recall_values[2], recall_values[3])

    def test_negative_scores_are_not_filtered_out(self) -> None:
        metric = ValidationRetrievalMetrics(k_list=[1], tie_break_seed=0)
        scores = torch.full((3,), -1.0, dtype=torch.float32)
        targets = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        indexes = torch.zeros(3, dtype=torch.long)
        tie_break = torch.tensor([0, 1, 2], dtype=torch.long)

        metric.append(scores, targets, indexes, tie_break)
        self.assertTrue(metric.gather(world_size=1, all_gather_fn=None))
        values = metric.compute()

        self.assertAlmostEqual(float(values["MRR_1"]), 1.0, places=6)
        self.assertAlmostEqual(float(values["Recall_1"]), 1.0, places=6)


class ValidationMetricsAccumulatorIntegrationTest(unittest.TestCase):
    def test_custom_backend_uses_deterministic_tie_break(self) -> None:
        cfg = OmegaConf.create(
            {
                "enabled": True,
                "backend": "custom",
                "tie_break_seed": 0,
                "k_list": [1, 5, 10, 100],
            }
        )
        accumulator = ValidationMetricsAccumulator(dataset_name="", metrics_cfg=cfg)
        accumulator.on_validation_start(torch.device("cpu"))

        pairwise_scores = torch.zeros((1, 100), dtype=torch.float32)
        pos_mask = torch.zeros((1, 100), dtype=torch.bool)
        pos_mask[0, 0] = True
        doc_mask = torch.ones((1, 100), dtype=torch.bool)

        accumulator.append_batch(
            pairwise_scores=pairwise_scores,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            world_size=1,
            global_rank=0,
        )

        has_data, metrics = accumulator.finalize_epoch(world_size=1, all_gather_fn=None)
        self.assertTrue(has_data)

        tie_break = accumulator._build_tie_break_values(
            global_query_idx=0,
            local_doc_indexes=torch.arange(100, dtype=torch.long),
            device=torch.device("cpu"),
        )
        rank_order = torch.argsort(tie_break)
        positive_rank = int((rank_order == 0).nonzero(as_tuple=False).item()) + 1

        expected_mrr_1 = 1.0 if positive_rank <= 1 else 0.0
        expected_mrr_5 = 1.0 / float(positive_rank) if positive_rank <= 5 else 0.0
        expected_mrr_10 = 1.0 / float(positive_rank) if positive_rank <= 10 else 0.0
        expected_mrr_100 = 1.0 / float(positive_rank)
        expected_recall_1 = 1.0 if positive_rank <= 1 else 0.0
        expected_recall_5 = 1.0 if positive_rank <= 5 else 0.0
        expected_recall_10 = 1.0 if positive_rank <= 10 else 0.0
        expected_recall_100 = 1.0

        self.assertAlmostEqual(float(metrics["val_MRR_1"]), expected_mrr_1, places=6)
        self.assertAlmostEqual(float(metrics["val_MRR_5"]), expected_mrr_5, places=6)
        self.assertAlmostEqual(float(metrics["val_MRR_10"]), expected_mrr_10, places=6)
        self.assertAlmostEqual(float(metrics["val_MRR_100"]), expected_mrr_100, places=6)
        self.assertAlmostEqual(
            float(metrics["val_Recall_1"]), expected_recall_1, places=6
        )
        self.assertAlmostEqual(
            float(metrics["val_Recall_5"]), expected_recall_5, places=6
        )
        self.assertAlmostEqual(
            float(metrics["val_Recall_10"]), expected_recall_10, places=6
        )
        self.assertAlmostEqual(
            float(metrics["val_Recall_100"]), expected_recall_100, places=6
        )

    def test_custom_backend_flattens_batch_once(self) -> None:
        cfg = OmegaConf.create(
            {
                "enabled": True,
                "backend": "custom",
                "tie_break_seed": 0,
                "k_list": [1, 5],
            }
        )
        accumulator = ValidationMetricsAccumulator(dataset_name="", metrics_cfg=cfg)
        accumulator.on_validation_start(torch.device("cpu"))

        pairwise_scores = torch.tensor(
            [
                [3.0, 1.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        pos_mask = torch.tensor(
            [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, False, False],
            ]
        )
        doc_mask = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
                [False, False, False, False],
            ]
        )

        accumulator.append_batch(
            pairwise_scores=pairwise_scores,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            world_size=1,
            global_rank=0,
        )

        metric_collection = accumulator._metric_collection
        self.assertIsInstance(metric_collection, ValidationRetrievalMetrics)
        self.assertEqual(len(metric_collection._accumulated_preds), 1)
        self.assertTrue(
            torch.equal(
                metric_collection._accumulated_preds[0],
                torch.tensor([3.0, 1.0, 2.0, 2.0, 2.0], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                metric_collection._accumulated_targets[0],
                torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                metric_collection._accumulated_indexes[0],
                torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
            )
        )
        expected_tie_break = torch.cat(
            [
                accumulator._build_tie_break_values(
                    global_query_idx=0,
                    local_doc_indexes=torch.tensor([0, 1], dtype=torch.long),
                    device=torch.device("cpu"),
                ),
                accumulator._build_tie_break_values(
                    global_query_idx=1,
                    local_doc_indexes=torch.tensor([0, 1, 2], dtype=torch.long),
                    device=torch.device("cpu"),
                ),
            ]
        )
        self.assertTrue(
            torch.equal(metric_collection._accumulated_tie_break[0], expected_tie_break)
        )


if __name__ == "__main__":
    unittest.main()
