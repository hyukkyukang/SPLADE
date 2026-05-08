import unittest

from src.metric.retrieval import RetrievalMetrics, resolve_metric_families


class RetrievalMetricsTest(unittest.TestCase):
    def test_resolve_metric_families_keeps_unique_order(self) -> None:
        self.assertEqual(
            resolve_metric_families(["mrr", "Recall", "mrr"]),
            ["MRR", "Recall"],
        )

    def test_retrieval_metrics_can_limit_metric_families(self) -> None:
        metrics = RetrievalMetrics(k_list=[1, 5], metric_families=["MRR", "Recall"])
        self.assertEqual(
            set(metrics.keys()),
            {"MRR_1", "MRR_5", "Recall_1", "Recall_5"},
        )


if __name__ == "__main__":
    unittest.main()
