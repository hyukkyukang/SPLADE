import unittest
from typing import Any

import torch

from src.utils.logging import get_logger
from src.utils.metrics_io import resolve_training_style_validation_metrics


class _DummyTrainer:
    def __init__(self, callback_metrics: dict[str, Any]) -> None:
        self.callback_metrics = callback_metrics


class ValidationMetricsSemanticsTest(unittest.TestCase):
    def test_prefers_callback_metrics_when_available(self) -> None:
        trainer = _DummyTrainer(
            {
                "val_loss": torch.tensor(1.10),
                "val_MRR_10": torch.tensor(0.25),
                "train_loss": torch.tensor(3.49),
            }
        )
        validate_results = [{"val_loss": 1.20, "val_MRR_10": 0.20}]
        logger = get_logger("test.validation_metrics_semantics")

        resolved, raw, callback = resolve_training_style_validation_metrics(
            validate_results=validate_results,
            trainer=trainer,  # type: ignore[arg-type]
            logger=logger,
        )

        self.assertAlmostEqual(resolved["val_loss"], 1.10, places=6)
        self.assertAlmostEqual(resolved["val_MRR_10"], 0.25, places=6)
        self.assertAlmostEqual(raw["val_loss"], 1.20, places=6)
        self.assertIn("val_loss", callback)
        self.assertNotIn("train_loss", callback)

    def test_falls_back_to_validate_results_when_callback_missing(self) -> None:
        trainer = _DummyTrainer({})
        validate_results = [{"val_loss": 2.0, "val_Recall_10": 0.5}]
        logger = get_logger("test.validation_metrics_semantics.fallback")

        resolved, raw, callback = resolve_training_style_validation_metrics(
            validate_results=validate_results,
            trainer=trainer,  # type: ignore[arg-type]
            logger=logger,
        )

        self.assertEqual(resolved, raw)
        self.assertEqual(resolved["val_loss"], 2.0)
        self.assertEqual(resolved["val_Recall_10"], 0.5)
        self.assertEqual(callback, {})


if __name__ == "__main__":
    unittest.main()
