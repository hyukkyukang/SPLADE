import unittest

import torch

from src.model.pl_module.metrics_service import TrainingMetricsService


class _DummyModule:
    def __init__(self, *, global_step: int) -> None:
        self.global_step = int(global_step)
        self.logged_names: list[str] = []

    def log(self, name: str, value: torch.Tensor, **kwargs: object) -> None:
        _ = value, kwargs
        self.logged_names.append(name)


class TrainingMetricsServiceTest(unittest.TestCase):
    def test_step_only_metrics_respect_interval(self) -> None:
        service = TrainingMetricsService(step_only_metric_log_interval=3)
        module = _DummyModule(global_step=2)
        service.log_training_metrics(
            module, {"loss": torch.tensor(1.0), "q_active_dims": torch.tensor(10.0)}
        )
        self.assertIn("train_loss", module.logged_names)
        self.assertNotIn("train_q_active_dims", module.logged_names)

    def test_step_only_metrics_logged_on_interval_boundary(self) -> None:
        service = TrainingMetricsService(step_only_metric_log_interval=3)
        module = _DummyModule(global_step=3)
        service.log_training_metrics(
            module, {"loss": torch.tensor(1.0), "q_active_dims": torch.tensor(10.0)}
        )
        self.assertIn("train_loss", module.logged_names)
        self.assertIn("train_q_active_dims", module.logged_names)

    def test_validation_diagnostics_respect_interval(self) -> None:
        service = TrainingMetricsService(validation_diagnostics_log_interval=3)
        self.assertFalse(service.should_compute_validation_diagnostics(batch_idx=2))
        self.assertTrue(service.should_compute_validation_diagnostics(batch_idx=3))

    def test_validation_diagnostics_can_be_disabled(self) -> None:
        service = TrainingMetricsService(validation_diagnostics_enabled=False)
        self.assertFalse(service.should_compute_validation_diagnostics(batch_idx=0))


if __name__ == "__main__":
    unittest.main()
