import unittest

from omegaconf import OmegaConf

from src.utils.trainer import resolve_validation_check_interval


class ResolveValidationCheckIntervalTest(unittest.TestCase):
    def test_uses_optimizer_step_interval_when_configured(self) -> None:
        cfg = OmegaConf.create(
            {
                "val_check_interval": 5000,
                "val_check_interval_optimizer_steps": 5000,
                "grad_accumulation": 6,
            }
        )
        self.assertEqual(resolve_validation_check_interval(cfg), 30000)

    def test_preserves_batch_interval_when_optimizer_step_interval_is_null(self) -> None:
        cfg = OmegaConf.create(
            {
                "val_check_interval": 1.0,
                "val_check_interval_optimizer_steps": None,
                "grad_accumulation": 6,
            }
        )
        self.assertEqual(float(resolve_validation_check_interval(cfg)), 1.0)

    def test_rejects_non_positive_optimizer_step_interval(self) -> None:
        cfg = OmegaConf.create(
            {
                "val_check_interval": 5000,
                "val_check_interval_optimizer_steps": 0,
                "grad_accumulation": 1,
            }
        )
        with self.assertRaises(ValueError):
            resolve_validation_check_interval(cfg)


if __name__ == "__main__":
    unittest.main()
