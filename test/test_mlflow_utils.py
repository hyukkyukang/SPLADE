import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from src.utils.logging import get_logger
from src.utils.mlflow_utils import (
    configure_mlflow_tls,
    finish_mlflow_system_metrics_monitor,
    start_mlflow_system_metrics_monitor,
)


class MlflowUtilsTest(unittest.TestCase):
    def test_configure_mlflow_tls_sets_env(self) -> None:
        cfg = OmegaConf.create(
            {
                "insecure_tls": True,
                "server_cert_path": "/tmp/server.crt",
                "client_cert_path": "/tmp/client.pem",
                "system_metrics_enabled": False,
                "system_metrics_sampling_interval": 30,
                "system_metrics_samples_before_logging": 2,
            }
        )
        with patch.dict(os.environ, {}, clear=True):
            configure_mlflow_tls(cfg)
            self.assertEqual(os.environ["MLFLOW_TRACKING_INSECURE_TLS"], "true")
            self.assertEqual(
                os.environ["MLFLOW_TRACKING_SERVER_CERT_PATH"], "/tmp/server.crt"
            )
            self.assertEqual(
                os.environ["MLFLOW_TRACKING_CLIENT_CERT_PATH"], "/tmp/client.pem"
            )
            self.assertEqual(os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"], "false")
            self.assertEqual(os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"], "30")
            self.assertEqual(
                os.environ["MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING"], "2"
            )

    def test_configure_mlflow_tls_validates_positive_int(self) -> None:
        cfg = OmegaConf.create(
            {
                "system_metrics_sampling_interval": 0,
                "system_metrics_samples_before_logging": 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            configure_mlflow_tls(cfg)

    def test_start_system_metrics_monitor_starts_run(self) -> None:
        cfg = OmegaConf.create({"system_metrics_enabled": True})
        mlflow_logger = SimpleNamespace(run_id="run-1")
        logger = get_logger("test.mlflow_utils.start")
        with patch("src.utils.mlflow_utils.mlflow.active_run", return_value=None), patch(
            "src.utils.mlflow_utils.mlflow.start_run"
        ) as start_run:
            resolved = start_mlflow_system_metrics_monitor(
                mlflow_logger=mlflow_logger,
                mlflow_cfg=cfg,
                logger=logger,
                is_logging_rank_zero=lambda: True,
            )
        self.assertEqual(resolved, "run-1")
        start_run.assert_called_once_with(run_id="run-1", log_system_metrics=True)

    def test_start_system_metrics_monitor_skips_on_rank(self) -> None:
        cfg = OmegaConf.create({"system_metrics_enabled": True})
        mlflow_logger = SimpleNamespace(run_id="run-1")
        logger = get_logger("test.mlflow_utils.start.rank")
        with patch("src.utils.mlflow_utils.mlflow.start_run") as start_run:
            resolved = start_mlflow_system_metrics_monitor(
                mlflow_logger=mlflow_logger,
                mlflow_cfg=cfg,
                logger=logger,
                is_logging_rank_zero=lambda: False,
            )
        self.assertIsNone(resolved)
        start_run.assert_not_called()

    def test_finish_system_metrics_monitor_ends_matching_run(self) -> None:
        logger = get_logger("test.mlflow_utils.finish")
        active_run = SimpleNamespace(info=SimpleNamespace(run_id="run-1"))
        with patch("src.utils.mlflow_utils.mlflow.active_run", return_value=active_run), patch(
            "src.utils.mlflow_utils.mlflow.end_run"
        ) as end_run:
            finish_mlflow_system_metrics_monitor(
                run_id="run-1",
                status="FINISHED",
                logger=logger,
                is_logging_rank_zero=lambda: True,
            )
        end_run.assert_called_once_with(status="FINISHED")

    def test_finish_system_metrics_monitor_skips_mismatch(self) -> None:
        logger = get_logger("test.mlflow_utils.finish.mismatch")
        active_run = SimpleNamespace(info=SimpleNamespace(run_id="run-2"))
        with patch("src.utils.mlflow_utils.mlflow.active_run", return_value=active_run), patch(
            "src.utils.mlflow_utils.mlflow.end_run"
        ) as end_run, patch("src.utils.mlflow_utils.log_if_rank_zero") as log_warn:
            finish_mlflow_system_metrics_monitor(
                run_id="run-1",
                status="FAILED",
                logger=logger,
                is_logging_rank_zero=lambda: True,
            )
        end_run.assert_not_called()
        log_warn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
