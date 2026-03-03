import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from script.train import (
    _build_lightning_loggers,
    _build_mlflow_tags,
    _resolve_mlflow_run_name,
)


class _FakeMLFlowLogger:
    instances: list["_FakeMLFlowLogger"] = []

    def __init__(self, **kwargs: Any) -> None:
        class _Experiment:
            def __init__(self) -> None:
                self.artifacts: list[dict[str, Any]] = []

            def log_artifact(
                self, run_id: str, local_path: str, artifact_path: str | None = None
            ) -> None:
                self.artifacts.append(
                    {
                        "run_id": run_id,
                        "local_path": local_path,
                        "artifact_path": artifact_path,
                    }
                )

        self.kwargs: dict[str, Any] = kwargs
        self.logged_hparams: dict[str, Any] | None = None
        self.run_id: str | None = "unit-test-run-id"
        self.experiment = _Experiment()
        _FakeMLFlowLogger.instances.append(self)

    def log_hyperparams(self, params: Any) -> None:
        self.logged_hparams = params


def _build_cfg(
    *,
    log_dir: str,
    tag: str | None,
    run_name: str | None = None,
    tags: dict[str, Any] | None = None,
) -> DictConfig:
    return OmegaConf.create(
        {
            "tag": tag,
            "log_dir": log_dir,
            "training": {
                "name": "splade_unit",
                "mlflow": {
                    "enabled": True,
                    "experiment_name": "splade_exp",
                    "tracking_uri": "http://127.0.0.1:5000",
                    "run_name": run_name,
                    "artifact_location": None,
                    "insecure_tls": None,
                    "server_cert_path": None,
                    "client_cert_path": None,
                    "system_metrics_enabled": True,
                    "tags": {} if tags is None else tags,
                    "log_model": False,
                    "save_dir": log_dir,
                    "prefix": "",
                },
            },
        }
    )


class TrainLoggerConfigTest(unittest.TestCase):
    def test_resolve_mlflow_run_name_precedence(self) -> None:
        cfg = _build_cfg(log_dir="/tmp/splade", tag="tag-run", run_name="explicit-run")
        resolved = _resolve_mlflow_run_name(cfg.training, "tag-run")
        self.assertEqual(resolved, "explicit-run")

        cfg.training.mlflow.run_name = None
        resolved_from_tag = _resolve_mlflow_run_name(cfg.training, "tag-run")
        self.assertEqual(resolved_from_tag, "tag-run")

        resolved_from_training_name = _resolve_mlflow_run_name(cfg.training, None)
        self.assertEqual(resolved_from_training_name, "splade_unit")

    def test_build_mlflow_tags_adds_metadata(self) -> None:
        cfg = _build_cfg(
            log_dir="/tmp/splade",
            tag="exp_a",
            tags={"team": "search", "priority": 1, "skip_me": None},
        )
        tags = _build_mlflow_tags(cfg, cfg.training, "exp_a")
        self.assertEqual(tags["team"], "search")
        self.assertEqual(tags["priority"], "1")
        self.assertEqual(tags["training_name"], "splade_unit")
        self.assertEqual(tags["log_dir"], "/tmp/splade")
        self.assertEqual(tags["tag"], "exp_a")
        self.assertNotIn("skip_me", tags)

    def test_build_lightning_loggers_skips_mlflow_for_debug_tag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="splade_logging_") as tmpdir:
            cfg = _build_cfg(log_dir=tmpdir, tag="debug")
            with patch("script.train.MLFlowLogger", side_effect=AssertionError):
                loggers = _build_lightning_loggers(cfg, cfg.training)
        self.assertEqual(len(loggers), 1)
        self.assertIsInstance(loggers[0], CSVLogger)

    def test_build_lightning_loggers_builds_mlflow_for_non_debug(self) -> None:
        _FakeMLFlowLogger.instances.clear()
        system_metrics_env_value: str | None = None
        with tempfile.TemporaryDirectory(prefix="splade_logging_") as tmpdir:
            cfg = _build_cfg(log_dir=tmpdir, tag="run_tag")
            cfg.training.mlflow.log_model = True
            with patch("script.train.MLFlowLogger", _FakeMLFlowLogger), patch.dict(
                os.environ, {}, clear=False
            ):
                loggers = _build_lightning_loggers(cfg, cfg.training)
                system_metrics_env_value = os.environ.get(
                    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"
                )

        self.assertEqual(len(loggers), 2)
        self.assertIsInstance(loggers[0], CSVLogger)
        fake_logger = _FakeMLFlowLogger.instances[-1]
        self.assertEqual(fake_logger.kwargs["experiment_name"], "splade_exp")
        self.assertEqual(fake_logger.kwargs["run_name"], "run_tag")
        self.assertEqual(
            fake_logger.kwargs["tracking_uri"], "http://127.0.0.1:5000"
        )
        self.assertEqual(fake_logger.kwargs["save_dir"], tmpdir)
        self.assertFalse(fake_logger.kwargs["log_model"])
        self.assertIsNotNone(fake_logger.logged_hparams)
        assert fake_logger.logged_hparams is not None
        self.assertEqual(fake_logger.logged_hparams["training"]["name"], "splade_unit")
        self.assertEqual(system_metrics_env_value, "true")
        self.assertEqual(len(fake_logger.experiment.artifacts), 0)


if __name__ == "__main__":
    unittest.main()
