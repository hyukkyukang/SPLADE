import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from script.train import (
    _build_lightning_loggers,
    _build_mlflow_tags,
    _resolve_mlflow_logged_model_name,
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
            "model": {
                "name": "splade_model_unit",
                "type": "splade",
                "huggingface_name": "distilbert-base-uncased",
            },
            "train_dataset": {
                "name": "msmarco_train",
                "type": "custom",
                "split": "train",
                "hf_name": "sentence-transformers/msmarco-hard-negatives",
                "hf_subset": "triplet-10000",
                "hf_split": "train",
                "beir_dataset": None,
            },
            "val_dataset": {
                "name": "msmarco_val",
                "type": "custom",
                "split": "validation",
                "hf_name": "sentence-transformers/msmarco-hard-negatives",
                "hf_subset": "triplet-10000",
                "hf_split": "train",
                "beir_dataset": None,
            },
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
                    "system_metrics_sampling_interval": 15,
                    "system_metrics_samples_before_logging": 1,
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

    def test_resolve_mlflow_logged_model_name_prefers_hf_leaf(self) -> None:
        cfg = _build_cfg(log_dir="/tmp/splade", tag="run")
        cfg.model.huggingface_name = "answerdotai/ModernBERT-base"
        self.assertEqual(
            _resolve_mlflow_logged_model_name(cfg.model), "ModernBERT-base"
        )

    def test_resolve_mlflow_logged_model_name_canonicalizes_known_backbones(self) -> None:
        cfg = _build_cfg(log_dir="/tmp/splade", tag="run")
        cfg.model.huggingface_name = "distilbert-base-uncased"
        self.assertEqual(
            _resolve_mlflow_logged_model_name(cfg.model), "DistilBERT-base-uncased"
        )

        cfg.model.huggingface_name = "outputs/model_creation/embeddinggemma_splade/hf_backbone"
        cfg.model.name = "splade_v2_pp_embeddinggemma_300m_lsr"
        self.assertEqual(
            _resolve_mlflow_logged_model_name(cfg.model), "EmbeddingGemma-300M"
        )

        cfg.model.huggingface_name = "Luyu/co-condenser-marco"
        self.assertEqual(
            _resolve_mlflow_logged_model_name(cfg.model), "CoCondenser-Marco"
        )

        cfg.model.huggingface_name = "/data/model/trained_anna_base_hf_pruned_hangul"
        self.assertEqual(_resolve_mlflow_logged_model_name(cfg.model), "ANNA-base")

        cfg.model.huggingface_name = "/data/model/anna_large_hf"
        self.assertEqual(_resolve_mlflow_logged_model_name(cfg.model), "ANNA-large")

    def test_resolve_mlflow_logged_model_name_falls_back_to_model_name(self) -> None:
        cfg = _build_cfg(log_dir="/tmp/splade", tag="run")
        cfg.model.huggingface_name = None
        self.assertEqual(_resolve_mlflow_logged_model_name(cfg.model), "splade_model_unit")

    def test_build_lightning_loggers_skips_mlflow_for_debug_tag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="splade_logging_") as tmpdir:
            cfg = _build_cfg(log_dir=tmpdir, tag="debug")
            with patch("script.train.MLFlowLogger", side_effect=AssertionError):
                loggers, managed_run_id = _build_lightning_loggers(cfg, cfg.training)
        self.assertEqual(len(loggers), 1)
        self.assertIsInstance(loggers[0], CSVLogger)
        self.assertIsNone(managed_run_id)

    def test_build_lightning_loggers_builds_mlflow_for_non_debug(self) -> None:
        _FakeMLFlowLogger.instances.clear()
        system_metrics_env_value: str | None = None
        system_metrics_interval_env_value: str | None = None
        system_metrics_samples_env_value: str | None = None
        with tempfile.TemporaryDirectory(prefix="splade_logging_") as tmpdir:
            cfg = _build_cfg(log_dir=tmpdir, tag="run_tag")
            cfg.training.mlflow.log_model = True
            fake_run = SimpleNamespace(
                info=SimpleNamespace(experiment_id="42"),
                inputs=SimpleNamespace(dataset_inputs=[]),
                outputs=SimpleNamespace(model_outputs=[]),
            )
            mlflow_client_instance: Mock = Mock()
            mlflow_client_instance.get_run.return_value = fake_run
            mlflow_client_instance.create_logged_model.return_value = SimpleNamespace(
                model_id="m-1234"
            )
            with patch("script.train.MLFlowLogger", _FakeMLFlowLogger), patch.dict(
                os.environ, {}, clear=False
            ), patch("src.utils.mlflow_utils.mlflow.active_run", return_value=None), patch(
                "src.utils.mlflow_utils.mlflow.start_run"
            ) as start_run, patch(
                "script.train.warn_if_mlflow_gpu_metrics_unavailable"
            ) as warn_gpu_metrics, patch(
                "script.train.MlflowClient", return_value=mlflow_client_instance
            ):
                loggers, managed_run_id = _build_lightning_loggers(cfg, cfg.training)
                system_metrics_env_value = os.environ.get(
                    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"
                )
                system_metrics_interval_env_value = os.environ.get(
                    "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"
                )
                system_metrics_samples_env_value = os.environ.get(
                    "MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING"
                )

        self.assertEqual(len(loggers), 2)
        self.assertIsInstance(loggers[0], CSVLogger)
        self.assertEqual(managed_run_id, "unit-test-run-id")
        fake_logger = _FakeMLFlowLogger.instances[-1]
        self.assertEqual(fake_logger.kwargs["experiment_name"], "Train-SPLADE")
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
        self.assertEqual(system_metrics_interval_env_value, "15")
        self.assertEqual(system_metrics_samples_env_value, "1")
        self.assertEqual(len(fake_logger.experiment.artifacts), 0)
        warn_gpu_metrics.assert_called_once()
        start_run.assert_called_once_with(
            run_id="unit-test-run-id", log_system_metrics=True
        )
        mlflow_client_instance.log_inputs.assert_called_once()
        self.assertEqual(
            mlflow_client_instance.log_inputs.call_args.kwargs["run_id"],
            "unit-test-run-id",
        )
        logged_datasets = mlflow_client_instance.log_inputs.call_args.kwargs["datasets"]
        self.assertEqual(len(logged_datasets), 2)
        logged_dataset_names = {dataset_input.dataset.name for dataset_input in logged_datasets}
        self.assertEqual(logged_dataset_names, {"msmarco_train", "msmarco_val"})
        mlflow_client_instance.create_logged_model.assert_called_once()
        self.assertEqual(
            mlflow_client_instance.create_logged_model.call_args.kwargs["experiment_id"],
            "42",
        )
        self.assertEqual(
            mlflow_client_instance.create_logged_model.call_args.kwargs["source_run_id"],
            "unit-test-run-id",
        )
        self.assertEqual(
            mlflow_client_instance.create_logged_model.call_args.kwargs["name"],
            "DistilBERT-base-uncased",
        )
        mlflow_client_instance.log_outputs.assert_called_once()
        self.assertEqual(
            mlflow_client_instance.log_outputs.call_args.kwargs["run_id"],
            "unit-test-run-id",
        )


if __name__ == "__main__":
    unittest.main()
