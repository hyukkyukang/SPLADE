import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import DictConfig, OmegaConf

from script.evaluate_mteb import _log_mlflow_run_datasets_and_model


def _build_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "name": "splade_v2_pp",
                "type": "splade",
                "huggingface_name": "naver/splade-v3",
            },
            "nanobeir": {
                "datasets": ["msmarco", "nfcorpus"],
                "batch_size": 16,
                "max_seq_length": 256,
                "use_huggingface_model": True,
            },
        }
    )


class EvaluateMtebMlflowLoggingTest(unittest.TestCase):
    def test_logs_dataset_inputs_and_model_outputs(self) -> None:
        cfg: DictConfig = _build_cfg()
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

        with patch(
            "script.evaluate_mteb.MlflowClient", return_value=mlflow_client_instance
        ):
            _log_mlflow_run_datasets_and_model(
                cfg=cfg,
                run_id="unit-run-id",
                tracking_uri="http://127.0.0.1:5000",
                model_source="naver/splade-v3",
                model_source_kind="huggingface",
            )

        mlflow_client_instance.log_inputs.assert_called_once()
        logged_datasets = mlflow_client_instance.log_inputs.call_args.kwargs["datasets"]
        self.assertEqual(len(logged_datasets), 2)
        logged_dataset_names = {dataset_input.dataset.name for dataset_input in logged_datasets}
        self.assertEqual(logged_dataset_names, {"msmarco", "nfcorpus"})

        mlflow_client_instance.create_logged_model.assert_called_once()
        self.assertEqual(
            mlflow_client_instance.create_logged_model.call_args.kwargs["experiment_id"],
            "42",
        )
        self.assertEqual(
            mlflow_client_instance.create_logged_model.call_args.kwargs["source_run_id"],
            "unit-run-id",
        )
        self.assertEqual(
            mlflow_client_instance.create_logged_model.call_args.kwargs["name"],
            "splade-v3",
        )

        mlflow_client_instance.log_outputs.assert_called_once()
        self.assertEqual(
            mlflow_client_instance.log_outputs.call_args.kwargs["run_id"],
            "unit-run-id",
        )

    def test_skips_inputs_and_outputs_when_already_logged(self) -> None:
        cfg: DictConfig = _build_cfg()
        fake_run = SimpleNamespace(
            info=SimpleNamespace(experiment_id="42"),
            inputs=SimpleNamespace(dataset_inputs=[object()]),
            outputs=SimpleNamespace(model_outputs=[object()]),
        )
        mlflow_client_instance: Mock = Mock()
        mlflow_client_instance.get_run.return_value = fake_run

        with patch(
            "script.evaluate_mteb.MlflowClient", return_value=mlflow_client_instance
        ):
            _log_mlflow_run_datasets_and_model(
                cfg=cfg,
                run_id="unit-run-id",
                tracking_uri="http://127.0.0.1:5000",
                model_source="naver/splade-v3",
                model_source_kind="huggingface",
            )

        mlflow_client_instance.log_inputs.assert_not_called()
        mlflow_client_instance.create_logged_model.assert_not_called()
        mlflow_client_instance.log_outputs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
