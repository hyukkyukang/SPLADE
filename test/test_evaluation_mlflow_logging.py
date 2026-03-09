import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import DictConfig, OmegaConf

from script.evaluation import _log_mlflow_run_datasets_and_model


def _build_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "evaluation": {
                "type": "retrieval",
            },
            "model": {
                "name": "splade_v2_pp",
                "type": "splade",
                "huggingface_name": "Luyu/co-condenser-marco",
            },
            "dataset": {
                "name": "msmarco",
                "type": "beir",
                "split": "test",
                "hf_name": "Hyukkyu/beir-msmarco",
                "hf_subset": None,
                "hf_split": "test",
                "beir_dataset": "msmarco",
                "qrels_hf_name": "Hyukkyu/beir-msmarco-qrels",
                "qrels_hf_subset": None,
                "qrels_hf_split": "validation",
            },
        }
    )


class EvaluationMlflowLoggingTest(unittest.TestCase):
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
            "script.evaluation.MlflowClient", return_value=mlflow_client_instance
        ):
            _log_mlflow_run_datasets_and_model(
                cfg=cfg,
                run_id="unit-run-id",
                tracking_uri="http://127.0.0.1:5000",
                model_source="/tmp/best.ckpt",
                model_source_kind="checkpoint",
            )

        mlflow_client_instance.log_inputs.assert_called_once()
        logged_datasets = mlflow_client_instance.log_inputs.call_args.kwargs["datasets"]
        self.assertEqual(len(logged_datasets), 2)
        logged_dataset_names = {dataset_input.dataset.name for dataset_input in logged_datasets}
        self.assertEqual(logged_dataset_names, {"msmarco", "msmarco_qrels"})

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
            "CoCondenser-Marco",
        )
        self.assertEqual(
            mlflow_client_instance.create_logged_model.call_args.kwargs["tags"][
                "model_source_kind"
            ],
            "checkpoint",
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
            "script.evaluation.MlflowClient", return_value=mlflow_client_instance
        ):
            _log_mlflow_run_datasets_and_model(
                cfg=cfg,
                run_id="unit-run-id",
                tracking_uri="http://127.0.0.1:5000",
                model_source="/tmp/best.ckpt",
                model_source_kind="checkpoint",
            )

        mlflow_client_instance.log_inputs.assert_not_called()
        mlflow_client_instance.create_logged_model.assert_not_called()
        mlflow_client_instance.log_outputs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
