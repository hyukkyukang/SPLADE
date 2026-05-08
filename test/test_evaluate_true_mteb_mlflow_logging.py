import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import DictConfig, OmegaConf

from import_stubs import (
    install_fake_hydra,
    install_fake_mlflow,
    install_fake_pandas,
    install_fake_pytorch_lightning_utilities,
    install_fake_sentence_transformers,
)

install_fake_hydra()
install_fake_mlflow()
install_fake_pandas()
install_fake_pytorch_lightning_utilities()
install_fake_sentence_transformers()

from script.evaluate_true_mteb import (
    _collect_numeric_metrics,
    _log_mlflow_run_datasets_and_model,
)


def _build_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "name": "lens_official_d4000",
                "family": "lens",
                "type": "splade",
                "huggingface_name": "yibinlei/LENS-d4000",
            },
            "mteb": {
                "benchmark_name": "BEIR",
                "tasks": ["NFCorpus", "SciFact"],
                "batch_size": 16,
                "use_huggingface_model": True,
                "parallel": {"enabled": True},
            },
        }
    )


class EvaluateTrueMtebMlflowLoggingTest(unittest.TestCase):
    def test_collect_numeric_metrics_flattens_task_scores(self) -> None:
        results = SimpleNamespace(
            task_results=[
                SimpleNamespace(
                    task_name="NFCorpus",
                    main_score=0.31,
                    evaluation_time=12.5,
                    scores={
                        "test": [
                            {
                                "ndcg_at_10": 0.31,
                                "map_at_100": 0.14,
                            }
                        ]
                    },
                ),
                SimpleNamespace(
                    task_name="SciFact",
                    main_score=0.72,
                    evaluation_time=8.0,
                    scores={
                        "test": [
                            {
                                "ndcg_at_10": 0.72,
                            }
                        ]
                    },
                ),
            ]
        )

        metrics = _collect_numeric_metrics(results)

        self.assertAlmostEqual(metrics["mean_main_score"], 0.515)
        self.assertEqual(metrics["NFCorpus.main_score"], 0.31)
        self.assertEqual(metrics["NFCorpus.evaluation_time"], 12.5)
        self.assertEqual(metrics["NFCorpus.test.ndcg_at_10"], 0.31)
        self.assertEqual(metrics["NFCorpus.test.map_at_100"], 0.14)
        self.assertEqual(metrics["SciFact.test.ndcg_at_10"], 0.72)

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
            "script.evaluate_true_mteb.MlflowClient",
            return_value=mlflow_client_instance,
        ):
            _log_mlflow_run_datasets_and_model(
                cfg=cfg,
                run_id="unit-run-id",
                tracking_uri="http://127.0.0.1:5000",
                task_names=["NFCorpus", "SciFact"],
                model_source="yibinlei/LENS-d4000",
                model_source_kind="huggingface",
            )

        mlflow_client_instance.log_inputs.assert_called_once()
        logged_datasets = mlflow_client_instance.log_inputs.call_args.kwargs["datasets"]
        self.assertEqual(len(logged_datasets), 2)
        logged_dataset_names = {dataset_input.dataset.name for dataset_input in logged_datasets}
        self.assertEqual(logged_dataset_names, {"NFCorpus", "SciFact"})

        mlflow_client_instance.create_logged_model.assert_called_once()
        create_kwargs = mlflow_client_instance.create_logged_model.call_args.kwargs
        self.assertEqual(create_kwargs["experiment_id"], "42")
        self.assertEqual(create_kwargs["source_run_id"], "unit-run-id")
        self.assertEqual(create_kwargs["name"], "LENS-d4000")
        self.assertEqual(create_kwargs["model_type"], "lens")
        self.assertEqual(create_kwargs["tags"]["model_family"], "lens")

        mlflow_client_instance.log_outputs.assert_called_once()
        self.assertEqual(
            mlflow_client_instance.log_outputs.call_args.kwargs["run_id"],
            "unit-run-id",
        )


if __name__ == "__main__":
    unittest.main()
