import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import DictConfig, OmegaConf

from import_stubs import (
    install_fake_hydra,
    install_fake_mlflow,
    install_fake_pandas,
    install_fake_pytorch_lightning_utilities,
)

install_fake_hydra()
install_fake_mlflow()
install_fake_pandas()
install_fake_pytorch_lightning_utilities()

from script.evaluation import (
    _build_mlflow_params,
    _build_mlflow_tags,
    _load_local_eval_artifact_metadata,
    _log_mlflow_run_datasets_and_model,
    _log_to_mlflow,
)


def _build_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "evaluation": {
                "type": "retrieval",
            },
            "model": {
                "name": "splade_v2_pp",
                "family": "splade",
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


def _build_patent_cfg(*, root_dir: str) -> DictConfig:
    eval_dir: str = str(Path(root_dir) / "eval_artifact")
    log_dir: str = str(Path(root_dir) / "logs")
    return OmegaConf.create(
        {
            "tag": "patent_eval_run",
            "log_dir": log_dir,
            "evaluation": {
                "type": "retrieval",
            },
            "model": {
                "name": "dpr_bilingual_negative1_ko_en",
                "family": "dense",
                "type": "dense",
                "huggingface_name": "facebook/dpr-question_encoder-multiset-base",
            },
            "dataset": {
                "name": "patent_us_small_eval_dpr_gcdpr",
                "type": "beir",
                "split": "test",
                "corpus_split": "train",
                "query_corpus_hf_name": "parquet",
                "query_hf_data_files": {
                    "test": str(Path(eval_dir) / "queries.parquet"),
                },
                "qrels_hf_name": "parquet",
                "qrels_hf_split": "test",
                "qrels_hf_data_files": {
                    "test": str(Path(eval_dir) / "qrels.parquet"),
                },
                "corpus_hf_data_files": {
                    "train": str(Path(root_dir) / "corpus" / "passages.parquet"),
                },
            },
            "testing": {
                "batch_size": 32,
                "precision": "bf16-mixed",
                "num_devices": 8,
                "strategy": "ddp",
                "exclude_self_match": False,
                "faiss_use_gpu": True,
                "faiss_gpu_shard": True,
                "faiss_use_float16": True,
                "result_group_key": "group_id",
                "group_candidate_pool": 200,
                "search_top_k": 200,
                "k_list": [1, 5, 10, 16, 32, 64, 150, 1000, 3000, 10000],
                "metric_families": ["Success"],
                "checkpoint_path": "/tmp/patent.ckpt",
                "hf_model_path": None,
                "scoring_method": "full",
                "scoring_backend": "threads",
                "wand_block_size": 256,
                "sparse_top_k": 512,
                "sparse_min_weight": 0.0,
            },
            "encoding": {
                "index_dir": "data/index_user",
                "index_tag": "patent_index_tag",
            },
            "mlflow": {
                "enabled": True,
                "experiment_name": "Eval-Patent-DPR",
                "tracking_uri": "http://127.0.0.1:5000",
                "run_name": None,
                "system_metrics_enabled": False,
                "log_artifacts": True,
                "tags": {
                    "protocol": "gcdpr_proxy",
                },
            },
        }
    )


def _write_patent_eval_artifact_metadata(root_dir: str) -> Path:
    eval_dir = Path(root_dir) / "eval_artifact"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "queries.parquet").touch()
    (eval_dir / "qrels.parquet").touch()
    metadata_path = eval_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "benchmark_source": "hf_dataset",
                "benchmark_repo": "Hyukkyu/patent-us-small",
                "query_text_template": "plain_claims",
                "query_count": 3735,
                "qrels_count": 4714,
            },
            metadata_file,
        )
    return metadata_path


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

    def test_prefers_model_family_for_logged_model_type(self) -> None:
        cfg: DictConfig = _build_cfg()
        cfg.model.family = "lens"
        cfg.model.type = "splade"
        cfg.model.doc_only = False
        cfg.model.peft = OmegaConf.create({"enabled": True})
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

        create_kwargs = mlflow_client_instance.create_logged_model.call_args.kwargs
        self.assertEqual(create_kwargs["model_type"], "lens")
        self.assertEqual(create_kwargs["tags"]["model_family"], "lens")
        self.assertEqual(create_kwargs["tags"]["model_peft_enabled"], "true")

    def test_load_local_eval_artifact_metadata_reads_sidecar_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="splade_eval_mlflow_") as tmpdir:
            cfg = _build_patent_cfg(root_dir=tmpdir)
            metadata_path = _write_patent_eval_artifact_metadata(tmpdir)

            metadata, resolved_path = _load_local_eval_artifact_metadata(cfg.dataset)

        self.assertEqual(metadata["benchmark_repo"], "Hyukkyu/patent-us-small")
        self.assertEqual(metadata["query_text_template"], "plain_claims")
        self.assertEqual(resolved_path, metadata_path)

    def test_build_mlflow_tags_adds_patent_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="splade_eval_mlflow_") as tmpdir:
            cfg = _build_patent_cfg(root_dir=tmpdir)
            _write_patent_eval_artifact_metadata(tmpdir)
            metadata, _ = _load_local_eval_artifact_metadata(cfg.dataset)

            tags = _build_mlflow_tags(
                cfg,
                model_source_kind="checkpoint",
                eval_artifact_metadata=metadata,
            )

        self.assertEqual(tags["domain"], "patent")
        self.assertEqual(tags["task"], "patent_document_retrieval")
        self.assertEqual(tags["retrieval_unit"], "grouped_passage_to_doc")
        self.assertEqual(tags["protocol"], "gcdpr_proxy")
        self.assertEqual(tags["query_text_template"], "plain_claims")
        self.assertEqual(tags["query_source_kind"], "hf_proxy")

    def test_build_mlflow_params_include_patent_retrieval_settings(self) -> None:
        with tempfile.TemporaryDirectory(prefix="splade_eval_mlflow_") as tmpdir:
            cfg = _build_patent_cfg(root_dir=tmpdir)
            _write_patent_eval_artifact_metadata(tmpdir)
            metadata, _ = _load_local_eval_artifact_metadata(cfg.dataset)
            index_path = Path(tmpdir) / "index"
            index_path.mkdir(parents=True, exist_ok=True)
            with (index_path / "metadata.json").open("w", encoding="utf-8") as metadata_file:
                json.dump({"doc_count": 15048639}, metadata_file)

            params = _build_mlflow_params(
                cfg,
                model_source="/tmp/patent.ckpt",
                model_source_kind="checkpoint",
                index_path=index_path,
                eval_artifact_metadata=metadata,
            )

        self.assertEqual(params["testing.search_top_k"], 200)
        self.assertEqual(params["testing.group_candidate_pool"], 200)
        self.assertEqual(params["testing.faiss_gpu_shard"], True)
        self.assertEqual(params["artifact.query_text_template"], "plain_claims")
        self.assertEqual(params["artifact.query_count"], 3735)
        self.assertEqual(params["dataset.query_hf_data_file"], str(Path(tmpdir) / "eval_artifact" / "queries.parquet"))
        self.assertEqual(params["dataset.qrels_hf_data_file"], str(Path(tmpdir) / "eval_artifact" / "qrels.parquet"))
        self.assertEqual(params["dataset.corpus_hf_data_file"], str(Path(tmpdir) / "corpus" / "passages.parquet"))
        self.assertEqual(params["testing.metric_families"], "['Success']")
        self.assertEqual(params["testing.k_list"], "[1, 5, 10, 16, 32, 64, 150, 1000, 3000, 10000]")
        self.assertEqual(params["index.doc_count"], 15048639)

    def test_log_to_mlflow_uploads_dataset_metadata_artifact(self) -> None:
        class _ActiveRun:
            def __init__(self) -> None:
                self.info = SimpleNamespace(run_id="unit-test-run-id")

            def __enter__(self) -> "_ActiveRun":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = exc_type, exc, tb
                return None

        with tempfile.TemporaryDirectory(prefix="splade_eval_mlflow_") as tmpdir:
            cfg = _build_patent_cfg(root_dir=tmpdir)
            metadata_path = _write_patent_eval_artifact_metadata(tmpdir)
            log_dir = Path(cfg.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            output_path = log_dir / "evaluation_metrics.json"
            output_path.write_text("{}", encoding="utf-8")
            (log_dir / "evaluate.log").write_text("log", encoding="utf-8")
            index_path = Path(tmpdir) / "index"
            index_path.mkdir(parents=True, exist_ok=True)
            index_metadata_path = index_path / "metadata.json"
            index_metadata_path.write_text('{"doc_count": 1}', encoding="utf-8")

            with patch("script.evaluation.is_logging_rank_zero", return_value=True), patch(
                "script.evaluation.configure_mlflow_tls"
            ), patch(
                "script.evaluation.mlflow.set_tracking_uri", create=True
            ), patch(
                "script.evaluation.mlflow.set_experiment", create=True
            ) as set_experiment, patch(
                "script.evaluation.mlflow.start_run",
                return_value=_ActiveRun(),
                create=True,
            ), patch(
                "script.evaluation.mlflow.set_tags", create=True
            ) as set_tags, patch(
                "script.evaluation.mlflow.log_params", create=True
            ) as log_params, patch(
                "script.evaluation.mlflow.log_metrics", create=True
            ) as log_metrics, patch(
                "script.evaluation.mlflow.log_artifact", create=True
            ) as log_artifact, patch(
                "script.evaluation._log_mlflow_run_datasets_and_model"
            ):
                _log_to_mlflow(
                    cfg=cfg,
                    model_source="/tmp/patent.ckpt",
                    model_source_kind="checkpoint",
                    numeric_metrics={"Success@1": 0.1},
                    output_path=output_path,
                    index_path=index_path,
                )

        set_experiment.assert_called_once_with("Eval-Patent-DPR")
        set_tags.assert_called_once()
        log_params.assert_called_once()
        self.assertIn(
            "artifact.query_text_template", log_params.call_args.args[0]
        )
        log_metrics.assert_called_once_with({"Success_1": 0.1})
        logged_artifacts = {
            (
                call.args[0],
                call.kwargs.get("artifact_path"),
            )
            for call in log_artifact.call_args_list
        }
        self.assertIn((str(output_path), None), logged_artifacts)
        self.assertIn((str(log_dir / "evaluate.log"), "logs"), logged_artifacts)
        self.assertIn((str(index_metadata_path), "index"), logged_artifacts)
        self.assertIn((str(metadata_path), "dataset"), logged_artifacts)


if __name__ == "__main__":
    unittest.main()
