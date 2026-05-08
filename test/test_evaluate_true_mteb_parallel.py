import os
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

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
    _partition_task_names,
    _prepare_worker_cfg,
    _resolve_parallel_gpu_ids,
    _resolve_visible_gpu_ids,
    _should_run_parallel,
)


class EvaluateTrueMTEBParallelTest(unittest.TestCase):
    def test_partition_task_names_round_robin(self) -> None:
        partitions = _partition_task_names(
            ["NFCorpus", "SciFact", "NQ", "HotpotQA", "FiQA2018"],
            worker_count=2,
        )
        self.assertEqual(
            partitions,
            [["NFCorpus", "NQ", "FiQA2018"], ["SciFact", "HotpotQA"]],
        )

    def test_resolve_visible_gpu_ids_from_env(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,5,7"}, clear=False):
            self.assertEqual(_resolve_visible_gpu_ids(), ["3", "5", "7"])

    def test_resolve_visible_gpu_ids_from_torch(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "script.evaluate_true_mteb.torch.cuda.device_count",
            return_value=4,
        ):
            self.assertEqual(_resolve_visible_gpu_ids(), ["0", "1", "2", "3"])

    def test_resolve_parallel_gpu_ids_prefers_config(self) -> None:
        cfg = OmegaConf.create(
            {"mteb": {"parallel": {"gpu_ids": [2, 4], "enabled": True}}}
        )
        self.assertEqual(_resolve_parallel_gpu_ids(cfg), ["2", "4"])

    def test_should_run_parallel_requires_multiple_tasks_and_gpus(self) -> None:
        cfg = OmegaConf.create(
            {
                "testing": {"use_cpu": False},
                "mteb": {"parallel": {"enabled": True}},
            }
        )
        self.assertTrue(
            _should_run_parallel(
                cfg,
                tasks=[object(), object()],
                device=__import__("torch").device("cuda"),
                gpu_ids=["0", "1"],
            )
        )
        self.assertFalse(
            _should_run_parallel(
                cfg,
                tasks=[object()],
                device=__import__("torch").device("cuda"),
                gpu_ids=["0", "1"],
            )
        )

    def test_prepare_worker_cfg_forces_json_and_disables_mlflow(self) -> None:
        cfg = OmegaConf.create(
            {
                "tag": "parent-run",
                "log_dir_base": "log",
                "log_dir": "log/parent-run",
                "mteb": {
                    "benchmark_name": "BEIR",
                    "tasks": ["NFCorpus", "SciFact"],
                    "save_json": False,
                    "parallel": {"enabled": True},
                },
                "mlflow": {"enabled": True},
            }
        )
        worker_cfg = _prepare_worker_cfg(
            cfg,
            worker_tag="parent-run__worker0",
            worker_task_names=["NFCorpus"],
        )
        self.assertEqual(worker_cfg.mteb.tasks, ["NFCorpus"])
        self.assertIsNone(worker_cfg.mteb.benchmark_name)
        self.assertTrue(worker_cfg.mteb.save_json)
        self.assertFalse(worker_cfg.mteb.parallel.enabled)
        self.assertFalse(worker_cfg.mlflow.enabled)


if __name__ == "__main__":
    unittest.main()
