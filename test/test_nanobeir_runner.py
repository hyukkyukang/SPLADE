import unittest

import torch
from omegaconf import DictConfig, OmegaConf

from src.model.pl_module.nanobeir_runner import NanoBEIREvaluationRunner
from src.utils.logging import get_logger


def _build_cfg(
    *,
    sparse_activation: str,
    query_pooling: str = "max",
    doc_pooling: str = "max",
    use_cpu: bool = False,
) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "query_pooling": query_pooling,
                "doc_pooling": doc_pooling,
                "sparse_activation": sparse_activation,
            },
            "nanobeir": {
                "enabled": True,
                "run_every_n_val": 1,
                "batch_size": 8,
                "save_json": False,
                "datasets": ["msmarco"],
                "use_cpu": use_cpu,
            },
        }
    )


class NanoBEIRRunnerCompatibilityTest(unittest.TestCase):
    def test_non_doc_only_log1p_softplus_uses_adapter_fallback(self) -> None:
        cfg: DictConfig = _build_cfg(sparse_activation="log1p_softplus")
        runner = NanoBEIREvaluationRunner(
            cfg=cfg,
            logger=get_logger("test.nanobeir_runner.softplus"),
            doc_only_enabled=False,
        )
        self.assertTrue(runner.enabled)
        self.assertTrue(runner._force_adapter_fallback)

    def test_non_doc_only_log1p_relu_keeps_sparse_encoder_path(self) -> None:
        cfg: DictConfig = _build_cfg(sparse_activation="log1p_relu")
        runner = NanoBEIREvaluationRunner(
            cfg=cfg,
            logger=get_logger("test.nanobeir_runner.relu"),
            doc_only_enabled=False,
        )
        self.assertTrue(runner.enabled)
        self.assertFalse(runner._force_adapter_fallback)

    def test_adapter_fallback_ignores_use_cpu_and_uses_training_device(self) -> None:
        cfg: DictConfig = _build_cfg(
            sparse_activation="log1p_softplus",
            use_cpu=True,
        )
        runner = NanoBEIREvaluationRunner(
            cfg=cfg,
            logger=get_logger("test.nanobeir_runner.adapter_device"),
            doc_only_enabled=False,
        )
        training_device: torch.device = torch.device("cuda")
        self.assertEqual(runner.resolve_device(training_device), training_device)


if __name__ == "__main__":
    unittest.main()
