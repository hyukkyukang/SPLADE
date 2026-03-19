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
    family: str = "splade",
    peft_enabled: bool = False,
    benchmark_adapter: str = "auto",
) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "family": family,
                "query_pooling": query_pooling,
                "doc_pooling": doc_pooling,
                "sparse_activation": sparse_activation,
                "benchmark_adapter": benchmark_adapter,
                "peft": {"enabled": peft_enabled},
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

    def test_lens_family_uses_native_adapter(self) -> None:
        cfg: DictConfig = _build_cfg(
            sparse_activation="log1p_relu",
            family="lens",
        )
        runner = NanoBEIREvaluationRunner(
            cfg=cfg,
            logger=get_logger("test.nanobeir_runner.lens"),
            doc_only_enabled=False,
        )
        self.assertTrue(runner.enabled)
        self.assertTrue(runner._force_adapter_fallback)

    def test_peft_enabled_uses_native_adapter(self) -> None:
        cfg: DictConfig = _build_cfg(
            sparse_activation="log1p_relu",
            peft_enabled=True,
        )
        runner = NanoBEIREvaluationRunner(
            cfg=cfg,
            logger=get_logger("test.nanobeir_runner.peft"),
            doc_only_enabled=False,
        )
        self.assertTrue(runner.enabled)
        self.assertTrue(runner._force_adapter_fallback)

    def test_benchmark_adapter_native_uses_native_adapter(self) -> None:
        cfg: DictConfig = _build_cfg(
            sparse_activation="log1p_relu",
            benchmark_adapter="native",
        )
        runner = NanoBEIREvaluationRunner(
            cfg=cfg,
            logger=get_logger("test.nanobeir_runner.native_adapter"),
            doc_only_enabled=False,
        )
        self.assertTrue(runner.enabled)
        self.assertTrue(runner._force_adapter_fallback)

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
