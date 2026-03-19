import unittest
from unittest.mock import ANY, Mock, patch

import torch
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

from script.evaluate_mteb import _build_sparse_encoder


def _build_cfg(*, family: str = "splade", peft_enabled: bool = False) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "family": family,
                "huggingface_name": "unit-test-model",
                "huggingface_model_class": "AutoModelForMaskedLM",
                "query_pooling": "max",
                "doc_pooling": "max",
                "sparse_activation": "log1p_relu",
                "benchmark_adapter": "auto",
                "dtype": "float32",
                "attn_implementation": "sdpa",
                "normalize": False,
                "doc_only": False,
                "tie_word_embeddings": True,
                "use_fast_tokenizer": True,
                "trust_remote_code": False,
                "require_fast_tokenizer": False,
                "peft": {"enabled": peft_enabled},
            },
            "nanobeir": {
                "batch_size": 8,
                "max_seq_length": 128,
            },
            "testing": {
                "checkpoint_path": "/tmp/model.ckpt",
                "use_cpu": True,
            },
        }
    )


class EvaluateMtebBackendRoutingTest(unittest.TestCase):
    def test_lens_checkpoint_uses_native_adapter_path(self) -> None:
        cfg: DictConfig = _build_cfg(family="lens")
        model: Mock = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        adapter: object = object()

        with patch("script.evaluate_mteb.build_splade_model", return_value=model) as build_model_mock, patch(
            "script.evaluate_mteb.load_splade_checkpoint"
        ) as load_checkpoint_mock, patch(
            "script.evaluate_mteb.build_native_sparse_encoder_adapter",
            return_value=adapter,
        ) as build_adapter_mock:
            resolved = _build_sparse_encoder(
                cfg,
                device=torch.device("cpu"),
                model_source_kind="checkpoint",
            )

        self.assertIs(resolved, adapter)
        build_model_mock.assert_called_once_with(cfg, use_cpu=True)
        load_checkpoint_mock.assert_called_once_with(
            model,
            "/tmp/model.ckpt",
            logger=ANY,
        )
        build_adapter_mock.assert_called_once_with(
            cfg=cfg,
            model=model,
            device=torch.device("cpu"),
            batch_size=8,
        )

    def test_compatible_splade_huggingface_uses_sentence_transformers_path(self) -> None:
        cfg: DictConfig = _build_cfg(family="splade")
        sparse_encoder: object = object()

        with patch(
            "script.evaluate_mteb.build_sparse_encoder_from_huggingface",
            return_value=sparse_encoder,
        ) as build_sparse_mock:
            resolved = _build_sparse_encoder(
                cfg,
                device=torch.device("cpu"),
                model_source_kind="huggingface",
            )

        self.assertIs(resolved, sparse_encoder)
        build_sparse_mock.assert_called_once_with(
            cfg=cfg,
            device=torch.device("cpu"),
        )


if __name__ == "__main__":
    unittest.main()
