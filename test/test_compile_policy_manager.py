import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from src.model.pl_module.compile_policy import TrainingCompilePolicyManager
from src.utils.logging import get_logger


class _DummyEncoder(torch.nn.Module):
    def __init__(self, *, freeze_backbone: bool, vocab_size: int = 30522) -> None:
        super().__init__()
        self.freeze_backbone = bool(freeze_backbone)
        self.vocab_size = int(vocab_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mode: torch.Tensor,
    ) -> torch.Tensor:
        _ = attention_mask
        return input_ids.float() + pooling_mode.float()


class _DummyWrapper(torch.nn.Module):
    def __init__(self, *, encoder: _DummyEncoder, pooling_mode: torch.Tensor) -> None:
        super().__init__()
        self._encoder = encoder
        self._pooling_mode = pooling_mode

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mode=self._pooling_mode,
        )


class _DummyModel(torch.nn.Module):
    def __init__(self, *, freeze_backbone: bool, vocab_size: int = 30522) -> None:
        super().__init__()
        self.encoder = _DummyEncoder(
            freeze_backbone=freeze_backbone,
            vocab_size=vocab_size,
        )
        self._query_pooling_mode = torch.tensor(1.0)
        self._doc_pooling_mode = torch.tensor(2.0)
        self._query_encoder_wrapper = _DummyWrapper(
            encoder=self.encoder,
            pooling_mode=self._query_pooling_mode,
        )
        self._doc_encoder_wrapper = _DummyWrapper(
            encoder=self.encoder,
            pooling_mode=self._doc_pooling_mode,
        )
        self._query_encoder_fn = self._query_encoder_wrapper
        self._doc_encoder_fn = self._doc_encoder_wrapper


class _DeviceTrackingModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_marker", torch.zeros((1,)))
        self.to_call_count: int = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids

    def to(self, *args, **kwargs):  # type: ignore[override]
        self.to_call_count += 1
        return super().to(*args, **kwargs)


def _build_training_cfg(**overrides: object):
    base = {
        "training": {
            "torch_compile": True,
            "disable_compile_for_validation": False,
            "torch_compile_mode": "default",
            "strategy": "ddp",
            "num_devices": 4,
            "static_graph": True,
            "find_unused_parameters": False,
            "torch_compile_large_vocab_threshold": 100000,
            "torch_compile_force_aten_gemm_for_large_vocab": True,
        }
    }
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        cfg.training[key] = value
    return cfg


@unittest.skipUnless(hasattr(torch, "compile"), "torch.compile is unavailable")
class CompilePolicyManagerTest(unittest.TestCase):
    def test_unfrozen_ddp_uses_shared_encoder_compile(self) -> None:
        model = _DummyModel(freeze_backbone=False, vocab_size=30522)
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.unfrozen"),
        )
        cfg = _build_training_cfg(torch_compile_mode="max-autotune")

        with patch("torch.compile", side_effect=lambda module, **kwargs: module) as compile_mock:
            manager.setup(cfg)

        self.assertTrue(manager.torch_compile_enabled)
        self.assertFalse(manager.torch_compile_full_model)
        self.assertTrue(manager.compile_enabled_for_current_stage)
        self.assertEqual(compile_mock.call_count, 1)
        self.assertIsNotNone(manager._compiled_query_encoder_fn)
        self.assertIsNotNone(manager._compiled_doc_encoder_fn)
        self.assertEqual(manager.loss_compile_mode_kwargs.get("mode"), "max-autotune")

    def test_frozen_ddp_dynamic_graph_uses_wrapper_compile(self) -> None:
        model = _DummyModel(freeze_backbone=True, vocab_size=30522)
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.frozen"),
        )
        cfg = _build_training_cfg(
            torch_compile_mode="max-autotune",
            static_graph=False,
        )

        with patch("torch.compile", side_effect=lambda module, **kwargs: module) as compile_mock:
            manager.setup(cfg)

        self.assertTrue(manager.torch_compile_enabled)
        self.assertFalse(manager.torch_compile_full_model)
        self.assertTrue(manager.compile_enabled_for_current_stage)
        # Query/doc wrappers are compiled separately in wrapper-only mode.
        self.assertEqual(compile_mock.call_count, 2)
        self.assertIsNotNone(manager._compiled_query_encoder_fn)
        self.assertIsNotNone(manager._compiled_doc_encoder_fn)

    def test_prepare_for_device_moves_wrapper_modules(self) -> None:
        model = _DummyModel(freeze_backbone=True, vocab_size=30522)
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.prepare.wrapper"),
        )
        manager.torch_compile_enabled = True
        manager.torch_compile_full_model = False
        query_module = _DeviceTrackingModule()
        doc_module = _DeviceTrackingModule()
        manager._compiled_query_encoder_fn = query_module
        manager._compiled_doc_encoder_fn = doc_module

        manager.prepare_for_device(device=torch.device("meta"), use_compiled=True)

        self.assertEqual(query_module._marker.device.type, "meta")
        self.assertEqual(doc_module._marker.device.type, "meta")
        self.assertGreater(query_module.to_call_count, 0)
        self.assertGreater(doc_module.to_call_count, 0)

    def test_prepare_for_device_moves_full_compiled_model(self) -> None:
        model = _DummyModel(freeze_backbone=True, vocab_size=30522)
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.prepare.full"),
        )
        manager.torch_compile_enabled = True
        manager.torch_compile_full_model = True
        compiled_model = _DeviceTrackingModule()
        manager.compiled_model = compiled_model

        manager.prepare_for_device(device=torch.device("meta"), use_compiled=True)

        self.assertEqual(compiled_model._marker.device.type, "meta")
        self.assertGreater(compiled_model.to_call_count, 0)

    def test_prepare_for_device_noop_when_not_using_compiled(self) -> None:
        model = _DummyModel(freeze_backbone=True, vocab_size=30522)
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.prepare.noop"),
        )
        manager.torch_compile_enabled = True
        manager.torch_compile_full_model = False
        query_module = _DeviceTrackingModule()
        doc_module = _DeviceTrackingModule()
        manager._compiled_query_encoder_fn = query_module
        manager._compiled_doc_encoder_fn = doc_module

        manager.prepare_for_device(device=torch.device("meta"), use_compiled=False)

        self.assertEqual(query_module._marker.device.type, "cpu")
        self.assertEqual(doc_module._marker.device.type, "cpu")
        self.assertEqual(query_module.to_call_count, 0)
        self.assertEqual(doc_module.to_call_count, 0)


if __name__ == "__main__":
    unittest.main()
