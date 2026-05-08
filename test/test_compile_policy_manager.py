import unittest
from unittest.mock import patch

import torch
import torch._inductor.config as inductor_config
from omegaconf import OmegaConf

from src.model.pl_module.compile_policy import TrainingCompilePolicyManager
from src.utils.logging import get_logger


class _DummyEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        freeze_backbone: bool,
        vocab_size: int = 30522,
        peft_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.freeze_backbone = bool(freeze_backbone)
        self.vocab_size = int(vocab_size)
        self.peft_enabled = bool(peft_enabled)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mode: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = attention_mask, pooling_mask
        return input_ids.float() + pooling_mode.float()


class _DummyWrapper(torch.nn.Module):
    def __init__(self, *, encoder: _DummyEncoder, pooling_mode: torch.Tensor) -> None:
        super().__init__()
        self._encoder = encoder
        self._pooling_mode = pooling_mode

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mode=self._pooling_mode,
            pooling_mask=pooling_mask,
        )


class _DummyModel(torch.nn.Module):
    def __init__(
        self,
        *,
        freeze_backbone: bool,
        vocab_size: int = 30522,
        doc_only: bool = False,
        peft_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = _DummyEncoder(
            freeze_backbone=freeze_backbone,
            vocab_size=vocab_size,
            peft_enabled=peft_enabled,
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
        self.doc_only = bool(doc_only)
        self.peft_enabled = bool(peft_enabled)


class _DummyMDLMModel(_DummyModel):
    supports_mdlm_aux_loss: bool = True

    def compute_mdlm_aux_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mask_probability_eps: float,
        force_mask_at_least_one: bool,
    ) -> torch.Tensor:
        _ = attention_mask, mask_probability_eps, force_mask_at_least_one
        return input_ids[:, 0].float().mean()


class _DummyOrderedMaskSlotModel(_DummyModel):
    supports_ordered_mask_slot_loss: bool = True

    def __init__(self, *, freeze_backbone: bool, vocab_size: int = 8) -> None:
        super().__init__(freeze_backbone=freeze_backbone, vocab_size=vocab_size)

    def encode_queries_with_slot_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = attention_mask, pooling_mask
        q_reps = input_ids.float()
        slot_logits = torch.nn.functional.one_hot(
            input_ids[:, -2:],
            num_classes=int(self.encoder.vocab_size),
        ).to(dtype=torch.float32) * 5.0
        return q_reps, slot_logits

    def encode_docs_with_slot_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = attention_mask, pooling_mask
        doc_reps = input_ids.float()
        slot_logits = torch.nn.functional.one_hot(
            input_ids[:, -2:],
            num_classes=int(self.encoder.vocab_size),
        ).to(dtype=torch.float32) * 5.0
        return doc_reps, slot_logits


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


class _DummyLossComputer(torch.nn.Module):
    def forward(
        self,
        *,
        q_reps: torch.Tensor,
        doc_reps: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        teacher_scores: torch.Tensor,
        lambda_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        _ = q_reps, doc_reps, pos_mask, doc_mask, teacher_scores
        zero = lambda_scale.new_zeros(())
        return (
            zero,
            zero.expand(1, 1),
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _build_training_cfg(**overrides: object):
    base = {
        "training": {
            "torch_compile": True,
            "disable_compile_for_validation": False,
            "torch_compile_mode": "default",
            "torch_compile_train_core_when_possible": False,
            "strategy": "ddp",
            "num_devices": 4,
            "use_cpu": True,
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

    def test_unfrozen_ddp_can_defer_to_compiled_train_core(self) -> None:
        model = _DummyModel(freeze_backbone=False, vocab_size=30522)
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.unfrozen_train_core"),
        )
        cfg = _build_training_cfg(
            torch_compile_mode="default",
            torch_compile_train_core_when_possible=True,
        )

        with patch("torch.compile", side_effect=lambda module, **kwargs: module) as compile_mock:
            manager.setup(cfg)
            self.assertTrue(manager.torch_compile_enabled)
            self.assertFalse(manager.compile_enabled_for_current_stage)
            self.assertTrue(manager._defer_train_core_compile)
            self.assertEqual(compile_mock.call_count, 0)
            manager.finalize_train_core_compile(loss_computer=_DummyLossComputer())

        self.assertEqual(compile_mock.call_count, 1)
        self.assertTrue(manager.compiled_train_core_available())
        self.assertIsNone(manager._compiled_shared_encoder_module)
        manager.set_train_core_active(True)
        manager.set_compile_state(use_compiled=True)
        self.assertTrue(manager.compile_enabled_for_current_stage)
        manager.set_train_core_active(False)
        manager.set_compile_state(use_compiled=True)
        self.assertFalse(manager.compile_enabled_for_current_stage)

    def test_unfrozen_ddp_can_compile_query_and_doc_mdlm_aux_modules(self) -> None:
        model = _DummyMDLMModel(freeze_backbone=False, vocab_size=30522)
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.unfrozen_mdlm_aux"),
        )
        cfg = _build_training_cfg(
            torch_compile_mode="default",
            torch_compile_train_core_when_possible=True,
        )

        with patch("torch.compile", side_effect=lambda module, **kwargs: module) as compile_mock:
            manager.setup(cfg)
            manager.finalize_train_core_compile(
                loss_computer=_DummyLossComputer(),
                mdlm_enabled=True,
            )

        self.assertEqual(compile_mock.call_count, 3)
        self.assertTrue(manager.compiled_train_core_available())
        self.assertIsNotNone(manager._compiled_query_mdlm_aux_module)
        self.assertIsNotNone(manager._compiled_doc_mdlm_aux_module)
        manager.set_train_core_active(True)
        manager.set_compile_state(use_compiled=True)
        self.assertTrue(manager.has_compiled_query_mdlm_aux())
        self.assertTrue(manager.has_compiled_doc_mdlm_aux())

    def test_compiled_train_core_mdlm_apply_mode_can_be_static(self) -> None:
        model = _DummyMDLMModel(freeze_backbone=False, vocab_size=30522)
        model._doc_pooling_mode = model._query_pooling_mode.clone()
        model.compute_grouped_mdlm_aux_losses = lambda **kwargs: (
            kwargs["input_id_groups"][0][:, 0].float().mean(),
            kwargs["input_id_groups"][1][:, 0].float().mean(),
        )
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.mdlm_apply_mode"),
        )
        cfg = _build_training_cfg(
            torch_compile_mode="default",
            torch_compile_train_core_when_possible=True,
        )

        with patch("torch.compile", side_effect=lambda module, **kwargs: module):
            manager.setup(cfg)
            manager.finalize_train_core_compile(
                loss_computer=_DummyLossComputer(),
                mdlm_enabled=True,
                mdlm_doc_selection="positives",
                mdlm_doc_chunk_size=0,
                mdlm_single_positive_assumption=True,
            )

        self.assertEqual(
            manager.compiled_train_core_mdlm_apply_mode(
                query_seq_len=128,
                doc_seq_len=128,
            ),
            "always",
        )
        self.assertEqual(
            manager.compiled_train_core_mdlm_apply_mode(
                query_seq_len=128,
                doc_seq_len=256,
            ),
            "never",
        )

    def test_unfrozen_ddp_compiled_train_core_can_include_ordered_mask_slot_losses(self) -> None:
        model = _DummyOrderedMaskSlotModel(freeze_backbone=False, vocab_size=8)
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.ordered_mask_slot"),
        )
        cfg = _build_training_cfg(
            torch_compile_mode="default",
            torch_compile_train_core_when_possible=True,
        )

        with patch("torch.compile", side_effect=lambda module, **kwargs: module):
            manager.setup(cfg)
            manager.finalize_train_core_compile(
                loss_computer=_DummyLossComputer(),
                ordered_mask_slot_enabled=True,
                ordered_mask_query_weight=0.3,
                ordered_mask_doc_weight=0.2,
                ordered_mask_ignore_index=-100,
            )

        self.assertTrue(manager.compiled_train_core_available())
        outputs = manager.run_compiled_train_core(
            query_input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
            query_attention_mask=torch.ones((2, 3), dtype=torch.long),
            doc_input_ids=torch.tensor(
                [[1, 2, 3], [6, 5, 4], [4, 5, 6], [3, 2, 1]],
                dtype=torch.long,
            ),
            doc_attention_mask=torch.ones((4, 3), dtype=torch.long),
            pos_mask=torch.tensor([[True, False], [True, False]], dtype=torch.bool),
            doc_mask=torch.tensor([[True, True], [True, True]], dtype=torch.bool),
            teacher_scores=torch.zeros((2, 2), dtype=torch.float32),
            lambda_scale=torch.tensor(1.0, dtype=torch.float32),
            query_slot_target_ids=torch.tensor([[2, 3], [5, 6]], dtype=torch.long),
            doc_slot_target_ids=torch.tensor(
                [[[2, 3], [-100, -100]], [[5, 6], [-100, -100]]],
                dtype=torch.long,
            ),
        )

        self.assertEqual(len(outputs), 27)
        self.assertGreater(float(outputs[2].item()), 0.0)
        self.assertGreater(float(outputs[20].item()), 0.0)
        self.assertGreater(float(outputs[21].item()), 0.0)
        self.assertGreater(float(outputs[22].item()), 0.0)

    def test_unfrozen_ddp_applies_large_vocab_aten_gemm_safety(self) -> None:
        model = _DummyModel(freeze_backbone=False, vocab_size=30522)
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.unfrozen_large_vocab"),
        )
        cfg = _build_training_cfg(
            torch_compile_mode="max-autotune",
            torch_compile_large_vocab_threshold=30000,
        )

        original_backend = getattr(
            inductor_config, "max_autotune_gemm_backends", None
        )
        try:
            inductor_config.max_autotune_gemm_backends = "TRITON"
            with patch(
                "torch.compile", side_effect=lambda module, **kwargs: module
            ) as compile_mock:
                manager.setup(cfg)
            self.assertEqual(compile_mock.call_count, 1)
            self.assertEqual(inductor_config.max_autotune_gemm_backends, "ATEN")
        finally:
            if original_backend is None:
                try:
                    delattr(inductor_config, "max_autotune_gemm_backends")
                except AttributeError:
                    pass
            else:
                inductor_config.max_autotune_gemm_backends = original_backend

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

    def test_effective_dynamic_ddp_blocks_full_model_compile(self) -> None:
        model = _DummyModel(freeze_backbone=True, vocab_size=30522)
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.dynamic_ddp_guard"),
        )
        cfg = _build_training_cfg(
            torch_compile_mode="max-autotune",
            static_graph=True,
            torch_compile_ddp_safe_mode=True,
            use_cpu=False,
        )

        with patch("torch.cuda.device_count", return_value=4), patch(
            "torch.compile", side_effect=lambda module, **kwargs: module
        ) as compile_mock:
            manager.setup(cfg)

        self.assertTrue(manager.torch_compile_enabled)
        self.assertFalse(manager.torch_compile_full_model)
        self.assertTrue(manager.compile_enabled_for_current_stage)
        self.assertEqual(compile_mock.call_count, 2)

    def test_doc_only_never_uses_full_model_compile(self) -> None:
        model = _DummyModel(freeze_backbone=True, vocab_size=30522, doc_only=True)
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.doc_only"),
        )
        cfg = _build_training_cfg(
            torch_compile_mode="max-autotune",
            strategy="single",
            num_devices=1,
        )

        with patch("torch.compile", side_effect=lambda module, **kwargs: module) as compile_mock:
            manager.setup(cfg)

        self.assertTrue(manager.torch_compile_enabled)
        self.assertFalse(manager.torch_compile_full_model)
        self.assertTrue(manager.compile_enabled_for_current_stage)
        # doc_only keeps the bag-of-words query path eager and compiles docs only.
        self.assertEqual(compile_mock.call_count, 1)

    def test_peft_never_uses_full_model_compile(self) -> None:
        model = _DummyModel(
            freeze_backbone=True,
            vocab_size=30522,
            peft_enabled=True,
        )
        manager = TrainingCompilePolicyManager(
            model=model,
            logger=get_logger("test.compile_policy.peft"),
        )
        cfg = _build_training_cfg(
            torch_compile_mode="max-autotune",
            strategy="single",
            num_devices=1,
        )

        with patch("torch.compile", side_effect=lambda module, **kwargs: module) as compile_mock:
            manager.setup(cfg)

        self.assertTrue(manager.torch_compile_enabled)
        self.assertFalse(manager.torch_compile_full_model)
        self.assertTrue(manager.compile_enabled_for_current_stage)
        self.assertEqual(compile_mock.call_count, 2)

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
