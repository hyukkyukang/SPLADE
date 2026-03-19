import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from omegaconf import OmegaConf

from src.utils.peft import (
    apply_peft_adapter,
    resolve_peft_settings,
    unwrap_peft_model,
)


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_weight = nn.Parameter(torch.zeros((1,)), requires_grad=False)


class _FakeLoraConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = dict(kwargs)


def _build_fake_peft_module() -> types.ModuleType:
    fake_module = types.ModuleType("peft")
    fake_module.LoraConfig = _FakeLoraConfig
    fake_module.TaskType = types.SimpleNamespace(
        CAUSAL_LM="CAUSAL_LM",
        FEATURE_EXTRACTION="FEATURE_EXTRACTION",
    )

    def fake_get_peft_model(model: nn.Module, config: _FakeLoraConfig) -> nn.Module:
        model.register_parameter(
            "lora_adapter_weight",
            nn.Parameter(torch.ones((1,)), requires_grad=True),
        )
        model.peft_config = {"default": config}
        model.received_lora_config = config
        return model

    fake_module.get_peft_model = fake_get_peft_model
    return fake_module


class _FakeLoraWrapper(nn.Module):
    pass


class _FakePeftModel(nn.Module):
    pass


class PeftUtilsTest(unittest.TestCase):
    def test_resolve_peft_settings_uses_causal_lm_defaults(self) -> None:
        peft_cfg = OmegaConf.create(
            {
                "enabled": True,
                "method": "lora",
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
            }
        )

        settings = resolve_peft_settings(
            peft_cfg,
            model_type="mistral",
            huggingface_model_class="AutoModelForCausalLM",
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.task_type, "CAUSAL_LM")
        self.assertEqual(settings.target_modules, ("q_proj", "k_proj", "v_proj", "o_proj"))

    def test_resolve_peft_settings_treats_custom_causal_loader_as_causal(self) -> None:
        peft_cfg = OmegaConf.create(
            {
                "enabled": True,
                "method": "lora",
                "r": 8,
                "alpha": 16,
            }
        )

        settings = resolve_peft_settings(
            peft_cfg,
            model_type="mistral",
            huggingface_model_class="MistralBiForCausalLM",
        )

        self.assertEqual(settings.task_type, "CAUSAL_LM")

    def test_resolve_peft_settings_requires_target_modules_for_unknown_model(self) -> None:
        peft_cfg = OmegaConf.create(
            {
                "enabled": True,
                "method": "lora",
                "r": 8,
                "alpha": 16,
            }
        )

        with self.assertRaisesRegex(ValueError, "target_modules must be set"):
            _ = resolve_peft_settings(
                peft_cfg,
                model_type="unknown_model",
                huggingface_model_class="AutoModelForCausalLM",
            )

    def test_apply_peft_adapter_uses_lazy_peft_import(self) -> None:
        settings = resolve_peft_settings(
            OmegaConf.create(
                {
                    "enabled": True,
                    "method": "lora",
                    "r": 8,
                    "alpha": 16,
                    "target_modules": ["q_proj"],
                }
            ),
            model_type="mistral",
            huggingface_model_class="AutoModelForCausalLM",
        )
        model = _DummyModel()
        fake_peft = _build_fake_peft_module()

        with patch.dict(sys.modules, {"peft": fake_peft}):
            wrapped_model, trainable_names = apply_peft_adapter(
                model,
                settings=settings,
            )

        self.assertIs(wrapped_model, model)
        self.assertIn("lora_adapter_weight", trainable_names)
        self.assertEqual(
            wrapped_model.received_lora_config.kwargs["target_modules"],
            ["q_proj"],
        )

    def test_unwrap_peft_model_walks_nested_wrappers(self) -> None:
        base_model = _DummyModel()
        lora_wrapper = _FakeLoraWrapper()
        lora_wrapper.model = base_model
        lora_wrapper.__class__.__module__ = "peft.tuners.lora"

        peft_model = _FakePeftModel()
        peft_model.base_model = lora_wrapper
        peft_model.peft_config = {"default": object()}
        peft_model.__class__.__module__ = "peft.peft_model"

        unwrapped = unwrap_peft_model(peft_model)
        self.assertIs(unwrapped, base_model)


if __name__ == "__main__":
    unittest.main()
