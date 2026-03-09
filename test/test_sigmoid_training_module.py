import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir

from config.path import ABS_CONFIG_DIR
from src.model.pl_module.train import SPLADETrainingModule


class _DummyTrainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.doc_only = False
        self.dummy_weight = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32))

    def encode_queries(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        _ = attention_mask
        return input_ids.float()

    def encode_docs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        _ = attention_mask
        return input_ids.float()


class _DummyValidationAccumulator:
    def __init__(self, *, dataset_name: str, metrics_cfg) -> None:
        _ = dataset_name, metrics_cfg
        self.enabled = False
        self.has_collection = False


class _DummyNanoBEIRRunner:
    def __init__(self, *, cfg, logger, doc_only_enabled: bool) -> None:
        _ = cfg, logger, doc_only_enabled


class SigmoidTrainingModuleTest(unittest.TestCase):
    def _compose_cfg(self):
        with initialize_config_dir(version_base=None, config_dir=ABS_CONFIG_DIR):
            return compose(
                config_name="train_splade_v2_pp_sigmoid_hard",
                overrides=[
                    "training.use_cpu=true",
                    "training.torch_compile=false",
                    "log_dir=tmp/test-sigmoid-training-module",
                    "training.mlflow.save_dir=tmp/test-sigmoid-training-module",
                    "nanobeir.enabled=false",
                ],
            )

    def test_sigmoid_training_module_registers_one_shared_loss_module(self) -> None:
        cfg = self._compose_cfg()
        with (
            patch(
                "src.model.pl_module.train.build_splade_model",
                return_value=_DummyTrainModel(),
            ),
            patch(
                "src.model.pl_module.train.ValidationMetricsAccumulator",
                _DummyValidationAccumulator,
            ),
            patch(
                "src.model.pl_module.train.NanoBEIREvaluationRunner",
                _DummyNanoBEIRRunner,
            ),
        ):
            module = SPLADETrainingModule(cfg)

        self.assertEqual(module.validation_loss_type, "sigmoid_pairwise_hard")
        self.assertIsNone(module._eager_validation_loss_computer)
        self.assertIs(
            module._resolve_stage_loss_computer(stage="val", use_compiled=False),
            module._eager_train_loss_computer,
        )
        parameter_names = [name for name, _param in module.named_parameters()]
        self.assertEqual(
            sum(name.endswith("logit_scale_param") for name in parameter_names), 1
        )
        self.assertEqual(sum(name.endswith("bias") for name in parameter_names), 1)


if __name__ == "__main__":
    unittest.main()
