import unittest
from types import SimpleNamespace

import torch

from src.model.pl_module.loss_service import LossComputationOutputs
from src.model.pl_module.train import SPLADETrainingModule


class _DummyCompilePolicy:
    compile_enabled_for_current_stage: bool = False
    torch_compile_full_model: bool = False

    def maybe_mark_step(self) -> None:
        return

    def resolve_active_model_for_train_step(self) -> torch.nn.Module:
        raise AssertionError("Full-model compile path should not be used in this test.")


class _DummyModel:
    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = attention_mask, pooling_mask
        return input_ids.float()

    def encode_docs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = attention_mask, pooling_mask
        return input_ids.float()


class _DummyLossService:
    def compute_loss(self, **_: object) -> LossComputationOutputs:
        scalar = torch.tensor(1.0, dtype=torch.float32)
        zero = torch.tensor(0.0, dtype=torch.float32)
        return LossComputationOutputs(
            loss=scalar,
            pairwise_scores=torch.zeros((2, 2), dtype=torch.float32),
            pairwise_loss=scalar,
            in_batch_loss=scalar,
            distill_loss=zero,
            distill_mse_loss=zero,
            distill_kl_loss=zero,
            distill_margin_mse_loss=zero,
            q_reg=zero,
            d_reg=zero,
            lambda_scale_value=1.0,
            lambda_scale=scalar,
            reg_query_lambda=zero,
            reg_doc_lambda=zero,
        )


class _DummySigmoidLossService(_DummyLossService):
    def compute_loss(self, **_: object) -> LossComputationOutputs:
        outputs = super().compute_loss()
        scalar = torch.tensor(1.0, dtype=torch.float32)
        return LossComputationOutputs(
            loss=outputs.loss,
            pairwise_scores=outputs.pairwise_scores,
            pairwise_loss=outputs.pairwise_loss,
            in_batch_loss=outputs.in_batch_loss,
            distill_loss=outputs.distill_loss,
            distill_mse_loss=outputs.distill_mse_loss,
            distill_kl_loss=outputs.distill_kl_loss,
            distill_margin_mse_loss=outputs.distill_margin_mse_loss,
            q_reg=outputs.q_reg,
            d_reg=outputs.d_reg,
            lambda_scale_value=outputs.lambda_scale_value,
            lambda_scale=outputs.lambda_scale,
            reg_query_lambda=outputs.reg_query_lambda,
            reg_doc_lambda=outputs.reg_doc_lambda,
            sigmoid_pos_loss=scalar,
            sigmoid_neg_loss=scalar * 2.0,
            sigmoid_logit_scale=scalar * 3.0,
            sigmoid_bias=-scalar * 4.0,
            sigmoid_pos_score_mean=scalar * 5.0,
            sigmoid_neg_score_mean=scalar * 6.0,
            sigmoid_pos_margin_mean=scalar * 7.0,
            sigmoid_neg_margin_mean=scalar * 8.0,
        )


class TrainingMetricGatingTest(unittest.TestCase):
    def test_skips_expensive_train_metrics_when_interval_not_due(self) -> None:
        module = SimpleNamespace()
        module._compile_policy = _DummyCompilePolicy()
        module.model = _DummyModel()
        module._validation_doc_encode_chunk_size = 16
        module._loss_service = _DummyLossService()
        module.loss_computer = object()
        module.loss_type = "in_batch"
        module.distill_cfg = SimpleNamespace(enabled=False)
        module.reg_cfg = SimpleNamespace(
            query_weight=0.0,
            doc_weight=0.0,
            paper_faithful=True,
            type="l1",
        )
        module._metrics_service = SimpleNamespace(
            should_compute_step_only_metrics=lambda _: False
        )
        module.global_step = 1
        module._compute_rep_magnitude = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expensive magnitude metric should be skipped")
        )
        module._add_sparsity_metrics = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expensive sparsity metric should be skipped")
        )
        batch = {
            "query_input_ids": torch.ones((2, 3), dtype=torch.long),
            "query_attention_mask": torch.ones((2, 3), dtype=torch.long),
            "doc_input_ids": torch.ones((2, 2, 3), dtype=torch.long),
            "doc_attention_mask": torch.ones((2, 2, 3), dtype=torch.long),
            "pos_mask": torch.tensor([[True, False], [True, False]]),
            "doc_mask": torch.tensor([[True, True], [True, True]]),
            "teacher_scores": torch.zeros((2, 2), dtype=torch.float32),
        }

        metrics = SPLADETrainingModule._training_step_shared(
            module, batch, stage="train"
        )

        self.assertNotIn("q_rep_magnitude", metrics)
        self.assertNotIn("doc_rep_magnitude", metrics)
        self.assertNotIn("q_active_dims", metrics)
        self.assertNotIn("reg_total_contrib", metrics)

    def test_skips_validation_diagnostics_when_disabled(self) -> None:
        module = SimpleNamespace()
        module._compile_policy = _DummyCompilePolicy()
        module.model = _DummyModel()
        module._validation_doc_encode_chunk_size = 16
        module._loss_service = _DummyLossService()
        module.loss_computer = object()
        module.loss_type = "in_batch"
        module.distill_cfg = SimpleNamespace(enabled=False)
        module.reg_cfg = SimpleNamespace(
            query_weight=0.0,
            doc_weight=0.0,
            paper_faithful=True,
            type="l1",
        )
        module._metrics_service = SimpleNamespace(
            should_compute_step_only_metrics=lambda _: True
        )
        module.global_step = 1
        module._compute_rep_magnitude = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("validation magnitude metric should be skipped")
        )
        module._add_sparsity_metrics = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("validation sparsity metric should be skipped")
        )
        batch = {
            "query_input_ids": torch.ones((2, 3), dtype=torch.long),
            "query_attention_mask": torch.ones((2, 3), dtype=torch.long),
            "doc_input_ids": torch.ones((2, 2, 3), dtype=torch.long),
            "doc_attention_mask": torch.ones((2, 2, 3), dtype=torch.long),
            "pos_mask": torch.tensor([[True, False], [True, False]]),
            "doc_mask": torch.tensor([[True, True], [True, True]]),
            "teacher_scores": torch.zeros((2, 2), dtype=torch.float32),
        }

        metrics = SPLADETrainingModule._training_step_shared(
            module,
            batch,
            stage="val",
            compute_validation_diagnostics=False,
        )

        self.assertNotIn("q_rep_magnitude", metrics)
        self.assertNotIn("doc_rep_magnitude", metrics)
        self.assertNotIn("q_active_dims", metrics)
        self.assertNotIn("doc_active_dims", metrics)

    def test_regularizer_metric_gating_uses_effective_weights(self) -> None:
        module = SimpleNamespace()
        module._compile_policy = _DummyCompilePolicy()
        module.model = _DummyModel()
        module._validation_doc_encode_chunk_size = 16
        module._loss_service = _DummyLossService()
        module.loss_computer = object()
        module.loss_type = "in_batch"
        module.distill_cfg = SimpleNamespace(enabled=False)
        module.reg_cfg = SimpleNamespace(
            query_weight=0.0,
            doc_weight=0.0,
            paper_faithful=True,
            type="l1",
        )
        module.reg_query_weight = 0.2
        module.reg_doc_weight = 0.1
        module._metrics_service = SimpleNamespace(
            should_compute_step_only_metrics=lambda _: False
        )
        module.global_step = 1
        module._compute_rep_magnitude = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expensive magnitude metric should be skipped")
        )
        module._add_sparsity_metrics = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expensive sparsity metric should be skipped")
        )
        batch = {
            "query_input_ids": torch.ones((2, 3), dtype=torch.long),
            "query_attention_mask": torch.ones((2, 3), dtype=torch.long),
            "doc_input_ids": torch.ones((2, 2, 3), dtype=torch.long),
            "doc_attention_mask": torch.ones((2, 2, 3), dtype=torch.long),
            "pos_mask": torch.tensor([[True, False], [True, False]]),
            "doc_mask": torch.tensor([[True, True], [True, True]]),
            "teacher_scores": torch.zeros((2, 2), dtype=torch.float32),
        }

        metrics = SPLADETrainingModule._training_step_shared(
            module, batch, stage="train"
        )

        self.assertIn("q_reg", metrics)
        self.assertIn("d_reg", metrics)

    def test_sigmoid_metrics_are_exposed_for_sigmoid_loss_type(self) -> None:
        module = SimpleNamespace()
        module._compile_policy = _DummyCompilePolicy()
        module.model = _DummyModel()
        module._validation_doc_encode_chunk_size = 16
        module._loss_service = _DummySigmoidLossService()
        module.loss_computer = object()
        module.loss_type = "sigmoid_pairwise_hard"
        module.distill_cfg = SimpleNamespace(enabled=False)
        module.reg_cfg = SimpleNamespace(
            query_weight=0.0,
            doc_weight=0.0,
            paper_faithful=True,
            type="l1",
        )
        module.reg_query_weight = 0.0
        module.reg_doc_weight = 0.0
        module._metrics_service = SimpleNamespace(
            should_compute_step_only_metrics=lambda _: False
        )
        module.global_step = 1
        module._compute_rep_magnitude = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expensive magnitude metric should be skipped")
        )
        module._add_sparsity_metrics = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expensive sparsity metric should be skipped")
        )
        batch = {
            "query_input_ids": torch.ones((2, 3), dtype=torch.long),
            "query_attention_mask": torch.ones((2, 3), dtype=torch.long),
            "doc_input_ids": torch.ones((2, 2, 3), dtype=torch.long),
            "doc_attention_mask": torch.ones((2, 2, 3), dtype=torch.long),
            "pos_mask": torch.tensor([[True, False], [True, False]]),
            "doc_mask": torch.tensor([[True, True], [True, True]]),
            "teacher_scores": torch.zeros((2, 2), dtype=torch.float32),
        }

        metrics = SPLADETrainingModule._training_step_shared(
            module, batch, stage="train"
        )

        self.assertIn("pairwise_loss", metrics)
        self.assertIn("sigmoid_pos_loss", metrics)
        self.assertIn("sigmoid_neg_loss", metrics)
        self.assertIn("sigmoid_logit_scale", metrics)
        self.assertIn("sigmoid_bias", metrics)
        self.assertIn("sigmoid_pos_score_mean", metrics)
        self.assertIn("sigmoid_neg_score_mean", metrics)
        self.assertIn("sigmoid_pos_margin_mean", metrics)
        self.assertIn("sigmoid_neg_margin_mean", metrics)


if __name__ == "__main__":
    unittest.main()
