import unittest
from types import SimpleNamespace

import torch

from src.model.pl_module.loss_service import LossComputationOutputs
from src.model.pl_module.train import SPLADETrainingModule


class _DummyCompilePolicy:
    compile_enabled_for_current_stage: bool = False
    torch_compile_full_model: bool = False

    def has_compiled_train_core(self) -> bool:
        return False

    def has_compiled_query_mdlm_aux(self) -> bool:
        return False

    def has_compiled_doc_mdlm_aux(self) -> bool:
        return False

    def run_compiled_query_mdlm_aux(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise AssertionError("Compiled query MDLM aux path should not be used in this test.")

    def run_compiled_doc_mdlm_aux(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise AssertionError("Compiled doc MDLM aux path should not be used in this test.")

    def can_fuse_query_doc_encoding(self) -> bool:
        return False

    def compiled_train_core_mdlm_apply_mode(
        self,
        *,
        query_seq_len: int,
        doc_seq_len: int,
    ) -> str:
        _ = query_seq_len, doc_seq_len
        return "never"

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


class _DummyMDLMAuxModel(_DummyModel):
    supports_mdlm_aux_loss: bool = True

    def __init__(self) -> None:
        self.mdlm_batch_sizes: list[int] = []
        self.grouped_mdlm_batch_sizes: list[list[int]] = []

    def compute_mdlm_aux_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mask_probability_eps: float,
        force_mask_at_least_one: bool,
    ) -> torch.Tensor:
        _ = attention_mask, mask_probability_eps, force_mask_at_least_one
        self.mdlm_batch_sizes.append(int(input_ids.shape[0]))
        return input_ids[:, 0].to(dtype=torch.float32).mean()

    def compute_grouped_mdlm_aux_losses(
        self,
        *,
        input_id_groups: tuple[torch.Tensor, ...],
        attention_mask_groups: tuple[torch.Tensor, ...],
        mask_probability_eps: float,
        force_mask_at_least_one: bool,
    ) -> tuple[torch.Tensor, ...]:
        _ = attention_mask_groups, mask_probability_eps, force_mask_at_least_one
        self.grouped_mdlm_batch_sizes.append(
            [int(group.shape[0]) for group in input_id_groups]
        )
        return tuple(
            group[:, 0].to(dtype=torch.float32).mean() for group in input_id_groups
        )


class _DummyOrderedMaskSlotModel(_DummyModel):
    supports_ordered_mask_slot_loss: bool = True

    def __init__(self) -> None:
        self.query_slot_logits = torch.tensor(
            [
                [[0.0, 4.0, 1.0], [0.0, 1.0, 4.0]],
                [[0.0, 4.0, 1.0], [0.0, 1.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        self.doc_slot_logits = torch.tensor(
            [
                [[0.0, 4.0, 1.0], [0.0, 1.0, 4.0]],
                [[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]],
                [[0.0, 4.0, 1.0], [0.0, 1.0, 4.0]],
                [[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]],
            ],
            dtype=torch.float32,
        )

    def encode_queries_with_slot_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = input_ids, attention_mask, pooling_mask
        q_reps = torch.tensor(
            [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
            dtype=torch.float32,
        )
        return q_reps, self.query_slot_logits

    def encode_docs_with_slot_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = input_ids, attention_mask, pooling_mask
        doc_reps = torch.tensor(
            [
                [5.0, 6.0, 0.0],
                [7.0, 8.0, 0.0],
                [9.0, 10.0, 0.0],
                [11.0, 12.0, 0.0],
            ],
            dtype=torch.float32,
        )
        return doc_reps, self.doc_slot_logits


class _DummyLossService:
    def lambda_schedule_multiplier(self, global_step: int) -> float:
        _ = global_step
        return 1.0

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


class _DummyCompiledTrainCorePolicy(_DummyCompilePolicy):
    compile_enabled_for_current_stage: bool = True

    def __init__(
        self,
        outputs: tuple[torch.Tensor, ...],
        *,
        mdlm_apply_mode: str = "runtime_flag",
    ) -> None:
        self.outputs = outputs
        self.mark_step_calls: int = 0
        self.mdlm_apply_mode: str = mdlm_apply_mode

    def has_compiled_train_core(self) -> bool:
        return True

    def run_compiled_train_core(self, **_: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.outputs

    def compiled_train_core_mdlm_apply_mode(
        self,
        *,
        query_seq_len: int,
        doc_seq_len: int,
    ) -> str:
        _ = query_seq_len, doc_seq_len
        return self.mdlm_apply_mode

    def maybe_mark_step(self) -> None:
        self.mark_step_calls += 1


class _DummyCompiledMDLMAuxPolicy(_DummyCompilePolicy):
    compile_enabled_for_current_stage: bool = True

    def __init__(self) -> None:
        self.query_batch_sizes: list[int] = []
        self.doc_batch_sizes: list[int] = []

    def has_compiled_query_mdlm_aux(self) -> bool:
        return True

    def has_compiled_doc_mdlm_aux(self) -> bool:
        return True

    def run_compiled_query_mdlm_aux(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        _ = attention_mask
        self.query_batch_sizes.append(int(input_ids.shape[0]))
        return input_ids[:, 0].float().mean()

    def run_compiled_doc_mdlm_aux(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        _ = attention_mask
        self.doc_batch_sizes.append(int(input_ids.shape[0]))
        return input_ids[:, 0].float().mean()


class _FailingLossService(_DummyLossService):
    def compute_loss(self, **_: object) -> LossComputationOutputs:
        raise AssertionError("Regular loss service should be bypassed by compiled train core.")


class TrainingMetricGatingTest(unittest.TestCase):
    def test_mdlm_doc_auxiliary_loss_is_chunked_and_reweighted(self) -> None:
        model = _DummyMDLMAuxModel()
        module = SimpleNamespace()
        module.model = model
        module._mdlm_enabled = True
        module._mdlm_weight = 0.01
        module._mdlm_eps = 1e-3
        module._mdlm_force_mask_at_least_one = True
        module._mdlm_doc_selection = "all"
        module._mdlm_doc_chunk_size = 2

        query_input_ids = torch.tensor([[10, 0], [20, 0]], dtype=torch.long)
        query_attention_mask = torch.ones_like(query_input_ids)
        flat_doc_input_ids = torch.tensor(
            [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
            dtype=torch.long,
        )
        flat_doc_attention_mask = torch.ones_like(flat_doc_input_ids)

        mdlm_q_loss, mdlm_d_loss, mdlm_total = (
            SPLADETrainingModule._compute_mdlm_auxiliary_metrics(
                module,
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                flat_doc_input_ids=flat_doc_input_ids,
                flat_doc_attention_mask=flat_doc_attention_mask,
            )
        )

        self.assertEqual(model.mdlm_batch_sizes, [2, 2, 2, 1])
        self.assertAlmostEqual(float(mdlm_q_loss.item()), 15.0)
        self.assertAlmostEqual(float(mdlm_d_loss.item()), 3.0)
        self.assertAlmostEqual(float(mdlm_total.item()), 18.0)

    def test_mdlm_doc_auxiliary_can_restrict_to_positive_docs(self) -> None:
        model = _DummyMDLMAuxModel()
        module = SimpleNamespace()
        module.model = model
        module._mdlm_enabled = True
        module._mdlm_weight = 0.01
        module._mdlm_eps = 1e-3
        module._mdlm_force_mask_at_least_one = True
        module._mdlm_doc_selection = "positives"
        module._mdlm_doc_chunk_size = 0

        query_input_ids = torch.tensor([[10, 0], [20, 0]], dtype=torch.long)
        query_attention_mask = torch.ones_like(query_input_ids)
        flat_doc_input_ids = torch.tensor(
            [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]],
            dtype=torch.long,
        )
        flat_doc_attention_mask = torch.ones_like(flat_doc_input_ids)
        pos_mask = torch.tensor(
            [[True, False, False], [True, False, True]],
            dtype=torch.bool,
        )
        doc_mask = torch.tensor(
            [[True, True, True], [True, True, False]],
            dtype=torch.bool,
        )

        mdlm_q_loss, mdlm_d_loss, mdlm_total = (
            SPLADETrainingModule._compute_mdlm_auxiliary_metrics(
                module,
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                flat_doc_input_ids=flat_doc_input_ids,
                flat_doc_attention_mask=flat_doc_attention_mask,
                pos_mask=pos_mask,
                doc_mask=doc_mask,
            )
        )

        self.assertEqual(model.mdlm_batch_sizes, [])
        self.assertEqual(model.grouped_mdlm_batch_sizes, [[2, 2]])
        self.assertAlmostEqual(float(mdlm_q_loss.item()), 15.0)
        self.assertAlmostEqual(float(mdlm_d_loss.item()), 2.5)
        self.assertAlmostEqual(float(mdlm_total.item()), 17.5)

    def test_mdlm_doc_auxiliary_falls_back_when_sequence_lengths_differ(self) -> None:
        model = _DummyMDLMAuxModel()
        module = SimpleNamespace()
        module.model = model
        module._compile_policy = _DummyCompilePolicy()
        module._mdlm_enabled = True
        module._mdlm_weight = 0.01
        module._mdlm_eps = 1e-3
        module._mdlm_force_mask_at_least_one = True
        module._mdlm_doc_selection = "positives"
        module._mdlm_doc_chunk_size = 0

        query_input_ids = torch.tensor([[10, 0], [20, 0]], dtype=torch.long)
        query_attention_mask = torch.ones_like(query_input_ids)
        flat_doc_input_ids = torch.tensor(
            [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            dtype=torch.long,
        )
        flat_doc_attention_mask = torch.ones_like(flat_doc_input_ids)
        pos_mask = torch.tensor(
            [[True, False], [True, False]],
            dtype=torch.bool,
        )
        doc_mask = torch.tensor(
            [[True, True], [True, True]],
            dtype=torch.bool,
        )

        mdlm_q_loss, mdlm_d_loss, mdlm_total = (
            SPLADETrainingModule._compute_mdlm_auxiliary_metrics(
                module,
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                flat_doc_input_ids=flat_doc_input_ids,
                flat_doc_attention_mask=flat_doc_attention_mask,
                pos_mask=pos_mask,
                doc_mask=doc_mask,
            )
        )

        self.assertEqual(model.grouped_mdlm_batch_sizes, [])
        self.assertEqual(model.mdlm_batch_sizes, [2, 2])
        self.assertAlmostEqual(float(mdlm_q_loss.item()), 15.0)
        self.assertAlmostEqual(float(mdlm_d_loss.item()), 2.0)
        self.assertAlmostEqual(float(mdlm_total.item()), 17.0)

    def test_mdlm_auxiliary_can_use_compiled_query_and_doc_paths(self) -> None:
        model = _DummyMDLMAuxModel()
        module = SimpleNamespace()
        module.model = model
        module._compile_policy = _DummyCompiledMDLMAuxPolicy()
        module._mdlm_enabled = True
        module._mdlm_weight = 0.01
        module._mdlm_eps = 1e-3
        module._mdlm_force_mask_at_least_one = True
        module._mdlm_doc_selection = "positives"
        module._mdlm_doc_chunk_size = 0

        query_input_ids = torch.tensor([[10, 0], [20, 0]], dtype=torch.long)
        query_attention_mask = torch.ones_like(query_input_ids)
        flat_doc_input_ids = torch.tensor(
            [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            dtype=torch.long,
        )
        flat_doc_attention_mask = torch.ones_like(flat_doc_input_ids)
        pos_mask = torch.tensor(
            [[True, False], [True, False]],
            dtype=torch.bool,
        )
        doc_mask = torch.tensor(
            [[True, True], [True, True]],
            dtype=torch.bool,
        )

        mdlm_q_loss, mdlm_d_loss, mdlm_total = (
            SPLADETrainingModule._compute_mdlm_auxiliary_metrics(
                module,
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                flat_doc_input_ids=flat_doc_input_ids,
                flat_doc_attention_mask=flat_doc_attention_mask,
                pos_mask=pos_mask,
                doc_mask=doc_mask,
            )
        )

        self.assertEqual(model.grouped_mdlm_batch_sizes, [])
        self.assertEqual(model.mdlm_batch_sizes, [])
        self.assertEqual(module._compile_policy.query_batch_sizes, [2])
        self.assertEqual(module._compile_policy.doc_batch_sizes, [2])
        self.assertAlmostEqual(float(mdlm_q_loss.item()), 15.0)
        self.assertAlmostEqual(float(mdlm_d_loss.item()), 2.0)
        self.assertAlmostEqual(float(mdlm_total.item()), 17.0)

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

    def test_ordered_mask_slot_losses_are_added_and_logged(self) -> None:
        module = SimpleNamespace()
        module._compile_policy = _DummyCompilePolicy()
        module.model = _DummyOrderedMaskSlotModel()
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
        module._mdlm_enabled = False
        module._mdlm_weight = 0.0
        module._ordered_mask_slot_enabled = True
        module._ordered_mask_query_weight = 0.3
        module._ordered_mask_doc_weight = 0.2
        module._ordered_mask_ignore_index = -100
        module._compute_ordered_mask_slot_loss = (
            SPLADETrainingModule._compute_ordered_mask_slot_loss.__get__(
                module, SimpleNamespace
            )
        )
        batch = {
            "query_input_ids": torch.ones((2, 3), dtype=torch.long),
            "query_attention_mask": torch.ones((2, 3), dtype=torch.long),
            "doc_input_ids": torch.ones((2, 2, 3), dtype=torch.long),
            "doc_attention_mask": torch.ones((2, 2, 3), dtype=torch.long),
            "pos_mask": torch.tensor([[True, False], [True, False]]),
            "doc_mask": torch.tensor([[True, True], [True, True]]),
            "teacher_scores": torch.zeros((2, 2), dtype=torch.float32),
            "query_slot_target_ids": torch.tensor([[1, 2], [1, 2]], dtype=torch.long),
            "doc_slot_target_ids": torch.tensor(
                [
                    [[1, 2], [-100, -100]],
                    [[1, 2], [-100, -100]],
                ],
                dtype=torch.long,
            ),
        }

        metrics = SPLADETrainingModule._training_step_shared(
            module, batch, stage="train"
        )

        self.assertIn("ordered_query_slot_loss", metrics)
        self.assertIn("ordered_doc_slot_loss", metrics)
        self.assertIn("ordered_mask_slot_loss", metrics)
        self.assertAlmostEqual(float(metrics["ordered_query_slot_weight"].item()), 0.3)
        self.assertAlmostEqual(float(metrics["ordered_doc_slot_weight"].item()), 0.2)
        self.assertGreater(float(metrics["ordered_query_slot_loss"].item()), 0.0)
        self.assertGreater(float(metrics["ordered_doc_slot_loss"].item()), 0.0)
        expected_total = 1.0 + float(metrics["ordered_mask_slot_loss"].item())
        self.assertAlmostEqual(float(metrics["loss"].item()), expected_total, places=5)

    def test_compiled_train_core_bypasses_separate_encode_and_loss(self) -> None:
        q_reps = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        flat_doc_reps = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            dtype=torch.float32,
        )
        loss = torch.tensor(5.0, dtype=torch.float32)
        pairwise_scores = torch.tensor([[1.0, 0.5], [0.7, 0.2]], dtype=torch.float32)
        scalar = torch.tensor(1.0, dtype=torch.float32)
        zero = torch.tensor(0.0, dtype=torch.float32)
        compiled_policy = _DummyCompiledTrainCorePolicy(
            (
                q_reps,
                flat_doc_reps,
                loss,
                pairwise_scores,
                scalar * 2.0,
                scalar * 3.0,
                zero,
                zero,
                zero,
                zero,
                scalar * 4.0,
                scalar * 5.0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                zero,
                zero,
                zero,
                zero,
            )
        )
        module = SimpleNamespace()
        module._compile_policy = compiled_policy
        module.model = object()
        module._validation_doc_encode_chunk_size = 16
        module._loss_service = _FailingLossService()
        module.loss_computer = object()
        module.loss_type = "in_batch_plus_pairwise"
        module.distill_cfg = SimpleNamespace(enabled=False)
        module.reg_cfg = SimpleNamespace(
            query_weight=0.1,
            doc_weight=0.2,
            paper_faithful=True,
            type="l1",
        )
        module.reg_query_weight = 0.1
        module.reg_doc_weight = 0.2
        module._metrics_service = SimpleNamespace(
            should_compute_step_only_metrics=lambda _: False
        )
        module.global_step = 7
        module._compute_rep_magnitude = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expensive magnitude metric should be skipped")
        )
        module._add_sparsity_metrics = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expensive sparsity metric should be skipped")
        )
        module._mdlm_enabled = False
        module._mdlm_weight = 0.0
        batch = {
            "query_input_ids": torch.ones((2, 3), dtype=torch.long),
            "query_attention_mask": torch.ones((2, 3), dtype=torch.long),
            "doc_input_ids": torch.ones((2, 2, 3), dtype=torch.long),
            "doc_attention_mask": torch.ones((2, 2, 3), dtype=torch.long),
            "pos_mask": torch.tensor([[True, False], [True, False]]),
            "doc_mask": torch.tensor([[True, True], [True, True]]),
            "teacher_scores": torch.zeros((2, 2), dtype=torch.float32),
        }

        metrics, rep_outputs = SPLADETrainingModule._training_step_shared(
            module, batch, stage="train", return_reps=True
        )

        self.assertEqual(compiled_policy.mark_step_calls, 1)
        self.assertAlmostEqual(float(metrics["loss"].item()), 5.0)
        self.assertAlmostEqual(float(metrics["pairwise_loss"].item()), 2.0)
        self.assertAlmostEqual(float(metrics["in_batch_loss"].item()), 3.0)
        self.assertAlmostEqual(float(metrics["q_reg"].item()), 4.0)
        self.assertAlmostEqual(float(metrics["d_reg"].item()), 5.0)
        self.assertTrue(torch.equal(rep_outputs["pairwise_scores"], pairwise_scores))

    def test_compiled_train_core_can_supply_mdlm_losses(self) -> None:
        q_reps = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        flat_doc_reps = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            dtype=torch.float32,
        )
        loss = torch.tensor(5.0, dtype=torch.float32)
        pairwise_scores = torch.tensor([[1.0, 0.5], [0.7, 0.2]], dtype=torch.float32)
        scalar = torch.tensor(1.0, dtype=torch.float32)
        zero = torch.tensor(0.0, dtype=torch.float32)
        compiled_policy = _DummyCompiledTrainCorePolicy(
            (
                q_reps,
                flat_doc_reps,
                loss,
                pairwise_scores,
                scalar * 2.0,
                scalar * 3.0,
                zero,
                zero,
                zero,
                zero,
                scalar * 4.0,
                scalar * 5.0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                scalar * 6.0,
                scalar * 7.0,
                scalar * 13.0,
                scalar,
            ),
            mdlm_apply_mode="always",
        )
        module = SimpleNamespace()
        module._compile_policy = compiled_policy
        module.model = object()
        module._validation_doc_encode_chunk_size = 16
        module._loss_service = _FailingLossService()
        module.loss_computer = object()
        module.loss_type = "in_batch_plus_pairwise"
        module.distill_cfg = SimpleNamespace(enabled=False)
        module.reg_cfg = SimpleNamespace(
            query_weight=0.1,
            doc_weight=0.2,
            paper_faithful=True,
            type="l1",
        )
        module.reg_query_weight = 0.1
        module.reg_doc_weight = 0.2
        module._metrics_service = SimpleNamespace(
            should_compute_step_only_metrics=lambda _: False
        )
        module.global_step = 7
        module._mdlm_enabled = True
        module._mdlm_weight = 0.5
        module._compute_mdlm_auxiliary_metrics = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Compiled MDLM path should bypass eager auxiliary computation.")
        )
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

        self.assertEqual(compiled_policy.mark_step_calls, 1)
        self.assertAlmostEqual(float(metrics["mdlm_q_loss"].item()), 6.0)
        self.assertAlmostEqual(float(metrics["mdlm_d_loss"].item()), 7.0)
        self.assertAlmostEqual(float(metrics["mdlm_loss"].item()), 13.0)
        self.assertAlmostEqual(float(metrics["loss"].item()), 11.5)

    def test_compiled_train_core_can_supply_ordered_mask_slot_losses(self) -> None:
        q_reps = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        flat_doc_reps = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            dtype=torch.float32,
        )
        loss = torch.tensor(4.0, dtype=torch.float32)
        pairwise_scores = torch.tensor([[1.0, 0.5], [0.7, 0.2]], dtype=torch.float32)
        scalar = torch.tensor(1.0, dtype=torch.float32)
        zero = torch.tensor(0.0, dtype=torch.float32)
        compiled_policy = _DummyCompiledTrainCorePolicy(
            (
                q_reps,
                flat_doc_reps,
                loss,
                pairwise_scores,
                scalar * 2.0,
                scalar * 3.0,
                zero,
                zero,
                zero,
                zero,
                scalar * 4.0,
                scalar * 5.0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                scalar * 6.0,
                scalar * 7.0,
                scalar * 8.0,
                zero,
                zero,
                zero,
                zero,
            )
        )
        module = SimpleNamespace()
        module._compile_policy = compiled_policy
        module.model = object()
        module._validation_doc_encode_chunk_size = 16
        module._loss_service = _FailingLossService()
        module.loss_computer = object()
        module.loss_type = "in_batch_plus_pairwise"
        module.distill_cfg = SimpleNamespace(enabled=False)
        module.reg_cfg = SimpleNamespace(
            query_weight=0.1,
            doc_weight=0.2,
            paper_faithful=True,
            type="l1",
        )
        module.reg_query_weight = 0.1
        module.reg_doc_weight = 0.2
        module._metrics_service = SimpleNamespace(
            should_compute_step_only_metrics=lambda _: False
        )
        module.global_step = 7
        module._mdlm_enabled = False
        module._mdlm_weight = 0.0
        module._ordered_mask_slot_enabled = True
        module._ordered_mask_query_weight = 0.3
        module._ordered_mask_doc_weight = 0.2
        module._ordered_mask_ignore_index = -100
        module._compute_ordered_mask_slot_loss = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Compiled ordered mask-slot path should bypass eager loss.")
        )
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
            "query_slot_target_ids": torch.tensor([[1, 2], [1, 2]], dtype=torch.long),
            "doc_slot_target_ids": torch.tensor(
                [
                    [[1, 2], [-100, -100]],
                    [[1, 2], [-100, -100]],
                ],
                dtype=torch.long,
            ),
        }

        metrics = SPLADETrainingModule._training_step_shared(
            module, batch, stage="train"
        )

        self.assertEqual(compiled_policy.mark_step_calls, 1)
        self.assertAlmostEqual(float(metrics["ordered_query_slot_loss"].item()), 6.0)
        self.assertAlmostEqual(float(metrics["ordered_doc_slot_loss"].item()), 7.0)
        self.assertAlmostEqual(float(metrics["ordered_mask_slot_loss"].item()), 8.0)
        self.assertAlmostEqual(float(metrics["loss"].item()), 4.0)

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

    def test_flops_proxy_metrics_are_logged_per_side(self) -> None:
        module = SimpleNamespace()
        module._compile_policy = _DummyCompilePolicy()
        module.model = _DummyModel()
        module._validation_doc_encode_chunk_size = 16
        module._loss_service = _DummyLossService()
        module.loss_computer = object()
        module.loss_type = "in_batch"
        module.distill_cfg = SimpleNamespace(enabled=False)
        module.reg_cfg = SimpleNamespace(
            query_weight=0.1,
            doc_weight=0.2,
            paper_faithful=True,
            type="l1",
            query_type="l1",
            doc_type="flops",
        )
        module.reg_query_weight = 0.1
        module.reg_doc_weight = 0.2
        module._metrics_service = SimpleNamespace(
            should_compute_step_only_metrics=lambda _: True
        )
        module.global_step = 1
        module._compute_rep_magnitude = (
            lambda reps, row_mask=None: torch.tensor(1.0, dtype=torch.float32)
        )
        module._add_sparsity_metrics = (
            lambda metrics, **kwargs: metrics.update(
                {
                    "q_active_dims": torch.tensor(1.0, dtype=torch.float32),
                    "doc_active_dims": torch.tensor(1.0, dtype=torch.float32),
                }
            )
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

        self.assertNotIn("q_flops_proxy_sum_equiv", metrics)
        self.assertNotIn("q_flops_proxy_mean_equiv", metrics)
        self.assertIn("d_flops_proxy_sum_equiv", metrics)
        self.assertIn("d_flops_proxy_mean_equiv", metrics)

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
