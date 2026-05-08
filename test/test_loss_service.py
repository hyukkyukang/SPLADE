import unittest

import torch
from omegaconf import OmegaConf

from src.model.pl_module.loss_service import LossRegularizationService


def _build_training_cfg():
    return OmegaConf.create(
        {
            "temperature": 1.0,
            "distill": {
                "enabled": False,
                "include_main_loss": False,
                "losses": [],
            },
            "regularization": {
                "weight": None,
                "query_weight": 0.0,
                "doc_weight": 0.0,
                "type": "l1",
                "query_type": None,
                "doc_type": None,
                "paper_faithful": True,
                "schedule_steps": 0,
            },
            "loss": {
                "type": "in_batch",
                "in_batch_weight": 1.0,
                "pairwise_weight": 1.0,
            },
        }
    )


class LossServiceTest(unittest.TestCase):
    def test_validation_loss_computer_can_be_fixed_to_pairwise(self) -> None:
        service = LossRegularizationService(_build_training_cfg())
        train_loss = service.build_loss_computer()
        validation_loss = service.build_loss_computer(loss_type="pairwise")

        q_reps = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        doc_reps = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        pos_mask = torch.tensor([[True, False], [True, False]])
        doc_mask = torch.tensor([[True, True], [True, True]])
        teacher_scores = torch.zeros((2, 2), dtype=torch.float32)
        lambda_scale = torch.tensor(1.0, dtype=torch.float32)

        train_outputs = train_loss(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            lambda_scale=lambda_scale,
        )
        validation_outputs = validation_loss(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            lambda_scale=lambda_scale,
        )

        self.assertGreater(float(train_outputs[3].item()), 0.0)
        self.assertAlmostEqual(float(validation_outputs[3].item()), 0.0, places=6)
        self.assertGreater(float(validation_outputs[2].item()), 0.0)

    def test_distill_components_are_fixed_and_duplicate_weights_are_aggregated(self) -> None:
        cfg = _build_training_cfg()
        cfg.distill.enabled = True
        cfg.distill.losses = [
            {"type": "mse", "weight": 1.0},
            {"type": "margin_mse", "weight": 2.0},
            {"type": "mse", "weight": 0.5},
        ]
        service = LossRegularizationService(cfg)
        loss_computer = service.build_loss_computer(loss_type="pairwise")

        q_reps = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        doc_reps = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[0.0, 2.0], [2.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        pos_mask = torch.tensor([[True, False], [True, False]])
        doc_mask = torch.tensor([[True, True], [True, True]])
        teacher_scores = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

        outputs = service.compute_loss(
            loss_computer=loss_computer,
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            global_step=0,
        )

        expected_distill = (
            1.5 * outputs.distill_mse_loss + 2.0 * outputs.distill_margin_mse_loss
        )
        self.assertAlmostEqual(
            float(outputs.distill_loss.item()),
            float(expected_distill.item()),
            places=6,
        )
        self.assertAlmostEqual(float(outputs.distill_kl_loss.item()), 0.0, places=6)
        metric_items = service.iter_enabled_distill_metric_tensors(outputs)
        self.assertEqual([name for name, _ in metric_items], ["mse", "margin_mse"])
        self.assertAlmostEqual(
            float(metric_items[0][1].item()),
            float(outputs.distill_mse_loss.item()),
            places=6,
        )
        self.assertAlmostEqual(
            float(metric_items[1][1].item()),
            float(outputs.distill_margin_mse_loss.item()),
            places=6,
        )

    def test_sigmoid_pairwise_hard_keeps_same_validation_loss_type(self) -> None:
        cfg = _build_training_cfg()
        cfg.loss.type = "sigmoid_pairwise_hard"
        cfg.loss.sigmoid = {
            "init_logit_scale": 2.302585093,
            "max_logit_scale": 100.0,
            "init_bias": -10.0,
            "max_bias": -5.0,
            "pos_weight": 1.0,
            "neg_weight": 1.0,
        }
        service = LossRegularizationService(cfg)

        self.assertEqual(service.resolve_validation_loss_type(), "sigmoid_pairwise_hard")
        self.assertTrue(service.has_trainable_loss_parameters)

        loss_computer = service.build_loss_computer()
        self.assertTrue(loss_computer.has_trainable_main_loss_parameters)

    def test_split_regularization_types_override_shared_type(self) -> None:
        cfg = _build_training_cfg()
        cfg.regularization.type = "l1"
        cfg.regularization.query_type = "l1"
        cfg.regularization.doc_type = "flops"

        service = LossRegularizationService(cfg)
        loss_computer = service.build_loss_computer()

        self.assertEqual(service.reg_query_type, "l1")
        self.assertEqual(service.reg_doc_type, "flops")
        self.assertEqual(loss_computer.reg_query_type, "l1")
        self.assertEqual(loss_computer.reg_doc_type, "flops")

    def test_split_regularization_types_fallback_to_shared_type(self) -> None:
        cfg = _build_training_cfg()
        cfg.regularization.type = "flops"

        service = LossRegularizationService(cfg)
        loss_computer = service.build_loss_computer()

        self.assertEqual(service.reg_query_type, "flops")
        self.assertEqual(service.reg_doc_type, "flops")
        self.assertEqual(loss_computer.reg_query_type, "flops")
        self.assertEqual(loss_computer.reg_doc_type, "flops")


if __name__ == "__main__":
    unittest.main()
