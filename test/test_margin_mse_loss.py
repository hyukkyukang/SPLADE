import unittest

import torch

from src.model.losses import LossComputer


def _build_loss_computer() -> LossComputer:
    return LossComputer(
        loss_type="in_batch",
        temperature=1.0,
        distill_enabled=True,
        include_main_loss_when_distilling=False,
        distill_losses=[("margin_mse", 1.0)],
        reg_query_weight=0.0,
        reg_doc_weight=0.0,
        reg_type="l1",
        reg_paper_faithful=True,
    )


class MarginMSELossTest(unittest.TestCase):
    def test_distill_can_include_main_in_batch_loss(self) -> None:
        loss_no_main = LossComputer(
            loss_type="in_batch",
            temperature=1.0,
            distill_enabled=True,
            include_main_loss_when_distilling=False,
            distill_losses=[("mse", 1.0)],
            reg_query_weight=0.0,
            reg_doc_weight=0.0,
            reg_type="l1",
            reg_paper_faithful=True,
        )
        loss_with_main = LossComputer(
            loss_type="in_batch",
            temperature=1.0,
            distill_enabled=True,
            include_main_loss_when_distilling=True,
            distill_losses=[("mse", 1.0)],
            reg_query_weight=0.0,
            reg_doc_weight=0.0,
            reg_type="l1",
            reg_paper_faithful=True,
        )

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
        lambda_scale = torch.tensor(1.0, dtype=torch.float32)

        out_no_main = loss_no_main(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            lambda_scale=lambda_scale,
        )
        out_with_main = loss_with_main(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            lambda_scale=lambda_scale,
        )

        self.assertAlmostEqual(float(out_no_main[3].item()), 0.0, places=6)
        self.assertGreater(float(out_with_main[3].item()), 0.0)
        self.assertGreater(float(out_with_main[0].item()), float(out_no_main[0].item()))

    def test_uses_all_positive_negative_pairs(self) -> None:
        loss_computer = _build_loss_computer()
        scores = torch.tensor([[4.0, 2.0, 1.0]], dtype=torch.float32)
        teacher_scores = torch.tensor([[3.0, 2.5, 0.0]], dtype=torch.float32)
        pos_mask = torch.tensor([[True, False, False]])
        doc_mask = torch.tensor([[True, True, True]])

        loss = loss_computer._distill_loss_margin_mse(
            scores, teacher_scores, pos_mask, doc_mask
        )

        # Pairs: (pos, neg1), (pos, neg2) -> ((2 - 0.5)^2 + (3 - 3)^2) / 2
        self.assertAlmostEqual(float(loss.item()), 1.125, places=6)

    def test_supports_multiple_positives(self) -> None:
        loss_computer = _build_loss_computer()
        scores = torch.tensor([[5.0, 4.0, 1.0]], dtype=torch.float32)
        teacher_scores = torch.tensor([[4.5, 3.0, 0.0]], dtype=torch.float32)
        pos_mask = torch.tensor([[True, True, False]])
        doc_mask = torch.tensor([[True, True, True]])

        loss = loss_computer._distill_loss_margin_mse(
            scores, teacher_scores, pos_mask, doc_mask
        )

        # Pairs: (pos0, neg), (pos1, neg) -> ((4 - 4.5)^2 + (3 - 3)^2) / 2
        self.assertAlmostEqual(float(loss.item()), 0.125, places=6)

    def test_ignores_non_finite_teacher_scores(self) -> None:
        loss_computer = _build_loss_computer()
        scores = torch.tensor([[4.0, 2.0, 1.0]], dtype=torch.float32)
        teacher_scores = torch.tensor([[3.0, 2.5, float("nan")]], dtype=torch.float32)
        pos_mask = torch.tensor([[True, False, False]])
        doc_mask = torch.tensor([[True, True, True]])

        loss = loss_computer._distill_loss_margin_mse(
            scores, teacher_scores, pos_mask, doc_mask
        )

        # Only (pos, neg1) is valid when neg2 teacher score is NaN.
        self.assertAlmostEqual(float(loss.item()), 2.25, places=6)

    def test_returns_zero_when_no_valid_pairs(self) -> None:
        loss_computer = _build_loss_computer()
        scores = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        teacher_scores = torch.tensor([[float("nan"), float("nan")]], dtype=torch.float32)
        pos_mask = torch.tensor([[True, False]])
        doc_mask = torch.tensor([[True, True]])

        loss = loss_computer._distill_loss_margin_mse(
            scores, teacher_scores, pos_mask, doc_mask
        )

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
