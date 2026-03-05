import unittest

import torch

from src.model.losses import LossComputer


def _build_loss_computer(
    *,
    loss_type: str,
    in_batch_weight: float = 1.0,
    pairwise_weight: float = 1.0,
) -> LossComputer:
    return LossComputer(
        loss_type=loss_type,
        temperature=1.0,
        distill_enabled=False,
        include_main_loss_when_distilling=False,
        distill_losses=[],
        reg_query_weight=0.0,
        reg_doc_weight=0.0,
        reg_type="flops",
        reg_paper_faithful=True,
        in_batch_weight=in_batch_weight,
        pairwise_weight=pairwise_weight,
    )


class HybridContrastiveLossTest(unittest.TestCase):
    def test_hybrid_main_loss_is_weighted_sum(self) -> None:
        model = _build_loss_computer(
            loss_type="in_batch_plus_pairwise",
            in_batch_weight=2.0,
            pairwise_weight=0.5,
        )
        q_reps = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        doc_reps = torch.tensor(
            [
                [[1.0, 0.0], [0.9, 0.1]],
                [[0.0, 1.0], [0.1, 0.9]],
            ],
            dtype=torch.float32,
        )
        pos_mask = torch.tensor([[True, False], [True, False]])
        doc_mask = torch.tensor([[True, True], [True, True]])
        teacher_scores = torch.zeros((2, 2), dtype=torch.float32)
        lambda_scale = torch.tensor(1.0, dtype=torch.float32)

        (
            total_loss,
            _pairwise_scores,
            pairwise_loss,
            in_batch_loss,
            _distill_loss,
            _distill_losses,
            _q_reg,
            _d_reg,
        ) = model(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            lambda_scale=lambda_scale,
        )

        expected = 2.0 * in_batch_loss + 0.5 * pairwise_loss
        self.assertAlmostEqual(float(total_loss.item()), float(expected.item()), places=6)

    def test_hybrid_easy_in_batch_excludes_same_query_negatives(self) -> None:
        standard = _build_loss_computer(loss_type="in_batch")
        easy_only = _build_loss_computer(
            loss_type="in_batch_plus_pairwise",
            in_batch_weight=1.0,
            pairwise_weight=0.0,
        )
        q_reps = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        # Each query has one very hard same-query negative (score 2.0),
        # which should affect standard in-batch but not easy-only in-batch.
        doc_reps = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0]],
                [[0.0, 1.0], [0.0, 2.0]],
            ],
            dtype=torch.float32,
        )
        pos_mask = torch.tensor([[True, False], [True, False]])
        doc_mask = torch.tensor([[True, True], [True, True]])
        teacher_scores = torch.zeros((2, 2), dtype=torch.float32)
        lambda_scale = torch.tensor(1.0, dtype=torch.float32)

        standard_outputs = standard(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            lambda_scale=lambda_scale,
        )
        easy_outputs = easy_only(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            lambda_scale=lambda_scale,
        )

        standard_in_batch_loss = standard_outputs[3]
        easy_in_batch_loss = easy_outputs[3]
        self.assertLess(float(easy_in_batch_loss.item()), float(standard_in_batch_loss.item()))


if __name__ == "__main__":
    unittest.main()
