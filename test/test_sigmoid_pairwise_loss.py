import unittest

import torch

from src.model.sigmoid_pairwise import SigmoidPairwiseConfig, SigmoidPairwiseState


def _build_state(**overrides: float) -> SigmoidPairwiseState:
    cfg = SigmoidPairwiseConfig(
        init_logit_scale=0.0,
        max_logit_scale=10.0,
        init_bias=0.0,
        max_bias=0.0,
        pos_weight=1.0,
        neg_weight=1.0,
    )
    cfg_dict = {
        "init_logit_scale": cfg.init_logit_scale,
        "max_logit_scale": cfg.max_logit_scale,
        "init_bias": cfg.init_bias,
        "max_bias": cfg.max_bias,
        "pos_weight": cfg.pos_weight,
        "neg_weight": cfg.neg_weight,
    }
    cfg_dict.update(overrides)
    return SigmoidPairwiseState(SigmoidPairwiseConfig(**cfg_dict))


class SigmoidPairwiseLossTest(unittest.TestCase):
    def test_positive_score_increase_reduces_positive_loss(self) -> None:
        state = _build_state()
        pos_mask = torch.tensor([[True, False]])
        doc_mask = torch.tensor([[True, True]])
        low_scores = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
        high_scores = torch.tensor([[2.0, 0.0]], dtype=torch.float32)

        low_outputs = state(scores=low_scores, pos_mask=pos_mask, doc_mask=doc_mask)
        high_outputs = state(scores=high_scores, pos_mask=pos_mask, doc_mask=doc_mask)

        self.assertLess(
            float(high_outputs.pos_loss.item()), float(low_outputs.pos_loss.item())
        )

    def test_negative_score_increase_raises_negative_loss(self) -> None:
        state = _build_state()
        pos_mask = torch.tensor([[True, False]])
        doc_mask = torch.tensor([[True, True]])
        easy_negative_scores = torch.tensor([[1.0, 0.1]], dtype=torch.float32)
        hard_negative_scores = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        easy_outputs = state(
            scores=easy_negative_scores, pos_mask=pos_mask, doc_mask=doc_mask
        )
        hard_outputs = state(
            scores=hard_negative_scores, pos_mask=pos_mask, doc_mask=doc_mask
        )

        self.assertGreater(
            float(hard_outputs.neg_loss.item()), float(easy_outputs.neg_loss.item())
        )

    def test_per_query_negative_normalization_is_invariant_to_duplicate_count(self) -> None:
        state = _build_state()
        pos_mask_short = torch.tensor([[True, False]])
        doc_mask_short = torch.tensor([[True, True]])
        scores_short = torch.tensor([[1.0, 0.5]], dtype=torch.float32)

        pos_mask_long = torch.tensor([[True, False, False, False]])
        doc_mask_long = torch.tensor([[True, True, True, True]])
        scores_long = torch.tensor([[1.0, 0.5, 0.5, 0.5]], dtype=torch.float32)

        short_outputs = state(
            scores=scores_short, pos_mask=pos_mask_short, doc_mask=doc_mask_short
        )
        long_outputs = state(
            scores=scores_long, pos_mask=pos_mask_long, doc_mask=doc_mask_long
        )

        self.assertAlmostEqual(
            float(short_outputs.neg_loss.item()),
            float(long_outputs.neg_loss.item()),
            places=6,
        )
        self.assertAlmostEqual(
            float(short_outputs.loss.item()),
            float(long_outputs.loss.item()),
            places=6,
        )

    def test_supports_multiple_positives(self) -> None:
        state = _build_state()
        pos_mask = torch.tensor([[True, True, False]])
        doc_mask = torch.tensor([[True, True, True]])
        scores = torch.tensor([[2.0, 1.5, 0.5]], dtype=torch.float32)

        outputs = state(scores=scores, pos_mask=pos_mask, doc_mask=doc_mask)

        self.assertGreaterEqual(float(outputs.pos_loss.item()), 0.0)
        self.assertGreaterEqual(float(outputs.neg_loss.item()), 0.0)

    def test_reports_score_and_margin_means(self) -> None:
        state = _build_state(init_logit_scale=0.0, init_bias=-1.0, max_bias=0.0)
        pos_mask = torch.tensor([[True, False, True]])
        doc_mask = torch.tensor([[True, True, True]])
        scores = torch.tensor([[2.0, 0.5, 4.0]], dtype=torch.float32)

        outputs = state(scores=scores, pos_mask=pos_mask, doc_mask=doc_mask)

        self.assertAlmostEqual(float(outputs.pos_score_mean.item()), 3.0, places=6)
        self.assertAlmostEqual(float(outputs.neg_score_mean.item()), 0.5, places=6)
        self.assertAlmostEqual(float(outputs.pos_margin_mean.item()), 2.0, places=6)
        self.assertAlmostEqual(float(outputs.neg_margin_mean.item()), -0.5, places=6)

    def test_clamp_parameters_limits_logit_scale_and_bias(self) -> None:
        state = _build_state(max_logit_scale=5.0, max_bias=-5.0)
        with torch.no_grad():
            state.logit_scale_param.fill_(100.0)
            state.bias.fill_(1.0)

        state.clamp_parameters()

        self.assertLessEqual(
            float(state.resolved_logit_scale().item()),
            5.0 + 1e-6,
        )
        self.assertLessEqual(float(state.resolved_bias().item()), -5.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
