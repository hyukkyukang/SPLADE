import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SigmoidPairwiseConfig:
    """Configuration for hard-only pairwise sigmoid ranking loss."""

    init_logit_scale: float
    max_logit_scale: float
    init_bias: float
    max_bias: float
    pos_weight: float
    neg_weight: float


@dataclass(frozen=True)
class SigmoidPairwiseOutputs:
    loss: torch.Tensor
    pos_loss: torch.Tensor
    neg_loss: torch.Tensor
    logit_scale: torch.Tensor
    bias: torch.Tensor
    pos_score_mean: torch.Tensor
    neg_score_mean: torch.Tensor
    pos_margin_mean: torch.Tensor
    neg_margin_mean: torch.Tensor


class SigmoidPairwiseState(nn.Module):
    """Owns trainable affine margin parameters for hard-only sigmoid loss."""

    def __init__(self, cfg: SigmoidPairwiseConfig) -> None:
        super().__init__()
        if float(cfg.max_logit_scale) <= 0.0:
            raise ValueError("sigmoid.max_logit_scale must be positive.")
        self.cfg: SigmoidPairwiseConfig = cfg
        self.logit_scale_param = nn.Parameter(
            torch.tensor(float(cfg.init_logit_scale), dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.tensor(float(cfg.init_bias), dtype=torch.float32))
        self._log_max_logit_scale: float = math.log(float(cfg.max_logit_scale))

    def resolved_logit_scale(self) -> torch.Tensor:
        return self.logit_scale_param.float().exp().clamp(max=self.cfg.max_logit_scale)

    def resolved_bias(self) -> torch.Tensor:
        return self.bias.float().clamp(max=self.cfg.max_bias)

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            self.logit_scale_param.clamp_(max=self._log_max_logit_scale)
            self.bias.clamp_(max=float(self.cfg.max_bias))

    @staticmethod
    def _masked_mean_per_row(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float: torch.Tensor = mask.to(dtype=values.dtype)
        row_sum: torch.Tensor = (values * mask_float).sum(dim=1)
        row_count: torch.Tensor = mask_float.sum(dim=1).clamp(min=1.0)
        return row_sum / row_count

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float: torch.Tensor = mask.to(dtype=values.dtype)
        total_count: torch.Tensor = mask_float.sum().clamp(min=1.0)
        return (values * mask_float).sum() / total_count

    def forward(
        self,
        *,
        scores: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> SigmoidPairwiseOutputs:
        scores_fp32: torch.Tensor = scores.float()
        pos_mask_bool: torch.Tensor = pos_mask.to(dtype=torch.bool)
        doc_mask_bool: torch.Tensor = doc_mask.to(dtype=torch.bool)
        neg_mask_bool: torch.Tensor = doc_mask_bool & ~pos_mask_bool
        valid_query_mask: torch.Tensor = doc_mask_bool.any(dim=1)

        resolved_scale: torch.Tensor = self.resolved_logit_scale()
        resolved_bias: torch.Tensor = self.resolved_bias()
        margin: torch.Tensor = resolved_scale * scores_fp32 + resolved_bias
        pos_term: torch.Tensor = -F.logsigmoid(margin)
        neg_term: torch.Tensor = -F.logsigmoid(-margin)

        pos_loss_per_query: torch.Tensor = self._masked_mean_per_row(pos_term, pos_mask_bool)
        neg_loss_per_query: torch.Tensor = self._masked_mean_per_row(neg_term, neg_mask_bool)
        valid_query_mask_float: torch.Tensor = valid_query_mask.to(dtype=scores_fp32.dtype)
        valid_query_count: torch.Tensor = valid_query_mask_float.sum().clamp(min=1.0)

        pos_loss: torch.Tensor = (
            pos_loss_per_query * valid_query_mask_float
        ).sum() / valid_query_count
        neg_loss: torch.Tensor = (
            neg_loss_per_query * valid_query_mask_float
        ).sum() / valid_query_count
        pos_score_mean: torch.Tensor = self._masked_mean(scores_fp32, pos_mask_bool)
        neg_score_mean: torch.Tensor = self._masked_mean(scores_fp32, neg_mask_bool)
        pos_margin_mean: torch.Tensor = self._masked_mean(margin, pos_mask_bool)
        neg_margin_mean: torch.Tensor = self._masked_mean(margin, neg_mask_bool)
        loss: torch.Tensor = (
            float(self.cfg.pos_weight) * pos_loss
            + float(self.cfg.neg_weight) * neg_loss
        )
        return SigmoidPairwiseOutputs(
            loss=loss,
            pos_loss=pos_loss,
            neg_loss=neg_loss,
            logit_scale=resolved_scale,
            bias=resolved_bias,
            pos_score_mean=pos_score_mean,
            neg_score_mean=neg_score_mean,
            pos_margin_mean=pos_margin_mean,
            neg_margin_mean=neg_margin_mean,
        )
