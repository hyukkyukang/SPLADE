from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig

from src.model.losses import LossComputer
from src.model.sigmoid_pairwise import SigmoidPairwiseConfig


@dataclass(frozen=True)
class LossComputationOutputs:
    loss: torch.Tensor
    pairwise_scores: torch.Tensor
    pairwise_loss: torch.Tensor
    in_batch_loss: torch.Tensor
    distill_loss: torch.Tensor
    distill_mse_loss: torch.Tensor
    distill_kl_loss: torch.Tensor
    distill_margin_mse_loss: torch.Tensor
    q_reg: torch.Tensor
    d_reg: torch.Tensor
    lambda_scale_value: float
    lambda_scale: torch.Tensor
    reg_query_lambda: torch.Tensor
    reg_doc_lambda: torch.Tensor
    sigmoid_pos_loss: torch.Tensor | None = None
    sigmoid_neg_loss: torch.Tensor | None = None
    sigmoid_logit_scale: torch.Tensor | None = None
    sigmoid_bias: torch.Tensor | None = None
    sigmoid_pos_score_mean: torch.Tensor | None = None
    sigmoid_neg_score_mean: torch.Tensor | None = None
    sigmoid_pos_margin_mean: torch.Tensor | None = None
    sigmoid_neg_margin_mean: torch.Tensor | None = None


class LossRegularizationService:
    """Own LossComputer config + per-step regularization scheduling."""

    def __init__(self, training_cfg: DictConfig) -> None:
        self.temperature: float = float(training_cfg.temperature)
        self.distill_cfg: DictConfig = training_cfg.distill
        self.reg_cfg: DictConfig = training_cfg.regularization
        self.loss_cfg: DictConfig = training_cfg.loss
        self.loss_type: str = str(self.loss_cfg.type).lower()
        self.in_batch_weight: float = float(self.loss_cfg.get("in_batch_weight", 1.0))
        self.pairwise_weight: float = float(self.loss_cfg.get("pairwise_weight", 1.0))
        self.sigmoid_cfg: SigmoidPairwiseConfig | None = self._resolve_sigmoid_config()
        self._resolved_distill_losses: tuple[tuple[str, float], ...] = (
            tuple(self._resolve_distill_losses())
            if bool(self.distill_cfg.enabled)
            else tuple()
        )

        reg_weight_value: float | None = self.reg_cfg.weight
        if reg_weight_value is None:
            self.reg_query_weight: float = float(self.reg_cfg.query_weight)
            self.reg_doc_weight: float = float(self.reg_cfg.doc_weight)
        else:
            # Single lambda applied to both query and document regularization.
            self.reg_query_weight = float(reg_weight_value)
            self.reg_doc_weight = float(reg_weight_value)
        self.reg_type: str = str(self.reg_cfg.type).lower()
        self.reg_query_type: str = self._resolve_regularization_type(
            explicit_type=self.reg_cfg.get("query_type"),
            fallback_type=self.reg_type,
        )
        self.reg_doc_type: str = self._resolve_regularization_type(
            explicit_type=self.reg_cfg.get("doc_type"),
            fallback_type=self.reg_type,
        )

    def _resolve_regularization_type(
        self, *, explicit_type: Any | None, fallback_type: str
    ) -> str:
        if explicit_type is None:
            return fallback_type
        return str(explicit_type).strip().lower()

    def _resolve_sigmoid_config(self) -> SigmoidPairwiseConfig | None:
        if self.loss_type != "sigmoid_pairwise_hard":
            return None
        raw_sigmoid_cfg: DictConfig | None = self.loss_cfg.get("sigmoid")
        if raw_sigmoid_cfg is None:
            raise ValueError("loss.sigmoid must be configured for sigmoid_pairwise_hard.")
        return SigmoidPairwiseConfig(
            init_logit_scale=float(raw_sigmoid_cfg.get("init_logit_scale", -6.907755279)),
            max_logit_scale=float(raw_sigmoid_cfg.get("max_logit_scale", 100.0)),
            init_bias=float(raw_sigmoid_cfg.get("init_bias", -8.0)),
            max_bias=float(raw_sigmoid_cfg.get("max_bias", -5.0)),
            pos_weight=float(raw_sigmoid_cfg.get("pos_weight", 1.0)),
            neg_weight=float(raw_sigmoid_cfg.get("neg_weight", 1.0)),
        )

    def _resolve_distill_losses(self) -> list[tuple[str, float]]:
        distill_losses_cfg: Any | None = self.distill_cfg.losses
        aggregated_weights: dict[str, float] = {
            "mse": 0.0,
            "kl": 0.0,
            "margin_mse": 0.0,
        }
        if distill_losses_cfg is None:
            return []
        for entry in distill_losses_cfg:
            loss_type: str = str(entry.type).replace("-", "_").lower()
            loss_weight: float = float(entry.weight)
            if loss_weight == 0.0:
                continue
            if loss_type not in aggregated_weights:
                raise ValueError(f"Unsupported distillation loss: {entry.type}")
            aggregated_weights[loss_type] += loss_weight
        return [
            (loss_type, loss_weight)
            for loss_type, loss_weight in aggregated_weights.items()
            if loss_weight != 0.0
        ]

    def build_loss_computer(self, *, loss_type: str | None = None) -> LossComputer:
        resolved_loss_type: str = (
            self.loss_type if loss_type is None else str(loss_type).lower()
        )
        sigmoid_config: SigmoidPairwiseConfig | None = (
            self.sigmoid_cfg if resolved_loss_type == "sigmoid_pairwise_hard" else None
        )
        return LossComputer(
            loss_type=resolved_loss_type,
            temperature=self.temperature,
            distill_enabled=bool(self.distill_cfg.enabled),
            include_main_loss_when_distilling=bool(
                self.distill_cfg.get("include_main_loss", False)
            ),
            distill_losses=list(self._resolved_distill_losses),
            reg_query_weight=self.reg_query_weight,
            reg_doc_weight=self.reg_doc_weight,
            reg_type=self.reg_type,
            reg_query_type=self.reg_query_type,
            reg_doc_type=self.reg_doc_type,
            reg_paper_faithful=bool(self.reg_cfg.paper_faithful),
            in_batch_weight=self.in_batch_weight,
            pairwise_weight=self.pairwise_weight,
            sigmoid_config=sigmoid_config,
        )

    def resolve_validation_loss_type(self) -> str:
        if self.loss_type == "sigmoid_pairwise_hard":
            return self.loss_type
        return "pairwise"

    @property
    def has_trainable_loss_parameters(self) -> bool:
        return self.sigmoid_cfg is not None

    def lambda_schedule_multiplier(self, global_step: int) -> float:
        schedule_steps: int | None = self.reg_cfg.schedule_steps
        if schedule_steps is None:
            return 1.0
        resolved_schedule_steps: int = int(schedule_steps)
        if resolved_schedule_steps <= 0:
            return 1.0
        step: int = max(int(global_step), 0)
        progress: float = min(step, resolved_schedule_steps) / float(
            resolved_schedule_steps
        )
        return progress * progress

    def iter_enabled_distill_metric_tensors(
        self, outputs: LossComputationOutputs
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        enabled_metrics: list[tuple[str, torch.Tensor]] = []
        for loss_type, _weight in self._resolved_distill_losses:
            if loss_type == "mse":
                enabled_metrics.append((loss_type, outputs.distill_mse_loss))
                continue
            if loss_type == "kl":
                enabled_metrics.append((loss_type, outputs.distill_kl_loss))
                continue
            if loss_type == "margin_mse":
                enabled_metrics.append((loss_type, outputs.distill_margin_mse_loss))
                continue
            raise ValueError(f"Unsupported distillation loss: {loss_type}")
        return tuple(enabled_metrics)

    def compute_loss(
        self,
        *,
        loss_computer: LossComputer,
        q_reps: torch.Tensor,
        doc_reps: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        teacher_scores: torch.Tensor,
        global_step: int,
    ) -> LossComputationOutputs:
        lambda_scale_value: float = self.lambda_schedule_multiplier(global_step)
        lambda_scale: torch.Tensor = torch.tensor(
            lambda_scale_value, device=q_reps.device, dtype=q_reps.dtype
        )
        # Effective per-step regularization weights with scheduling applied.
        reg_query_lambda: torch.Tensor = lambda_scale * float(self.reg_query_weight)
        reg_doc_lambda: torch.Tensor = lambda_scale * float(self.reg_doc_weight)

        (
            loss,
            pairwise_scores,
            pairwise_loss,
            in_batch_loss,
            distill_loss,
            distill_mse_loss,
            distill_kl_loss,
            distill_margin_mse_loss,
            q_reg,
            d_reg,
            sigmoid_pos_loss,
            sigmoid_neg_loss,
            sigmoid_logit_scale,
            sigmoid_bias,
            sigmoid_pos_score_mean,
            sigmoid_neg_score_mean,
            sigmoid_pos_margin_mean,
            sigmoid_neg_margin_mean,
        ) = loss_computer(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            lambda_scale=lambda_scale,
        )
        return LossComputationOutputs(
            loss=loss,
            pairwise_scores=pairwise_scores,
            pairwise_loss=pairwise_loss,
            in_batch_loss=in_batch_loss,
            distill_loss=distill_loss,
            distill_mse_loss=distill_mse_loss,
            distill_kl_loss=distill_kl_loss,
            distill_margin_mse_loss=distill_margin_mse_loss,
            q_reg=q_reg,
            d_reg=d_reg,
            lambda_scale_value=lambda_scale_value,
            lambda_scale=lambda_scale,
            reg_query_lambda=reg_query_lambda,
            reg_doc_lambda=reg_doc_lambda,
            sigmoid_pos_loss=sigmoid_pos_loss,
            sigmoid_neg_loss=sigmoid_neg_loss,
            sigmoid_logit_scale=sigmoid_logit_scale,
            sigmoid_bias=sigmoid_bias,
            sigmoid_pos_score_mean=sigmoid_pos_score_mean,
            sigmoid_neg_score_mean=sigmoid_neg_score_mean,
            sigmoid_pos_margin_mean=sigmoid_pos_margin_mean,
            sigmoid_neg_margin_mean=sigmoid_neg_margin_mean,
        )
