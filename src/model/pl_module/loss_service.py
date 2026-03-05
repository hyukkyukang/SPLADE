from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig

from src.model.losses import LossComputer


@dataclass(frozen=True)
class LossComputationOutputs:
    loss: torch.Tensor
    pairwise_scores: torch.Tensor
    pairwise_loss: torch.Tensor
    in_batch_loss: torch.Tensor
    distill_loss: torch.Tensor
    distill_losses: dict[str, torch.Tensor]
    q_reg: torch.Tensor
    d_reg: torch.Tensor
    lambda_scale_value: float
    lambda_scale: torch.Tensor
    reg_query_lambda: torch.Tensor
    reg_doc_lambda: torch.Tensor


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

        reg_weight_value: float | None = self.reg_cfg.weight
        if reg_weight_value is None:
            self.reg_query_weight: float = float(self.reg_cfg.query_weight)
            self.reg_doc_weight: float = float(self.reg_cfg.doc_weight)
        else:
            # Single lambda applied to both query and document regularization.
            self.reg_query_weight = float(reg_weight_value)
            self.reg_doc_weight = float(reg_weight_value)

    def _resolve_distill_losses(self) -> list[tuple[str, float]]:
        distill_losses_cfg: Any | None = self.distill_cfg.losses
        distill_losses: list[tuple[str, float]] = []
        if distill_losses_cfg is None:
            return distill_losses
        for entry in distill_losses_cfg:
            loss_type: str = str(entry.type)
            loss_weight: float = float(entry.weight)
            distill_losses.append((loss_type, loss_weight))
        return distill_losses

    def build_loss_computer(self) -> LossComputer:
        return LossComputer(
            loss_type=self.loss_type,
            temperature=self.temperature,
            distill_enabled=bool(self.distill_cfg.enabled),
            include_main_loss_when_distilling=bool(
                self.distill_cfg.get("include_main_loss", False)
            ),
            distill_losses=self._resolve_distill_losses(),
            reg_query_weight=self.reg_query_weight,
            reg_doc_weight=self.reg_doc_weight,
            reg_type=str(self.reg_cfg.type),
            reg_paper_faithful=bool(self.reg_cfg.paper_faithful),
            in_batch_weight=self.in_batch_weight,
            pairwise_weight=self.pairwise_weight,
        )

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

    def compute_loss(
        self,
        *,
        loss_computer: LossComputer,
        q_reps: torch.Tensor,
        doc_reps: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        teacher_scores: torch.Tensor,
        stage: str,
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
            distill_losses,
            q_reg,
            d_reg,
        ) = loss_computer(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            lambda_scale=lambda_scale,
            # Keep validation candidate pools query-local for stable comparisons.
            main_loss_type_override="pairwise" if stage == "val" else None,
        )
        return LossComputationOutputs(
            loss=loss,
            pairwise_scores=pairwise_scores,
            pairwise_loss=pairwise_loss,
            in_batch_loss=in_batch_loss,
            distill_loss=distill_loss,
            distill_losses=distill_losses,
            q_reg=q_reg,
            d_reg=d_reg,
            lambda_scale_value=lambda_scale_value,
            lambda_scale=lambda_scale,
            reg_query_lambda=reg_query_lambda,
            reg_doc_lambda=reg_doc_lambda,
        )
