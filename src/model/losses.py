from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


def multi_positive_contrastive_loss(
    scores: torch.Tensor,
    pos_mask: torch.Tensor,
    doc_mask: torch.Tensor,
    temperature: float,
    neg_inf: torch.Tensor,
) -> torch.Tensor:
    # Compute the contrastive loss in fp32 to avoid fp16 overflow.
    scores_fp32: torch.Tensor = scores.float()
    temperature_value: float = max(float(temperature), 1e-8)
    scaled_scores: torch.Tensor = scores_fp32 / temperature_value
    neg_inf_value: torch.Tensor = neg_inf.to(
        dtype=scores_fp32.dtype, device=scores_fp32.device
    )
    scaled_scores = scaled_scores.masked_fill(~doc_mask, neg_inf_value)
    pos_scores: torch.Tensor = scaled_scores.masked_fill(~pos_mask, neg_inf_value)
    logsumexp_pos: torch.Tensor = torch.logsumexp(pos_scores, dim=1)
    logsumexp_all: torch.Tensor = torch.logsumexp(scaled_scores, dim=1)
    loss: torch.Tensor = -(logsumexp_pos - logsumexp_all)
    return loss.mean()


_MainLossFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
_DistillLossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
_RegLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LossComputer(nn.Module):
    # --- Special methods ---
    def __init__(
        self,
        *,
        loss_type: str,
        temperature: float,
        distill_enabled: bool,
        distill_weight: float,
        distill_loss_type: str,
        reg_query_weight: float,
        reg_doc_weight: float,
        reg_type: str,
        reg_paper_faithful: bool,
    ) -> None:
        super().__init__()
        self.loss_type: str = loss_type.replace("-", "_").lower()
        self.temperature: float = float(temperature)
        self.distill_enabled: bool = bool(distill_enabled)
        self.distill_weight: float = float(distill_weight)
        self.distill_loss_type: str = str(distill_loss_type).lower()
        self.reg_query_weight: float = float(reg_query_weight)
        self.reg_doc_weight: float = float(reg_doc_weight)
        self.reg_type: str = str(reg_type).lower()
        self.reg_paper_faithful: bool = bool(reg_paper_faithful)
        self._neg_inf: torch.Tensor
        self.register_buffer(
            "_neg_inf", torch.tensor(float("-inf"), dtype=torch.float32), persistent=False
        )
        self._main_loss_fn: _MainLossFn = self._resolve_main_loss_fn(self.loss_type)
        self._distill_loss_fn: _DistillLossFn = (
            self._resolve_distill_loss_fn(self.distill_loss_type)
            if self.distill_enabled
            else self._distill_loss_noop
        )
        self._reg_query_fn: _RegLossFn = self._resolve_reg_loss_fn(
            self.reg_type, enabled=self.reg_query_weight > 0
        )
        self._reg_doc_fn: _RegLossFn = self._resolve_reg_loss_fn(
            self.reg_type, enabled=self.reg_doc_weight > 0
        )

    # --- Protected methods ---
    def _compute_pairwise_scores(
        self, q_reps: torch.Tensor, doc_reps: torch.Tensor
    ) -> torch.Tensor:
        device_type: str = str(q_reps.device.type)
        q_reps_fp32: torch.Tensor = q_reps.float()
        doc_reps_fp32: torch.Tensor = doc_reps.float()
        # Ensure FP32 matmul to prevent AMP overflow.
        with torch.autocast(device_type=device_type, enabled=False):
            scores_fp32: torch.Tensor = torch.bmm(
                doc_reps_fp32, q_reps_fp32.unsqueeze(2)
            ).squeeze(2)
        return scores_fp32

    def _compute_in_batch_scores(
        self,
        q_reps: torch.Tensor,
        doc_reps: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz: int
        doc_count: int
        rep_dim: int
        bsz, doc_count, rep_dim = doc_reps.shape

        # Flatten docs so each query scores against all docs in the batch.
        flat_doc_reps: torch.Tensor = doc_reps.view(bsz * doc_count, rep_dim)
        device_type: str = str(q_reps.device.type)
        q_reps_fp32: torch.Tensor = q_reps.float()
        flat_doc_reps_fp32: torch.Tensor = flat_doc_reps.float()
        # Ensure FP32 matmul to prevent AMP overflow.
        with torch.autocast(device_type=device_type, enabled=False):
            scores: torch.Tensor = torch.matmul(
                q_reps_fp32, flat_doc_reps_fp32.transpose(0, 1)
            )
        # Broadcast valid-document mask across all queries.
        flat_doc_mask: torch.Tensor = doc_mask.view(-1)
        in_batch_doc_mask: torch.Tensor = flat_doc_mask.unsqueeze(0).expand(bsz, -1)

        # Build a per-query positive mask aligned with flattened docs.
        pos_mask_in_batch: torch.Tensor = torch.zeros(
            (bsz, bsz * doc_count),
            dtype=torch.bool,
            device=pos_mask.device,
        )
        doc_offsets: torch.Tensor = (
            torch.arange(bsz, device=pos_mask.device).unsqueeze(1) * doc_count
        )
        doc_indices: torch.Tensor = doc_offsets + torch.arange(
            doc_count, device=pos_mask.device
        ).unsqueeze(0)
        pos_mask_in_batch.scatter_(1, doc_indices, pos_mask)
        return scores, pos_mask_in_batch, in_batch_doc_mask

    def _main_loss_pairwise(
        self,
        pairwise_scores: torch.Tensor,
        q_reps: torch.Tensor,
        doc_reps: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = q_reps, doc_reps
        pairwise_loss: torch.Tensor = multi_positive_contrastive_loss(
            pairwise_scores,
            pos_mask,
            doc_mask,
            temperature=self.temperature,
            neg_inf=self._neg_inf,
        )
        in_batch_loss: torch.Tensor = torch.zeros_like(pairwise_loss)
        return pairwise_loss, pairwise_loss, in_batch_loss

    def _main_loss_in_batch(
        self,
        pairwise_scores: torch.Tensor,
        q_reps: torch.Tensor,
        doc_reps: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = pairwise_scores
        in_batch_scores: torch.Tensor
        in_batch_pos_mask: torch.Tensor
        in_batch_doc_mask: torch.Tensor
        in_batch_scores, in_batch_pos_mask, in_batch_doc_mask = (
            self._compute_in_batch_scores(q_reps, doc_reps, pos_mask, doc_mask)
        )
        in_batch_loss: torch.Tensor = multi_positive_contrastive_loss(
            in_batch_scores,
            in_batch_pos_mask,
            in_batch_doc_mask,
            temperature=self.temperature,
            neg_inf=self._neg_inf,
        )
        pairwise_loss: torch.Tensor = torch.zeros_like(in_batch_loss)
        return in_batch_loss, pairwise_loss, in_batch_loss

    def _distill_loss_noop(
        self,
        scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        _ = teacher_scores, doc_mask
        return scores.new_zeros(())

    def _distill_loss_mse(
        self,
        scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask: torch.Tensor = doc_mask & torch.isfinite(teacher_scores)
        mask_float: torch.Tensor = mask.to(dtype=scores.dtype)
        denom: torch.Tensor = mask_float.sum().clamp(min=1.0)
        # Replace masked teacher scores to avoid NaNs in the diff.
        safe_teacher: torch.Tensor = torch.where(mask, teacher_scores, scores)
        diff: torch.Tensor = scores - safe_teacher
        loss_sum: torch.Tensor = (diff.pow(2) * mask_float).sum()
        return loss_sum / denom

    def _distill_loss_kl(
        self,
        scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask: torch.Tensor = doc_mask & torch.isfinite(teacher_scores)
        mask_float: torch.Tensor = mask.to(dtype=scores.dtype)
        denom: torch.Tensor = mask_float.sum().clamp(min=1.0)
        scores_masked: torch.Tensor = scores.masked_fill(~mask, -1e4)
        teacher_masked: torch.Tensor = teacher_scores.masked_fill(~mask, -1e4)
        student_log_probs: torch.Tensor = F.log_softmax(scores_masked, dim=1)
        teacher_probs: torch.Tensor = F.softmax(teacher_masked, dim=1)
        kl: torch.Tensor = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        loss_sum: torch.Tensor = (kl * mask_float).sum()
        return loss_sum / denom

    def _reg_loss_noop(
        self, reps: torch.Tensor, row_mask: torch.Tensor
    ) -> torch.Tensor:
        _ = row_mask
        return reps.new_zeros(())

    def _reg_loss_l1(
        self, reps: torch.Tensor, row_mask: torch.Tensor
    ) -> torch.Tensor:
        mask: torch.Tensor = row_mask.to(dtype=torch.bool)
        mask_float: torch.Tensor = mask.to(dtype=reps.dtype)
        row_count: torch.Tensor = mask_float.sum().clamp(min=1.0)
        abs_reps: torch.Tensor = reps.abs()
        if self.reg_paper_faithful:
            per_row: torch.Tensor = abs_reps.sum(dim=1)
            masked_sum: torch.Tensor = (per_row * mask_float).sum()
            return masked_sum / row_count
        masked_sum: torch.Tensor = (abs_reps * mask_float.unsqueeze(1)).sum()
        denom: torch.Tensor = row_count * float(reps.shape[1])
        return masked_sum / denom

    def _reg_loss_flops(
        self, reps: torch.Tensor, row_mask: torch.Tensor
    ) -> torch.Tensor:
        mask: torch.Tensor = row_mask.to(dtype=torch.bool)
        mask_float: torch.Tensor = mask.to(dtype=reps.dtype)
        row_count: torch.Tensor = mask_float.sum().clamp(min=1.0)
        masked_sum: torch.Tensor = (reps * mask_float.unsqueeze(1)).sum(dim=0)
        mean_activation: torch.Tensor = masked_sum / row_count
        if self.reg_paper_faithful:
            return torch.sum(mean_activation.pow(2))
        return torch.mean(mean_activation.pow(2))

    def _resolve_main_loss_fn(self, loss_type: str) -> _MainLossFn:
        if loss_type == "pairwise":
            return self._main_loss_pairwise
        if loss_type == "in_batch":
            return self._main_loss_in_batch
        raise ValueError(f"Unsupported loss type: {loss_type}")

    def _resolve_distill_loss_fn(self, loss_type: str) -> _DistillLossFn:
        if loss_type == "mse":
            return self._distill_loss_mse
        if loss_type == "kl":
            return self._distill_loss_kl
        raise ValueError(f"Unsupported distillation loss: {loss_type}")

    def _resolve_reg_loss_fn(self, reg_type: str, *, enabled: bool) -> _RegLossFn:
        if not enabled:
            return self._reg_loss_noop
        if reg_type == "l1":
            return self._reg_loss_l1
        if reg_type == "flops":
            return self._reg_loss_flops
        raise ValueError(f"Unsupported regularization: {reg_type}")

    # --- Public methods ---
    def forward(
        self,
        q_reps: torch.Tensor,
        doc_reps: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        teacher_scores: torch.Tensor,
        lambda_scale: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        pairwise_scores: torch.Tensor = self._compute_pairwise_scores(q_reps, doc_reps)
        loss: torch.Tensor
        pairwise_loss: torch.Tensor
        in_batch_loss: torch.Tensor
        loss, pairwise_loss, in_batch_loss = self._main_loss_fn(
            pairwise_scores, q_reps, doc_reps, pos_mask, doc_mask
        )

        distill_loss: torch.Tensor = self._distill_loss_fn(
            pairwise_scores, teacher_scores, doc_mask
        )
        loss = loss + self.distill_weight * distill_loss

        query_row_mask: torch.Tensor = torch.ones(
            q_reps.shape[0], device=q_reps.device, dtype=torch.bool
        )
        q_reg: torch.Tensor = self._reg_query_fn(q_reps, query_row_mask)
        loss = loss + (self.reg_query_weight * lambda_scale) * q_reg

        flat_doc_reps: torch.Tensor = doc_reps.view(-1, doc_reps.shape[-1])
        flat_doc_mask: torch.Tensor = doc_mask.view(-1)
        d_reg: torch.Tensor = self._reg_doc_fn(flat_doc_reps, flat_doc_mask)
        loss = loss + (self.reg_doc_weight * lambda_scale) * d_reg

        return (
            loss,
            pairwise_scores,
            pairwise_loss,
            in_batch_loss,
            distill_loss,
            q_reg,
            d_reg,
        )
