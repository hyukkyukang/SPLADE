from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def multi_positive_contrastive_loss(
    scores: torch.Tensor,
    pos_mask: torch.Tensor,
    doc_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    # Compute the contrastive loss in fp32 to avoid fp16 overflow.
    scores_fp32: torch.Tensor = scores.float()
    temperature_value: float = max(float(temperature), 1e-8)
    scaled_scores: torch.Tensor = scores_fp32 / temperature_value
    neg_inf: float = torch.finfo(scores_fp32.dtype).min
    scaled_scores = scaled_scores.masked_fill(~doc_mask, neg_inf)
    pos_scores: torch.Tensor = scaled_scores.masked_fill(~pos_mask, neg_inf)
    logsumexp_pos: torch.Tensor = torch.logsumexp(pos_scores, dim=1)
    logsumexp_all: torch.Tensor = torch.logsumexp(scaled_scores, dim=1)
    loss: torch.Tensor = -(logsumexp_pos - logsumexp_all)
    return loss.mean()


def distillation_loss(
    scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    doc_mask: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    mask: torch.Tensor = doc_mask & torch.isfinite(teacher_scores)
    mask_float: torch.Tensor = mask.to(dtype=scores.dtype)
    denom: torch.Tensor = mask_float.sum().clamp(min=1.0)
    if loss_type == "mse":
        # Replace masked teacher scores to avoid NaNs in the diff.
        safe_teacher: torch.Tensor = torch.where(mask, teacher_scores, scores)
        diff: torch.Tensor = scores - safe_teacher
        loss_sum: torch.Tensor = (diff.pow(2) * mask_float).sum()
        return loss_sum / denom
    if loss_type == "kl":
        scores_masked: torch.Tensor = scores.masked_fill(~mask, -1e4)
        teacher_masked: torch.Tensor = teacher_scores.masked_fill(~mask, -1e4)
        student_log_probs: torch.Tensor = F.log_softmax(scores_masked, dim=1)
        teacher_probs: torch.Tensor = F.softmax(teacher_masked, dim=1)
        kl: torch.Tensor = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        loss_sum: torch.Tensor = (kl * mask_float).sum()
        return loss_sum / denom
    raise ValueError(f"Unsupported distillation loss: {loss_type}")


def regularization_loss(
    reps: torch.Tensor,
    reg_type: str,
    paper_faithful: bool,
    *,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mask: torch.Tensor
    if row_mask is None:
        mask = torch.ones(reps.shape[0], device=reps.device, dtype=torch.bool)
    else:
        mask = row_mask.to(dtype=torch.bool)
    mask_float: torch.Tensor = mask.to(dtype=reps.dtype)
    row_count: torch.Tensor = mask_float.sum().clamp(min=1.0)
    if reg_type == "l1":
        abs_reps: torch.Tensor = reps.abs()
        if paper_faithful:
            per_row: torch.Tensor = abs_reps.sum(dim=1)
            masked_sum: torch.Tensor = (per_row * mask_float).sum()
            return masked_sum / row_count
        masked_sum: torch.Tensor = (abs_reps * mask_float.unsqueeze(1)).sum()
        denom: torch.Tensor = row_count * float(reps.shape[1])
        return masked_sum / denom
    if reg_type == "flops":
        masked_sum: torch.Tensor = (reps * mask_float.unsqueeze(1)).sum(dim=0)
        mean_activation: torch.Tensor = masked_sum / row_count
        if paper_faithful:
            return torch.sum(mean_activation.pow(2))
        return torch.mean(mean_activation.pow(2))
    raise ValueError(f"Unsupported regularization: {reg_type}")


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

    # --- Protected methods ---
    def _compute_pairwise_scores(
        self, q_reps: torch.Tensor, doc_reps: torch.Tensor
    ) -> torch.Tensor:
        return torch.bmm(doc_reps, q_reps.unsqueeze(2)).squeeze(2)

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
        scores: torch.Tensor = torch.matmul(q_reps, flat_doc_reps.transpose(0, 1))

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

        if self.loss_type == "pairwise":
            pairwise_loss: torch.Tensor = multi_positive_contrastive_loss(
                pairwise_scores, pos_mask, doc_mask, temperature=self.temperature
            )
            in_batch_loss: torch.Tensor = torch.zeros_like(pairwise_loss)
            loss: torch.Tensor = pairwise_loss
        elif self.loss_type == "in_batch":
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
            )
            pairwise_loss: torch.Tensor = torch.zeros_like(in_batch_loss)
            loss: torch.Tensor = in_batch_loss
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        distill_loss: torch.Tensor = torch.zeros_like(loss)
        if self.distill_enabled:
            distill_loss = distillation_loss(
                pairwise_scores,
                teacher_scores,
                doc_mask,
                loss_type=self.distill_loss_type,
            )
            loss = loss + self.distill_weight * distill_loss

        q_reg: torch.Tensor = torch.zeros_like(loss)
        if self.reg_query_weight > 0:
            q_reg = regularization_loss(q_reps, self.reg_type, self.reg_paper_faithful)
            loss = loss + (self.reg_query_weight * lambda_scale) * q_reg

        d_reg: torch.Tensor = torch.zeros_like(loss)
        if self.reg_doc_weight > 0:
            flat_doc_reps: torch.Tensor = doc_reps.view(-1, doc_reps.shape[-1])
            flat_doc_mask: torch.Tensor = doc_mask.view(-1)
            d_reg = regularization_loss(
                flat_doc_reps,
                self.reg_type,
                self.reg_paper_faithful,
                row_mask=flat_doc_mask,
            )
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
