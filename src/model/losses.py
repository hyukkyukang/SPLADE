from __future__ import annotations

import torch
from torch.nn import functional as F


def multi_positive_contrastive_loss(
    scores: torch.Tensor,
    pos_mask: torch.Tensor,
    doc_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    scaled_scores = scores / max(temperature, 1e-8)
    neg_inf = torch.finfo(scores.dtype).min
    scaled_scores = scaled_scores.masked_fill(~doc_mask, neg_inf)
    pos_scores = scaled_scores.masked_fill(~pos_mask, neg_inf)
    logsumexp_pos = torch.logsumexp(pos_scores, dim=1)
    logsumexp_all = torch.logsumexp(scaled_scores, dim=1)
    loss = -(logsumexp_pos - logsumexp_all)
    return loss.mean()


def distillation_loss(
    scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    doc_mask: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    mask = doc_mask & torch.isfinite(teacher_scores)
    if mask.sum() == 0:
        return torch.zeros([], device=scores.device, dtype=scores.dtype)
    if loss_type == "mse":
        diff = scores - teacher_scores
        return (diff.pow(2) * mask).sum() / mask.sum()
    if loss_type == "kl":
        scores_masked = scores.masked_fill(~doc_mask, -1e4)
        teacher_masked = teacher_scores.masked_fill(~doc_mask, -1e4)
        student_log_probs = F.log_softmax(scores_masked, dim=1)
        teacher_probs = F.softmax(teacher_masked, dim=1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        return (kl * mask).sum() / mask.sum()
    raise ValueError(f"Unsupported distillation loss: {loss_type}")


def regularization_loss(
    reps: torch.Tensor, reg_type: str, paper_faithful: bool
) -> torch.Tensor:
    if reg_type == "l1":
        if paper_faithful:
            return reps.abs().sum(dim=1).mean()
        return reps.abs().mean()
    if reg_type == "flops":
        mean_activation = reps.mean(dim=0)
        if paper_faithful:
            return torch.sum(mean_activation.pow(2))
        return torch.mean(mean_activation.pow(2))
    raise ValueError(f"Unsupported regularization: {reg_type}")
