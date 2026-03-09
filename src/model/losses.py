from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from src.model.sigmoid_pairwise import (
    SigmoidPairwiseConfig,
    SigmoidPairwiseOutputs,
    SigmoidPairwiseState,
)


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
_DistillLossFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]
_RegLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LossComputer(nn.Module):
    # --- Special methods ---
    def __init__(
        self,
        *,
        loss_type: str,
        temperature: float,
        distill_enabled: bool,
        include_main_loss_when_distilling: bool,
        distill_losses: list[tuple[str, float]],
        reg_query_weight: float,
        reg_doc_weight: float,
        reg_type: str,
        reg_paper_faithful: bool,
        in_batch_weight: float = 1.0,
        pairwise_weight: float = 1.0,
        sigmoid_config: SigmoidPairwiseConfig | None = None,
    ) -> None:
        super().__init__()
        self.loss_type: str = loss_type.replace("-", "_").lower()
        self.temperature: float = float(temperature)
        self.distill_enabled: bool = bool(distill_enabled)
        self.include_main_loss_when_distilling: bool = bool(
            include_main_loss_when_distilling
        )
        self.reg_query_weight: float = float(reg_query_weight)
        self.reg_doc_weight: float = float(reg_doc_weight)
        self.reg_type: str = str(reg_type).lower()
        self.reg_paper_faithful: bool = bool(reg_paper_faithful)
        self.in_batch_weight: float = float(in_batch_weight)
        self.pairwise_weight: float = float(pairwise_weight)
        self.sigmoid_config: SigmoidPairwiseConfig | None = sigmoid_config
        if self.loss_type == "sigmoid_pairwise_hard" and self.sigmoid_config is None:
            raise ValueError(
                "sigmoid_config must be provided for sigmoid_pairwise_hard."
            )
        self._neg_inf: torch.Tensor
        self.register_buffer(
            "_neg_inf",
            torch.tensor(float("-inf"), dtype=torch.float32),
            persistent=False,
        )
        self._sigmoid_state: SigmoidPairwiseState | None = (
            None
            if sigmoid_config is None
            else SigmoidPairwiseState(sigmoid_config)
        )
        self._main_loss_fn: _MainLossFn | None = (
            None
            if self.loss_type == "sigmoid_pairwise_hard"
            else self._resolve_main_loss_fn(self.loss_type)
        )
        (
            self._distill_mse_weight,
            self._distill_kl_weight,
            self._distill_margin_mse_weight,
        ) = (
            self._resolve_distill_loss_weights(distill_losses)
            if self.distill_enabled
            else (0.0, 0.0, 0.0)
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
        *,
        include_same_query_negatives: bool = True,
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
        flat_pos_mask: torch.Tensor = (pos_mask & doc_mask).view(-1)
        doc_owner: torch.Tensor = torch.arange(
            bsz, device=doc_mask.device
        ).repeat_interleave(doc_count)
        query_ids: torch.Tensor = torch.arange(bsz, device=doc_mask.device).unsqueeze(1)
        same_query_mask: torch.Tensor = doc_owner.unsqueeze(0) == query_ids
        if include_same_query_negatives:
            same_query_docs: torch.Tensor = same_query_mask & flat_doc_mask.unsqueeze(0)
        else:
            same_query_docs = same_query_mask & flat_pos_mask.unsqueeze(0)
        other_query_pos: torch.Tensor = (~same_query_mask) & flat_pos_mask.unsqueeze(0)
        in_batch_doc_mask: torch.Tensor = same_query_docs | other_query_pos

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
            self._compute_in_batch_scores(
                q_reps,
                doc_reps,
                pos_mask,
                doc_mask,
            )
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

    def _main_loss_in_batch_plus_pairwise(
        self,
        pairwise_scores: torch.Tensor,
        q_reps: torch.Tensor,
        doc_reps: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pairwise_loss: torch.Tensor = multi_positive_contrastive_loss(
            pairwise_scores,
            pos_mask,
            doc_mask,
            temperature=self.temperature,
            neg_inf=self._neg_inf,
        )
        in_batch_scores: torch.Tensor
        in_batch_pos_mask: torch.Tensor
        in_batch_doc_mask: torch.Tensor
        in_batch_scores, in_batch_pos_mask, in_batch_doc_mask = (
            self._compute_in_batch_scores(
                q_reps,
                doc_reps,
                pos_mask,
                doc_mask,
                include_same_query_negatives=False,
            )
        )
        in_batch_loss: torch.Tensor = multi_positive_contrastive_loss(
            in_batch_scores,
            in_batch_pos_mask,
            in_batch_doc_mask,
            temperature=self.temperature,
            neg_inf=self._neg_inf,
        )
        combined_loss: torch.Tensor = (
            self.in_batch_weight * in_batch_loss
            + self.pairwise_weight * pairwise_loss
        )
        return combined_loss, pairwise_loss, in_batch_loss

    def _distill_loss_noop(
        self,
        scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        _ = teacher_scores, pos_mask, doc_mask
        return scores.new_zeros(())

    def _distill_loss_mse(
        self,
        scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        _ = pos_mask
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
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        _ = pos_mask
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

    def _distill_loss_margin_mse(
        self,
        scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MarginMSE over all valid positive/negative pairs per query."""
        valid_teacher: torch.Tensor = torch.isfinite(teacher_scores)
        valid_docs: torch.Tensor = doc_mask & valid_teacher
        pos_valid: torch.Tensor = pos_mask & valid_docs
        neg_valid: torch.Tensor = (~pos_mask) & valid_docs

        # Build every (positive, negative) pair in each row:
        # pair_mask[b, i, j] == True means doc i is a valid positive and doc j
        # is a valid negative for query b.
        pair_mask: torch.Tensor = pos_valid.unsqueeze(2) & neg_valid.unsqueeze(1)
        pair_mask_float: torch.Tensor = pair_mask.to(dtype=scores.dtype)
        pair_count: torch.Tensor = pair_mask_float.sum().clamp(min=1.0)

        safe_teacher: torch.Tensor = torch.where(valid_docs, teacher_scores, scores)
        student_margin: torch.Tensor = scores.unsqueeze(2) - scores.unsqueeze(1)
        teacher_margin: torch.Tensor = (
            safe_teacher.unsqueeze(2) - safe_teacher.unsqueeze(1)
        )
        margin_error: torch.Tensor = (student_margin - teacher_margin).pow(2)
        loss_sum: torch.Tensor = (margin_error * pair_mask_float).sum()
        return loss_sum / pair_count

    def _reg_loss_noop(
        self, reps: torch.Tensor, row_mask: torch.Tensor
    ) -> torch.Tensor:
        _ = row_mask
        return reps.new_zeros(())

    def _reg_loss_l1(self, reps: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
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
        if loss_type == "in_batch_plus_pairwise":
            return self._main_loss_in_batch_plus_pairwise
        raise ValueError(f"Unsupported loss type: {loss_type}")

    def _resolve_distill_loss_weights(
        self, distill_losses: list[tuple[str, float]]
    ) -> tuple[float, float, float]:
        mse_weight: float = 0.0
        kl_weight: float = 0.0
        margin_mse_weight: float = 0.0
        for loss_type, weight in distill_losses:
            loss_key: str = str(loss_type).replace("-", "_").lower()
            loss_weight: float = float(weight)
            if loss_weight == 0.0:
                continue
            if loss_key == "mse":
                mse_weight += loss_weight
                continue
            if loss_key == "kl":
                kl_weight += loss_weight
                continue
            if loss_key == "margin_mse":
                margin_mse_weight += loss_weight
                continue
            raise ValueError(f"Unsupported distillation loss: {loss_type}")
        if (
            mse_weight == 0.0
            and kl_weight == 0.0
            and margin_mse_weight == 0.0
        ):
            raise ValueError(
                "distill.losses must include at least one non-zero weight loss "
                "when distill is enabled."
            )
        return mse_weight, kl_weight, margin_mse_weight

    def _resolve_reg_loss_fn(self, reg_type: str, *, enabled: bool) -> _RegLossFn:
        if not enabled:
            return self._reg_loss_noop
        if reg_type == "l1":
            return self._reg_loss_l1
        if reg_type == "flops":
            return self._reg_loss_flops
        raise ValueError(f"Unsupported regularization: {reg_type}")

    # --- Public methods ---
    @property
    def has_trainable_main_loss_parameters(self) -> bool:
        return self._sigmoid_state is not None

    def clamp_parameters(self) -> None:
        if self._sigmoid_state is None:
            return
        self._sigmoid_state.clamp_parameters()

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
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        pairwise_scores: torch.Tensor = self._compute_pairwise_scores(q_reps, doc_reps)
        zero_scalar: torch.Tensor = pairwise_scores.new_zeros(())
        loss: torch.Tensor
        pairwise_loss: torch.Tensor
        in_batch_loss: torch.Tensor
        use_main_loss: bool = (
            not self.distill_enabled or self.include_main_loss_when_distilling
        )
        sigmoid_outputs: SigmoidPairwiseOutputs | None = None
        if use_main_loss:
            if self.loss_type == "sigmoid_pairwise_hard":
                if self._sigmoid_state is None:
                    raise RuntimeError(
                        "sigmoid_pairwise_hard requires sigmoid state."
                    )
                sigmoid_outputs = self._sigmoid_state(
                    scores=pairwise_scores,
                    pos_mask=pos_mask,
                    doc_mask=doc_mask,
                )
                loss = sigmoid_outputs.loss
                pairwise_loss = sigmoid_outputs.loss
                in_batch_loss = torch.zeros_like(sigmoid_outputs.loss)
            else:
                if self._main_loss_fn is None:
                    raise RuntimeError(
                        f"Main loss function is not configured for {self.loss_type!r}."
                    )
                loss, pairwise_loss, in_batch_loss = self._main_loss_fn(
                    pairwise_scores, q_reps, doc_reps, pos_mask, doc_mask
                )
        else:
            # Distillation-only mode keeps retrieval objective disabled.
            zero = pairwise_scores.new_zeros(())
            loss = zero
            pairwise_loss = zero
            in_batch_loss = zero

        distill_loss: torch.Tensor = pairwise_scores.new_zeros(())
        distill_mse_loss: torch.Tensor = pairwise_scores.new_zeros(())
        distill_kl_loss: torch.Tensor = pairwise_scores.new_zeros(())
        distill_margin_mse_loss: torch.Tensor = pairwise_scores.new_zeros(())
        sigmoid_pos_loss: torch.Tensor = zero_scalar
        sigmoid_neg_loss: torch.Tensor = zero_scalar
        sigmoid_logit_scale: torch.Tensor = zero_scalar
        sigmoid_bias: torch.Tensor = zero_scalar
        sigmoid_pos_score_mean: torch.Tensor = zero_scalar
        sigmoid_neg_score_mean: torch.Tensor = zero_scalar
        sigmoid_pos_margin_mean: torch.Tensor = zero_scalar
        sigmoid_neg_margin_mean: torch.Tensor = zero_scalar
        if self._sigmoid_state is not None and self.loss_type == "sigmoid_pairwise_hard":
            if sigmoid_outputs is None:
                sigmoid_outputs = self._sigmoid_state(
                    scores=pairwise_scores,
                    pos_mask=pos_mask,
                    doc_mask=doc_mask,
                )
            sigmoid_pos_loss = sigmoid_outputs.pos_loss
            sigmoid_neg_loss = sigmoid_outputs.neg_loss
            sigmoid_logit_scale = sigmoid_outputs.logit_scale
            sigmoid_bias = sigmoid_outputs.bias
            sigmoid_pos_score_mean = sigmoid_outputs.pos_score_mean
            sigmoid_neg_score_mean = sigmoid_outputs.neg_score_mean
            sigmoid_pos_margin_mean = sigmoid_outputs.pos_margin_mean
            sigmoid_neg_margin_mean = sigmoid_outputs.neg_margin_mean
        if self.distill_enabled:
            if self._distill_mse_weight != 0.0:
                distill_mse_loss = self._distill_loss_mse(
                    pairwise_scores, teacher_scores, pos_mask, doc_mask
                )
                distill_loss = distill_loss + (
                    self._distill_mse_weight * distill_mse_loss
                )
            if self._distill_kl_weight != 0.0:
                distill_kl_loss = self._distill_loss_kl(
                    pairwise_scores, teacher_scores, pos_mask, doc_mask
                )
                distill_loss = distill_loss + (
                    self._distill_kl_weight * distill_kl_loss
                )
            if self._distill_margin_mse_weight != 0.0:
                distill_margin_mse_loss = self._distill_loss_margin_mse(
                    pairwise_scores, teacher_scores, pos_mask, doc_mask
                )
                distill_loss = distill_loss + (
                    self._distill_margin_mse_weight * distill_margin_mse_loss
                )
        loss = loss + distill_loss

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
        )
