"""LightningModule for full paper-faithful LENS training.

Ties together:
- :class:`src.model.pl_module.lens_encoder.LENSBiEncoder` (sub-batched forward)
- cross-device :func:`src.utils.dist.dist_gather_with_local_grad`
- LENS-specific losses in :mod:`src.model.losses`
- LoRA wrapping around a ``MistralBiForCausalLM`` backbone with a clustered
  ``lm_head``

Kept deliberately small and self-contained so we can iterate on the training
loop without fighting the broader SPLADETrainingModule.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import lightning as L
import torch
from torch import Tensor

from src.model.losses import (
    lens_kl_distill_loss,
    lens_local_contrastive_loss,
    lens_stride_in_batch_contrastive_loss,
)
from src.model.pl_module.lens_encoder import LENSBiEncoder
from src.utils.dist import dist_gather_with_local_grad


class LENSTrainingModule(L.LightningModule):
    """Bi-encoder training on the multi-task bge-full-data mix."""

    def __init__(
        self,
        *,
        encoder: LENSBiEncoder,
        temperature: float = 0.02,
        cross_device_negatives: bool = True,
        distill_enabled: bool = True,
        distill_weight: float = 1.0,
        lr: float = 1e-4,
        warmup_steps: int = 100,
        total_steps: Optional[int] = None,
        weight_decay: float = 0.0,
        deepspeed_enabled: bool = False,
        val_task: Optional[str] = None,
        val_tokenizer: Any = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.temperature: float = float(temperature)
        self.cross_device_negatives: bool = bool(cross_device_negatives)
        self.distill_enabled: bool = bool(distill_enabled)
        self.distill_weight: float = float(distill_weight)
        self._opt_lr: float = float(lr)
        self._warmup_steps: int = int(warmup_steps)
        self._total_steps: Optional[int] = total_steps
        self._weight_decay: float = float(weight_decay)
        # Flag set at construction time (configure_optimizers / clip can run
        # before self.trainer is wired up). Driven by cfg.training.deepspeed
        # via train_lens.py:_is_deepspeed_enabled.
        self._deepspeed_enabled: bool = bool(deepspeed_enabled)
        # Online validation: if val_task is set, on_validation_epoch_end will
        # run an MTEB single-task eval against the live model. val_tokenizer
        # MUST be passed in (we can't extract it from the PEFT-wrapped backbone
        # cleanly).
        self._val_task: Optional[str] = val_task
        self._val_tokenizer = val_tokenizer
        self.save_hyperparameters(ignore=["encoder", "val_tokenizer"])

    # --- Lightning hooks -----------------------------------------------------

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        q_features = batch["query"]
        p_features = batch["passage"]
        messages: str = str(batch.get("messages", "normal"))
        group_size: int = int(batch.get("train_group_size", 8))
        teacher_scores: Optional[Tensor] = batch.get("teacher_scores")

        q_reps: Tensor = self.encoder.encode(q_features)
        p_reps: Tensor = self.encoder.encode(p_features)

        is_not_in_batch: bool = messages == "not in-batch"

        if self.cross_device_negatives and not is_not_in_batch:
            q_reps = dist_gather_with_local_grad(q_reps)
            p_reps = dist_gather_with_local_grad(p_reps)
            # Note: teacher_scores is per-query; we only gather when needed below.

        loss_main: Tensor
        scores: Tensor
        if is_not_in_batch:
            loss_main, scores = lens_local_contrastive_loss(
                q_reps, p_reps,
                temperature=self.temperature,
                train_group_size=group_size,
            )
        else:
            loss_main, scores = lens_stride_in_batch_contrastive_loss(
                q_reps, p_reps,
                temperature=self.temperature,
                train_group_size=group_size,
            )

        total_loss: Tensor = loss_main
        if self.distill_enabled and teacher_scores is not None and not is_not_in_batch:
            # teacher_scores on this rank only (not gathered) — slice after gather.
            if self.cross_device_negatives:
                teacher_scores_gathered = dist_gather_with_local_grad(
                    teacher_scores.detach().to(q_reps.device)
                )
            else:
                teacher_scores_gathered = teacher_scores.detach().to(q_reps.device)
            distill_loss: Tensor = lens_kl_distill_loss(
                q_reps, p_reps, scores, teacher_scores_gathered,
                train_group_size=group_size,
            )
            total_loss = total_loss + self.distill_weight * distill_loss
            self.log("train/distill_loss", distill_loss, prog_bar=False,
                     on_step=True, on_epoch=False, sync_dist=False)

        self.log("train/loss", total_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/contrastive_loss", loss_main, prog_bar=False,
                 on_step=True, on_epoch=False, sync_dist=False)
        self.log("train/task_type", float(hash(str(batch.get("task_type", ""))) % 1000),
                 prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        return total_loss

    # --- Online validation -------------------------------------------------

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """No-op. The data module's val_dataloader emits a single dummy batch
        purely to trigger ``on_validation_epoch_end`` -- the real eval runs
        there, against the live model, on rank 0 only.
        """
        return None

    def on_validation_epoch_end(self) -> None:
        if not self._val_task:
            return
        if self._val_tokenizer is None:
            self.print(
                "[val] val_task=%s set but no tokenizer provided; skipping eval"
                % self._val_task
            )
            return
        # Run heavy eval on rank 0 only; other ranks receive the broadcast
        # below so all of them log the same number and Lightning's
        # EarlyStopping callback sees a consistent value.
        score: Optional[float] = None
        if self.global_rank == 0:
            from src.utils.lens_inline_validation import run_single_task_eval
            # Free any non-resident PyTorch cache so the eval encode has
            # headroom on top of training's near-full memory footprint.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.encoder.model.eval()
            try:
                score = run_single_task_eval(
                    model=self.encoder.model,
                    tokenizer=self._val_tokenizer,
                    device=self.device,
                    task_name=self._val_task,
                    batch_size=16,    # conservative; training is at ~99% memory
                    max_length=256,   # NFCorpus docs are short medical abstracts
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.print(f"[val] {self._val_task} eval crashed: {exc}")
                score = None
            finally:
                self.encoder.model.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Broadcast the float score across ranks so every rank logs the same.
        # NaN means "no signal" -- EarlyStopping is configured to ignore NaNs.
        device = self.device if torch.cuda.is_available() else torch.device("cpu")
        if score is None:
            score_tensor = torch.tensor(float("nan"), device=device)
        else:
            score_tensor = torch.tensor(float(score), device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(score_tensor, src=0)

        self.log(
            "val_ndcg10",
            score_tensor.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.print(
            f"[val][step={self.global_step}] {self._val_task} "
            f"nDCG@10 = {score_tensor.item():.4f}"
        )

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Manual clipping that bypasses Lightning's fused-optimizer guard.

        Lightning raises ``RuntimeError("...does not allow for gradient
        clipping because it performs unscaling of gradients internally...")``
        whenever ``fused=True`` is set on AdamW, regardless of whether a
        GradScaler is actually in use. Under bf16-mixed there is no scaler,
        so the unscaling concern doesn't apply and ``clip_grad_norm_`` is
        safe to call directly on the parameters.

        Under DeepSpeed, clipping is owned by the DeepSpeed engine via the
        ``gradient_clipping`` config field, and intercepting it here would
        either double-clip or step on partitioned-grad invariants (ZeRO-3
        only ever exposes a slice of the gradient on each rank). Skip in
        that case.
        """
        if self._deepspeed_enabled:
            return
        if gradient_clip_val is None or gradient_clip_val <= 0:
            return
        params = [
            p
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not params:
            return
        algo = (gradient_clip_algorithm or "norm").lower()
        if algo == "value":
            torch.nn.utils.clip_grad_value_(params, clip_value=float(gradient_clip_val))
        else:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(gradient_clip_val))

    def configure_optimizers(self) -> dict[str, Any]:
        params = [p for p in self.parameters() if p.requires_grad]
        # Profiling on phase 1 config showed foreach AdamW spent 3.18 s per
        # call (27% of wall time) launching one kernel per LoRA tensor.
        # Fused AdamW collapses that into a single kernel. Lightning blocks
        # fused + gradient_clip_val by default, but we override
        # ``configure_gradient_clipping`` above with a direct
        # ``clip_grad_norm_`` call that's safe under bf16-mixed.
        # configure_optimizers runs before Lightning moves params to GPU,
        # so we gate on CUDA availability rather than per-param device.
        # Under DeepSpeed, the optimizer is wrapped by DeepSpeed's own
        # FusedAdam during ``setup_optimizers``; passing ``fused=True`` to
        # the inner ``torch.optim.AdamW`` then double-counts CUDA-graph
        # capture and breaks ZeRO-3's partitioned-state assumptions. Use a
        # plain (foreach) AdamW and let DeepSpeed re-wrap it.
        use_fused = torch.cuda.is_available() and not self._deepspeed_enabled
        optimizer = torch.optim.AdamW(
            params,
            lr=self._opt_lr,
            weight_decay=self._weight_decay,
            fused=use_fused,
        )
        if self._total_steps is None:
            return {"optimizer": optimizer}

        def lr_lambda(step: int) -> float:
            if step < self._warmup_steps:
                return float(step) / max(1, self._warmup_steps)
            progress = (step - self._warmup_steps) / max(
                1, self._total_steps - self._warmup_steps
            )
            return max(0.0, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


__all__ = ["LENSTrainingModule"]
