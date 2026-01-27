from __future__ import annotations

from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig

from src.model.losses import (
    distillation_loss,
    multi_positive_contrastive_loss,
    regularization_loss,
)
from src.model.retriever.sparse.neural.splade_model import SpladeModel
from src.utils.model_utils import build_splade_model


class SPLADETrainingModule(L.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg

        # Build the encoder with a dtype appropriate for the device.
        self.model: SpladeModel = build_splade_model(cfg, use_cpu=cfg.training.use_cpu)

        self.temperature: float = float(cfg.training.temperature)
        self.distill_cfg: DictConfig = cfg.training.distill
        self.reg_cfg: DictConfig = cfg.training.regularization

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        q: torch.Tensor = self.model.encode_queries(
            batch["query_input_ids"], batch["query_attention_mask"]
        )
        d: torch.Tensor = self.model.encode_docs(
            batch["doc_input_ids"], batch["doc_attention_mask"]
        )
        return {"q": q, "d": d}

    def _compute_scores(
        self, q_reps: torch.Tensor, d_reps: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("bv,bnv->bn", q_reps, d_reps)

    def _training_step_shared(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> dict[str, torch.Tensor]:
        q_reps: torch.Tensor = self.model.encode_queries(
            batch["query_input_ids"], batch["query_attention_mask"]
        )
        doc_input_ids: torch.Tensor = batch["doc_input_ids"]
        doc_attention_mask: torch.Tensor = batch["doc_attention_mask"]

        bsz: int
        doc_count: int
        seq_len: int
        bsz, doc_count, seq_len = doc_input_ids.shape
        flat_docs: torch.Tensor = doc_input_ids.view(bsz * doc_count, seq_len)
        flat_masks: torch.Tensor = doc_attention_mask.view(bsz * doc_count, seq_len)

        flat_doc_reps: torch.Tensor = self.model.encode_docs(flat_docs, flat_masks)
        doc_reps: torch.Tensor = flat_doc_reps.view(bsz, doc_count, -1)

        scores: torch.Tensor = self._compute_scores(q_reps, doc_reps)

        pos_mask: torch.Tensor = batch["pos_mask"]
        doc_mask: torch.Tensor = batch["doc_mask"]

        loss: torch.Tensor = multi_positive_contrastive_loss(
            scores, pos_mask, doc_mask, temperature=self.temperature
        )

        metrics: dict[str, torch.Tensor] = {"loss": loss}

        lambda_scale: float = self._lambda_schedule_multiplier()

        if self.distill_cfg.enabled:
            teacher_scores: torch.Tensor = batch["teacher_scores"]
            distill: torch.Tensor = distillation_loss(
                scores,
                teacher_scores,
                doc_mask,
                loss_type=self.distill_cfg.loss,
            )
            loss = loss + self.distill_cfg.weight * distill
            metrics["distill_loss"] = distill

        if self.reg_cfg.query_weight > 0:
            q_reg: torch.Tensor = regularization_loss(
                q_reps, self.reg_cfg.type, self.reg_cfg.paper_faithful
            )
            loss = loss + (self.reg_cfg.query_weight * lambda_scale) * q_reg
            metrics["q_reg"] = q_reg

        if self.reg_cfg.doc_weight > 0:
            flat_doc_mask: torch.Tensor = doc_mask.view(-1)
            if flat_doc_mask.any():
                doc_reg: torch.Tensor = regularization_loss(
                    flat_doc_reps[flat_doc_mask],
                    self.reg_cfg.type,
                    self.reg_cfg.paper_faithful,
                )
                loss = loss + (self.reg_cfg.doc_weight * lambda_scale) * doc_reg
                metrics["d_reg"] = doc_reg

        metrics["loss"] = loss

        if stage != "train":
            metrics.update(self._compute_mrr(scores, pos_mask, doc_mask, k=10))

        return metrics

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        metrics: dict[str, torch.Tensor] = self._training_step_shared(
            batch, stage="train"
        )
        self.log(
            "train_loss", metrics["loss"], on_step=True, on_epoch=True, prog_bar=True
        )
        if "distill_loss" in metrics:
            self.log(
                "train_distill_loss",
                metrics["distill_loss"],
                on_step=True,
                on_epoch=True,
            )
        if "q_reg" in metrics:
            self.log("train_q_reg", metrics["q_reg"], on_step=True, on_epoch=True)
        if "d_reg" in metrics:
            self.log("train_d_reg", metrics["d_reg"], on_step=True, on_epoch=True)
        return metrics["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        metrics: dict[str, torch.Tensor] = self._training_step_shared(
            batch, stage="val"
        )
        batch_size: int = int(batch["query_input_ids"].shape[0])
        self.log(
            "val_loss",
            metrics["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        if "mrr10" in metrics:
            self.log(
                "val_mrr10",
                metrics["mrr10"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

        if self.cfg.training.scheduler == "linear":
            from transformers import get_linear_schedule_with_warmup

            scheduler: Any = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.training.warmup_steps,
                num_training_steps=self.cfg.training.max_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer

    def _lambda_schedule_multiplier(self) -> float:
        schedule_steps: int | None = getattr(self.reg_cfg, "schedule_steps", 0)
        if schedule_steps is None:
            return 1.0
        schedule_steps = int(schedule_steps)
        if schedule_steps <= 0:
            return 1.0
        step: int = max(int(self.global_step), 0)
        progress: float = min(step, schedule_steps) / float(schedule_steps)
        return progress * progress

    @staticmethod
    def _compute_mrr(
        scores: torch.Tensor, pos_mask: torch.Tensor, doc_mask: torch.Tensor, k: int
    ) -> dict[str, torch.Tensor]:
        # Mask padded documents before ranking.
        # Compute in fp32 to avoid fp16 overflow on large negative masks.
        scores_fp32: torch.Tensor = scores.float()
        masked_scores: torch.Tensor = scores_fp32.masked_fill(
            ~doc_mask, torch.finfo(scores_fp32.dtype).min
        )
        topk_indices: torch.Tensor = torch.topk(
            masked_scores, k=min(k, scores.size(1)), dim=1
        ).indices
        # Gather positives within the ranked list for vectorized MRR.
        topk_pos_mask: torch.Tensor = pos_mask.gather(1, topk_indices)
        has_positive: torch.Tensor = topk_pos_mask.any(dim=1)
        first_positive: torch.Tensor = torch.argmax(
            topk_pos_mask.to(torch.int64), dim=1
        )
        rank: torch.Tensor = first_positive + 1
        mrr_values: torch.Tensor = torch.where(
            has_positive,
            1.0 / rank.to(dtype=masked_scores.dtype),
            torch.zeros_like(rank, dtype=masked_scores.dtype),
        )
        return {"mrr10": mrr_values.mean()}
