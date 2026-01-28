from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar, cast

import lightning as L
import torch
from omegaconf import DictConfig

from src.model.losses import LossComputer
from src.model.retriever.sparse.neural.splade_model import SpladeModel
from src.utils.model_utils import build_splade_model

logger: logging.Logger = logging.getLogger("SPLADETrainingModule")
_TCallable = TypeVar("_TCallable", bound=Callable[..., Any])


def _dynamo_disable(fn: _TCallable) -> _TCallable:
    """Keep logging helpers out of torch.compile graphs."""
    disable_fn: Any = getattr(getattr(torch, "_dynamo", None), "disable", None)
    if callable(disable_fn):
        return cast(_TCallable, disable_fn(fn))
    return fn


class SPLADETrainingModule(L.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg

        # Build the encoder with a dtype appropriate for the device.
        self.model: SpladeModel = build_splade_model(cfg, use_cpu=cfg.training.use_cpu)
        # Transformers from_pretrained defaults to eval; ensure training mode here.
        self.model.train()
        compile_enabled: bool = bool(getattr(cfg.training, "torch_compile", False))
        compile_available: bool = hasattr(torch, "compile")
        if compile_enabled and not compile_available:
            logger.warning(
                "torch.compile is not available in this PyTorch build; continuing "
                "without compilation."
            )
        if compile_enabled and compile_available:
            encoder_module: torch.nn.Module | None = getattr(
                self.model, "encoder", None
            )
            # Compile only the encoder to keep Lightning bookkeeping eager.
            if encoder_module is not None:
                self.model.encoder = torch.compile(encoder_module)
            else:
                self.model = torch.compile(self.model)

        self.temperature: float = float(cfg.training.temperature)
        self.distill_cfg: DictConfig = cfg.training.distill
        self.reg_cfg: DictConfig = cfg.training.regularization
        self.loss_cfg: DictConfig = cfg.training.loss
        # Loss type controls pairwise vs in-batch negatives.
        self.loss_type: str = str(getattr(self.loss_cfg, "type", "pairwise")).lower()
        self.pairwise_weight: float = float(
            getattr(self.loss_cfg, "pairwise_weight", 1.0)
        )
        self.in_batch_weight: float = float(
            getattr(self.loss_cfg, "in_batch_weight", 1.0)
        )
        self.loss_computer: LossComputer = LossComputer(
            loss_type=self.loss_type,
            temperature=self.temperature,
            pairwise_weight=self.pairwise_weight,
            in_batch_weight=self.in_batch_weight,
            distill_enabled=bool(self.distill_cfg.enabled),
            distill_weight=float(self.distill_cfg.weight),
            distill_loss_type=str(self.distill_cfg.loss),
            reg_query_weight=float(self.reg_cfg.query_weight),
            reg_doc_weight=float(self.reg_cfg.doc_weight),
            reg_type=str(self.reg_cfg.type),
            reg_paper_faithful=bool(self.reg_cfg.paper_faithful),
        )
        if compile_enabled and compile_available:
            self.loss_computer = torch.compile(self.loss_computer)

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

        pos_mask: torch.Tensor = batch["pos_mask"]
        doc_mask: torch.Tensor = batch["doc_mask"]
        teacher_scores: torch.Tensor = batch["teacher_scores"]
        lambda_scale_value: float = self._lambda_schedule_multiplier()
        lambda_scale: torch.Tensor = torch.tensor(
            lambda_scale_value, device=q_reps.device, dtype=q_reps.dtype
        )
        loss: torch.Tensor
        pairwise_scores: torch.Tensor
        pairwise_loss: torch.Tensor
        in_batch_loss: torch.Tensor
        distill_loss: torch.Tensor
        q_reg: torch.Tensor
        d_reg: torch.Tensor
        (
            loss,
            pairwise_scores,
            pairwise_loss,
            in_batch_loss,
            distill_loss,
            q_reg,
            d_reg,
        ) = self.loss_computer(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            lambda_scale=lambda_scale,
        )

        metrics: dict[str, torch.Tensor] = {"loss": loss}
        if self.loss_type in {"pairwise", "both"}:
            metrics["pairwise_loss"] = pairwise_loss
        if self.loss_type in {"in_batch", "both"}:
            metrics["in_batch_loss"] = in_batch_loss
        if self.distill_cfg.enabled:
            metrics["distill_loss"] = distill_loss
        if self.reg_cfg.query_weight > 0:
            metrics["q_reg"] = q_reg
        if self.reg_cfg.doc_weight > 0:
            metrics["d_reg"] = d_reg

        if stage != "train":
            metrics.update(self._compute_mrr(pairwise_scores, pos_mask, doc_mask, k=10))

        return metrics

    @_dynamo_disable
    def _log_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        detached_metrics: dict[str, torch.Tensor] = {
            name: value.detach() for name, value in metrics.items()
        }
        self.log(
            "train_loss",
            detached_metrics["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if "distill_loss" in detached_metrics:
            self.log(
                "train_distill_loss",
                detached_metrics["distill_loss"],
                on_step=True,
                on_epoch=True,
            )
        if "q_reg" in detached_metrics:
            self.log(
                "train_q_reg", detached_metrics["q_reg"], on_step=True, on_epoch=True
            )
        if "d_reg" in detached_metrics:
            self.log(
                "train_d_reg", detached_metrics["d_reg"], on_step=True, on_epoch=True
            )
        if "pairwise_loss" in detached_metrics:
            self.log(
                "train_pairwise_loss",
                detached_metrics["pairwise_loss"],
                on_step=True,
                on_epoch=True,
            )
        if "in_batch_loss" in detached_metrics:
            self.log(
                "train_in_batch_loss",
                detached_metrics["in_batch_loss"],
                on_step=True,
                on_epoch=True,
            )

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

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        q: torch.Tensor = self.model.encode_queries(
            batch["query_input_ids"], batch["query_attention_mask"]
        )
        d: torch.Tensor = self.model.encode_docs(
            batch["doc_input_ids"], batch["doc_attention_mask"]
        )
        return {"q": q, "d": d}

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        metrics: dict[str, torch.Tensor] = self._training_step_shared(
            batch, stage="train"
        )
        self._log_metrics(metrics)
        return metrics["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        metrics: dict[str, torch.Tensor] = self._training_step_shared(
            batch, stage="val"
        )
        batch_size: int = int(batch["query_input_ids"].shape[0])
        val_loss: torch.Tensor = metrics["loss"].detach()
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        if "mrr10" in metrics:
            val_mrr10: torch.Tensor = metrics["mrr10"].detach()
            self.log(
                "val_mrr10",
                val_mrr10,
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
