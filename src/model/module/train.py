from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar, cast

import lightning as L
import torch
from omegaconf import DictConfig

from src.metric.retrieval import RetrievalMetrics, resolve_k_list
from src.model.losses import LossComputer
from src.model.retriever.sparse.neural.splade_model import SpladeModel
from src.utils.logging import log_if_rank_zero
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
            log_if_rank_zero(
                logger,
                "torch.compile is not available in this PyTorch build; continuing "
                "without compilation.",
                level="warning",
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

        self.val_metrics_cfg: DictConfig | None = getattr(
            cfg.training, "validation_metrics", None
        )
        self.val_metrics_enabled: bool = bool(
            getattr(self.val_metrics_cfg, "enabled", False)
            if self.val_metrics_cfg is not None
            else False
        )
        self._val_query_offset: int = 0
        self.val_metric_collection: RetrievalMetrics | None = None
        if self.val_metrics_enabled:
            k_list: list[int] = resolve_k_list(
                getattr(self.val_metrics_cfg, "k_list", None)
            )
            self.val_metric_collection = RetrievalMetrics(
                dataset_name="",
                k_list=k_list,
                sync_on_compute=False,
            )

    def _training_step_shared(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
        *,
        return_reps: bool = False,
    ) -> (
        dict[str, torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ):
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

        if return_reps:
            return metrics, {
                "q_reps": q_reps,
                "doc_reps": doc_reps,
                "pos_mask": pos_mask,
                "doc_mask": doc_mask,
            }

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

    def on_validation_start(self) -> None:
        if self.val_metric_collection is None:
            return
        self._val_query_offset = 0
        self.val_metric_collection.reset()
        # Ensure metric buffers live on the same device as predictions.
        self.val_metric_collection.to(self.device)

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

    def _append_validation_metrics(
        self,
        q_reps: torch.Tensor,
        doc_reps: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> None:
        if self.val_metric_collection is None:
            return
        in_batch_scores: torch.Tensor
        in_batch_pos_mask: torch.Tensor
        in_batch_doc_mask: torch.Tensor
        in_batch_scores, in_batch_pos_mask, in_batch_doc_mask = (
            self.loss_computer._compute_in_batch_scores(
                q_reps=q_reps,
                doc_reps=doc_reps,
                pos_mask=pos_mask,
                doc_mask=doc_mask,
            )
        )

        batch_size: int = int(in_batch_scores.shape[0])
        base_offset: int = self._val_query_offset
        self._val_query_offset += batch_size
        world_size: int = int(self.trainer.world_size)
        global_rank: int = int(self.trainer.global_rank)

        for i in range(batch_size):
            valid_mask: torch.Tensor = in_batch_doc_mask[i]
            if not valid_mask.any():
                continue
            scores: torch.Tensor = (
                in_batch_scores[i][valid_mask]
                .float()
                .detach()
                .to(self.val_metric_collection._device_ref.device)
            )
            targets: torch.Tensor = (
                in_batch_pos_mask[i][valid_mask]
                .float()
                .detach()
                .to(self.val_metric_collection._device_ref.device)
            )
            global_query_idx: int = global_rank + world_size * (base_offset + i)
            indexes: torch.Tensor = torch.full(
                (scores.shape[0],),
                global_query_idx,
                dtype=torch.long,
                device=scores.device,
            )
            self.val_metric_collection.append(scores, targets, indexes)

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
        if self.val_metric_collection is None:
            metrics: dict[str, torch.Tensor] = self._training_step_shared(
                batch, stage="val"
            )
        else:
            metrics, rep_cache = self._training_step_shared(
                batch, stage="val", return_reps=True
            )
            self._append_validation_metrics(
                rep_cache["q_reps"],
                rep_cache["doc_reps"],
                rep_cache["pos_mask"],
                rep_cache["doc_mask"],
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

    def on_validation_epoch_end(self) -> None:
        if self.val_metric_collection is None:
            return
        has_data: bool = self.val_metric_collection.gather(
            world_size=self.trainer.world_size,
            all_gather_fn=self.all_gather if self.trainer.world_size > 1 else None,
        )
        if not has_data:
            log_if_rank_zero(
                logger, "No predictions accumulated during validation.", level="warning"
            )
            return

        if self.trainer.is_global_zero:
            metrics: dict[str, torch.Tensor] = self.val_metric_collection.compute()
            filtered_metrics: dict[str, torch.Tensor] = {
                f"val_{name}": value
                for name, value in metrics.items()
                if name.startswith(("nDCG_", "MRR_", "Recall_"))
            }
            if filtered_metrics:
                self.log_dict(
                    filtered_metrics,
                    sync_dist=False,
                    prog_bar=False,
                    rank_zero_only=True,
                )
        self.val_metric_collection.reset()

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
