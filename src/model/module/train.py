from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, TypeVar, cast

import lightning as L
import torch
from omegaconf import DictConfig
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

from src.metric.retrieval import RetrievalMetrics, resolve_k_list
from src.model.losses import LossComputer
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import log_if_rank_zero
from src.utils.model_utils import build_splade_model
from src.utils.sparse_encoder import (
    SparseEncoderCache,
    build_sparse_encoder_cache,
    update_sparse_encoder_cache,
)

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
        compile_enabled: bool = bool(cfg.training.torch_compile)
        # Loss compilation is optional to avoid fragile Inductor/Triton paths.
        compile_loss: bool = bool(cfg.training.torch_compile_loss)
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
        self.loss_type: str = str(self.loss_cfg.type).lower()
        reg_weight_value: float | None = self.reg_cfg.weight
        if reg_weight_value is None:
            self.reg_query_weight = float(self.reg_cfg.query_weight)
            self.reg_doc_weight = float(self.reg_cfg.doc_weight)
        else:
            # Single lambda applied to both query and document regularization.
            self.reg_query_weight: float = float(reg_weight_value)
            self.reg_doc_weight: float = float(reg_weight_value)

        self.loss_computer: LossComputer = LossComputer(
            loss_type=self.loss_type,
            temperature=self.temperature,
            distill_enabled=bool(self.distill_cfg.enabled),
            distill_weight=float(self.distill_cfg.weight),
            distill_loss_type=str(self.distill_cfg.loss),
            reg_query_weight=self.reg_query_weight,
            reg_doc_weight=self.reg_doc_weight,
            reg_type=str(self.reg_cfg.type),
            reg_paper_faithful=bool(self.reg_cfg.paper_faithful),
        )
        if compile_enabled and compile_available and compile_loss:
            self.loss_computer = torch.compile(self.loss_computer)

        self.val_metrics_cfg: DictConfig = cfg.training.validation_metrics
        self.val_metrics_enabled: bool = bool(self.val_metrics_cfg.enabled)
        self._val_query_offset: int = 0
        self.val_metric_collection: RetrievalMetrics | None = None
        if self.val_metrics_enabled:
            k_list: list[int] = resolve_k_list(self.val_metrics_cfg.k_list)
            self.val_metric_collection = RetrievalMetrics(
                dataset_name="",
                k_list=k_list,
                sync_on_compute=False,
            )

        nanobeir_cfg: DictConfig | None = getattr(cfg, "nanobeir", None)
        self.nanobeir_enabled: bool = False
        self.nanobeir_run_every_n_val: int = 1
        self.nanobeir_batch_size: int = int(cfg.testing.batch_size)
        self.nanobeir_save_json: bool = False
        self.nanobeir_dataset_names: list[str] = []
        self.nanobeir_use_cpu: bool = bool(cfg.training.use_cpu)
        base_device_id: int | None = cfg.training.device_id
        self.nanobeir_device_id: int | None = (
            None if base_device_id is None else int(base_device_id)
        )
        self._nanobeir_val_counter: int = 0
        self._nanobeir_cache: SparseEncoderCache | None = None
        self._nanobeir_cache_device: torch.device | None = None
        self._nanobeir_evaluator: SparseNanoBEIREvaluator | None = None
        self._nanobeir_evaluator_datasets: list[str] = []
        self._nanobeir_evaluator_batch_size: int = int(self.nanobeir_batch_size)
        if nanobeir_cfg is not None:
            self.nanobeir_enabled = bool(nanobeir_cfg.enabled)
            run_every_val_value: int = int(nanobeir_cfg.run_every_n_val)
            self.nanobeir_run_every_n_val = run_every_val_value
            batch_size_value: int = int(nanobeir_cfg.batch_size)
            self.nanobeir_batch_size = batch_size_value
            self.nanobeir_save_json = bool(nanobeir_cfg.save_json)
            use_cpu_value: bool = bool(
                getattr(nanobeir_cfg, "use_cpu", self.nanobeir_use_cpu)
            )
            self.nanobeir_use_cpu = use_cpu_value
            device_id_value: int | None = getattr(
                nanobeir_cfg, "device_id", self.nanobeir_device_id
            )
            self.nanobeir_device_id = (
                None if device_id_value is None else int(device_id_value)
            )
            dataset_name: str
            dataset_names: list[str] = []
            for dataset_name in nanobeir_cfg.datasets:
                dataset_names.append(str(dataset_name))
            self.nanobeir_dataset_names = dataset_names

        self._nanobeir_evaluator_batch_size = int(self.nanobeir_batch_size)

        if self.nanobeir_enabled and not self.nanobeir_dataset_names:
            log_if_rank_zero(
                logger,
                "NanoBEIR evaluation enabled but no datasets provided; disabling.",
                level="warning",
            )
            self.nanobeir_enabled = False

    def _compute_rep_magnitude(
        self, reps: torch.Tensor, row_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Track L2 norm to capture sparse output scale.
        reps_fp32: torch.Tensor = reps.float()
        per_row_norm: torch.Tensor = torch.linalg.vector_norm(
            reps_fp32, ord=2, dim=-1
        )
        if row_mask is None:
            return per_row_norm.mean()
        mask: torch.Tensor = row_mask.to(dtype=torch.bool)
        mask_float: torch.Tensor = mask.to(dtype=per_row_norm.dtype)
        denom: torch.Tensor = mask_float.sum().clamp(min=1.0)
        masked_sum: torch.Tensor = (per_row_norm * mask_float).sum()
        return masked_sum / denom

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
        # Compute magnitudes for logging purposes only.
        q_rep_magnitude: torch.Tensor = self._compute_rep_magnitude(q_reps)
        flat_doc_reps_for_mag: torch.Tensor = doc_reps.view(-1, doc_reps.shape[-1])
        flat_doc_mask_for_mag: torch.Tensor = doc_mask.view(-1)
        doc_rep_magnitude: torch.Tensor = self._compute_rep_magnitude(
            flat_doc_reps_for_mag, flat_doc_mask_for_mag
        )
        lambda_scale_value: float = self._lambda_schedule_multiplier()
        lambda_scale: torch.Tensor = torch.tensor(
            lambda_scale_value, device=q_reps.device, dtype=q_reps.dtype
        )
        # Effective per-step regularization weights with scheduling applied.
        reg_query_lambda: torch.Tensor = lambda_scale * float(self.reg_query_weight)
        reg_doc_lambda: torch.Tensor = lambda_scale * float(self.reg_doc_weight)
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

        metrics: dict[str, torch.Tensor] = {
            "loss": loss,
            "reg_query_lambda": reg_query_lambda,
            "reg_doc_lambda": reg_doc_lambda,
            "q_rep_magnitude": q_rep_magnitude,
            "doc_rep_magnitude": doc_rep_magnitude,
        }
        if self.loss_type == "pairwise":
            metrics["pairwise_loss"] = pairwise_loss
        if self.loss_type == "in_batch":
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
        if "reg_query_lambda" in detached_metrics:
            self.log(
                "train_reg_query_lambda",
                detached_metrics["reg_query_lambda"],
                on_step=True,
                on_epoch=True,
            )
        if "reg_doc_lambda" in detached_metrics:
            self.log(
                "train_reg_doc_lambda",
                detached_metrics["reg_doc_lambda"],
                on_step=True,
                on_epoch=True,
            )
        if "q_rep_magnitude" in detached_metrics:
            self.log(
                "train_q_rep_magnitude",
                detached_metrics["q_rep_magnitude"],
                on_step=True,
                on_epoch=True,
            )
        if "doc_rep_magnitude" in detached_metrics:
            self.log(
                "train_doc_rep_magnitude",
                detached_metrics["doc_rep_magnitude"],
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
        schedule_steps: int | None = self.reg_cfg.schedule_steps
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

    def _resolve_nanobeir_device(self) -> torch.device:
        use_cpu: bool = bool(self.nanobeir_use_cpu)
        if use_cpu:
            return torch.device("cpu")
        device_id: int | None = self.nanobeir_device_id
        if torch.cuda.is_available():
            if device_id is None:
                return torch.device("cuda")
            return torch.device(f"cuda:{int(device_id)}")
        return torch.device("cpu")

    def _should_run_nanobeir_eval(self) -> bool:
        if not self.nanobeir_enabled:
            return False
        if self.trainer.sanity_checking:
            return False
        if not self.nanobeir_dataset_names:
            return False
        run_every_val: int = int(self.nanobeir_run_every_n_val)
        if run_every_val <= 0:
            return False
        self._nanobeir_val_counter += 1
        return self._nanobeir_val_counter % run_every_val == 0

    def _nanobeir_barrier(self) -> None:
        world_size: int = int(self.trainer.world_size)
        if world_size <= 1:
            return
        strategy: Any = self.trainer.strategy
        barrier_fn: Any = getattr(strategy, "barrier", None)
        if callable(barrier_fn):
            barrier_fn()
            return
        distributed_available: bool = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if not distributed_available:
            return
        torch.distributed.barrier()

    def _run_nanobeir_eval(self) -> None:
        device: torch.device = self._resolve_nanobeir_device()
        cache: SparseEncoderCache | None = self._nanobeir_cache
        cache_device: torch.device | None = self._nanobeir_cache_device
        if cache is None or cache_device != device:
            # Rebuild the encoder when no cache exists or device changes.
            cache = build_sparse_encoder_cache(
                cfg=self.cfg, model=self.model, device=device
            )
            self._nanobeir_cache = cache
            self._nanobeir_cache_device = device
        else:
            # Update weights in-place to avoid reloading HF modules each validation.
            update_sparse_encoder_cache(cache=cache, model=self.model, device=device)
        sparse_encoder: SparseEncoder = cache.sparse_encoder

        evaluator: SparseNanoBEIREvaluator
        evaluator_datasets: list[str] = self._nanobeir_evaluator_datasets
        evaluator_batch_size: int = self._nanobeir_evaluator_batch_size
        if (
            self._nanobeir_evaluator is None
            or evaluator_datasets != self.nanobeir_dataset_names
            or evaluator_batch_size != self.nanobeir_batch_size
        ):
            evaluator = SparseNanoBEIREvaluator(
                dataset_names=self.nanobeir_dataset_names,
                batch_size=self.nanobeir_batch_size,
            )
            self._nanobeir_evaluator = evaluator
            self._nanobeir_evaluator_datasets = list(self.nanobeir_dataset_names)
            self._nanobeir_evaluator_batch_size = int(self.nanobeir_batch_size)
        else:
            evaluator = self._nanobeir_evaluator
        with torch.no_grad():
            results: dict[str, Any] = evaluator(sparse_encoder)
        metric_name: str
        metric_value: Any
        logged_metrics: dict[str, float] = {}
        for metric_name, metric_value in results.items():
            log_if_rank_zero(logger, f"NanoBEIR {metric_name}: {metric_value}")
            try:
                metric_float: float = float(metric_value)
            except (TypeError, ValueError):
                continue
            logged_metrics[f"val_nanobeir_{metric_name}"] = metric_float
        if logged_metrics:
            self.log_dict(
                logged_metrics,
                sync_dist=False,
                prog_bar=False,
                rank_zero_only=True,
            )
        if self.nanobeir_save_json:
            output_path: str = os.path.join(
                self.cfg.log_dir, f"nanobeir_metrics_step{int(self.global_step)}.json"
            )
            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(results, json_file, indent=2)
            log_if_rank_zero(logger, f"Saved NanoBEIR metrics to {output_path}")

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
        if "q_rep_magnitude" in metrics:
            self.log(
                "val_q_rep_magnitude",
                metrics["q_rep_magnitude"].detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )
        if "doc_rep_magnitude" in metrics:
            self.log(
                "val_doc_rep_magnitude",
                metrics["doc_rep_magnitude"].detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )

    def on_validation_epoch_end(self) -> None:
        should_run_nanobeir: bool = self._should_run_nanobeir_eval()
        if self.val_metric_collection is not None:
            has_data: bool = self.val_metric_collection.gather(
                world_size=self.trainer.world_size,
                all_gather_fn=self.all_gather if self.trainer.world_size > 1 else None,
            )
            if not has_data:
                log_if_rank_zero(
                    logger,
                    "No predictions accumulated during validation.",
                    level="warning",
                )
            else:
                metrics: dict[str, torch.Tensor] = self.val_metric_collection.compute()
                filtered_metrics: dict[str, torch.Tensor] = {
                    f"val_{name}": value
                    for name, value in metrics.items()
                    if name.startswith(("nDCG_", "MRR_", "Recall_"))
                }
                if filtered_metrics:
                    self.log_dict(
                        filtered_metrics,
                        sync_dist=True,
                        prog_bar=False,
                        rank_zero_only=True,
                    )
                self.val_metric_collection.reset()

        if should_run_nanobeir:
            if self.trainer.is_global_zero:
                nanobeir_error: Exception | None = None
                try:
                    self._run_nanobeir_eval()
                except Exception as exc:
                    nanobeir_error = exc
                    self._nanobeir_cache = None
                    self._nanobeir_cache_device = None
                    self._nanobeir_evaluator = None
                    self._nanobeir_evaluator_datasets = []
                    log_if_rank_zero(
                        logger,
                        f"NanoBEIR evaluation failed: {nanobeir_error}",
                        level="warning",
                    )
            self._nanobeir_barrier()

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        optimizer_name: str = str(self.cfg.training.optimizer).lower()
        optimizer_cls: type[torch.optim.Optimizer]
        if optimizer_name == "adamw":
            optimizer_cls = torch.optim.AdamW
        elif optimizer_name == "adam":
            optimizer_cls = torch.optim.Adam
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        optimizer: torch.optim.Optimizer = optimizer_cls(
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
