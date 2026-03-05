from typing import Any, Callable, TypeVar, cast

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from src.model.losses import LossComputer
from src.model.pl_module.compile_policy import TrainingCompilePolicyManager
from src.model.pl_module.loss_service import LossRegularizationService
from src.model.pl_module.metrics_service import TrainingMetricsService
from src.model.pl_module.nanobeir_runner import NanoBEIREvaluationRunner
from src.model.pl_module.utils import validate_torch_compile_mode
from src.model.pl_module.validation_service import ValidationMetricsAccumulator
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.model_utils import build_splade_model, load_splade_checkpoint
from src.utils.script_setup import normalize_optional_str

logger = get_logger("SPLADETrainingModule")
_TCallable = TypeVar("_TCallable", bound=Callable[..., Any])
_VALIDATION_DOC_ENCODE_CHUNK_SIZE = 10


def _dynamo_disable(fn: _TCallable) -> _TCallable:
    """Keep logging helpers out of torch.compile graphs."""
    disable_fn: Any | None = None
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
        disable_fn = torch._dynamo.disable
    if callable(disable_fn):
        return cast(_TCallable, disable_fn(fn))
    return fn


def _is_cuda_oom_error(exc: Exception) -> bool:
    """Return True when an exception indicates CUDA out-of-memory."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "cuda out of memory" in str(exc).lower()


def _is_masked_lm_incompatibility_error(exc: Exception) -> bool:
    """Return True when NanoBEIR's MLMTransformer cannot load the backbone type."""
    message: str = str(exc).lower()
    if "automodelformaskedlm" not in message:
        return False
    return (
        "unrecognized configuration class" in message
        or "for this kind of automodel" in message
    )


class SPLADETrainingModule(L.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        hparams_container: dict[str, Any] = cast(
            dict[str, Any], OmegaConf.to_container(cfg, resolve=True)
        )
        # Persist the resolved config in checkpoints for reproducible eval.
        self.save_hyperparameters(hparams_container)
        self.cfg: DictConfig = cfg
        # Build the encoder with a dtype appropriate for the device.
        self.model: SpladeModel = build_splade_model(cfg, use_cpu=cfg.training.use_cpu)
        resume_checkpoint_path: str | None = normalize_optional_str(
            cfg.training.resume_checkpoint_path
        )
        init_checkpoint_path: str | None = normalize_optional_str(
            cfg.training.init_checkpoint_path
        )
        if init_checkpoint_path is not None and resume_checkpoint_path is None:
            missing, unexpected = load_splade_checkpoint(
                self.model, init_checkpoint_path, logger=logger
            )
            log_if_rank_zero(
                logger,
                "Initialized SPLADE weights from checkpoint "
                f"{init_checkpoint_path}. Missing={len(missing)}, "
                f"unexpected={len(unexpected)}.",
            )
        # Transformers from_pretrained defaults to eval; ensure training mode here.
        self.model.train()
        self._doc_only_flag: bool = bool(self.model.doc_only)
        self._eager_loss_computer: LossComputer | None = None
        self._compiled_loss_computer: Any | None = None
        self._compiled_validation_loss_computer: Any | None = None
        self._train_compile_policy = TrainingCompilePolicyManager(
            model=self.model, logger=logger
        )
        self._validation_compile_policy: TrainingCompilePolicyManager | None = None
        self._compile_policy: TrainingCompilePolicyManager = self._train_compile_policy
        # Loss compilation is optional to avoid fragile Inductor/Triton paths.
        compile_loss: bool = bool(cfg.training.torch_compile_loss)
        self._relaxed_checkpoint_loading: bool = False
        if resume_checkpoint_path is not None and bool(cfg.training.torch_compile):
            # torch.compile can introduce wrapper-only module keys (e.g., _orig_mod)
            # that are absent in eager checkpoints. Allow partial key matching on
            # resume so base model weights restore while compile wrappers rebind.
            self.strict_loading = False
            self._relaxed_checkpoint_loading = True
            log_if_rank_zero(
                logger,
                "Enabled non-strict checkpoint loading for compiled resume to "
                "tolerate wrapper-key differences.",
                level="warning",
            )
        self._train_compile_policy.setup(cfg)
        self._validation_compile_policy = self._maybe_build_validation_compile_policy(cfg)

        self._loss_service = LossRegularizationService(cfg.training)
        self._metrics_service = TrainingMetricsService(
            step_only_metric_log_interval=int(
                cfg.training.get("step_only_metric_log_interval", 1)
            ),
        )
        # Keep compatibility with existing call sites while service extraction is ongoing.
        self.temperature = self._loss_service.temperature
        self.distill_cfg = self._loss_service.distill_cfg
        self.reg_cfg = self._loss_service.reg_cfg
        self.loss_cfg = self._loss_service.loss_cfg
        self.loss_type = self._loss_service.loss_type
        self.reg_query_weight = self._loss_service.reg_query_weight
        self.reg_doc_weight = self._loss_service.reg_doc_weight

        self.loss_computer: LossComputer = self._loss_service.build_loss_computer()
        self._eager_loss_computer = self.loss_computer
        if self._train_compile_policy.torch_compile_enabled and compile_loss:
            self._compiled_loss_computer = torch.compile(
                self._eager_loss_computer,
                **self._train_compile_policy.loss_compile_mode_kwargs,
            )
        validation_policy: TrainingCompilePolicyManager | None = (
            self._validation_compile_policy
        )
        if (
            validation_policy is not None
            and validation_policy is not self._train_compile_policy
            and validation_policy.torch_compile_enabled
            and compile_loss
        ):
            self._compiled_validation_loss_computer = torch.compile(
                self._eager_loss_computer,
                **validation_policy.loss_compile_mode_kwargs,
            )
        else:
            self._compiled_validation_loss_computer = self._compiled_loss_computer
        self._activate_compile_policy(
            policy=self._train_compile_policy,
            use_compiled=self._train_compile_policy.compile_enabled_for_current_stage,
            compiled_loss_computer=self._compiled_loss_computer,
        )
        self._setup_eval_metrics(cfg)
        self._validation_doc_encode_chunk_size: int = int(
            cfg.training.get(
                "validation_doc_encode_chunk_size",
                _VALIDATION_DOC_ENCODE_CHUNK_SIZE,
            )
        )
        if self._validation_doc_encode_chunk_size <= 0:
            self._validation_doc_encode_chunk_size = _VALIDATION_DOC_ENCODE_CHUNK_SIZE

    def _resolve_validation_compile_mode(self) -> str:
        mode_value: Any = self.cfg.training.get("torch_compile_validation_mode", "default")
        compile_mode: str
        compile_mode, _ = validate_torch_compile_mode(mode_value)
        return compile_mode

    def _maybe_build_validation_compile_policy(
        self, cfg: DictConfig
    ) -> TrainingCompilePolicyManager | None:
        if not self._train_compile_policy.torch_compile_enabled:
            return None
        if bool(cfg.training.disable_compile_for_validation):
            return None
        train_mode: str = str(cfg.training.torch_compile_mode).strip().lower()
        validation_mode: str = self._resolve_validation_compile_mode()
        if validation_mode == train_mode:
            return self._train_compile_policy

        # Build a second compile policy when validation mode differs from training.
        cfg_container: dict[str, Any] = cast(
            dict[str, Any], OmegaConf.to_container(cfg, resolve=True)
        )
        validation_cfg: DictConfig = cast(DictConfig, OmegaConf.create(cfg_container))
        validation_cfg.training.torch_compile_mode = validation_mode
        validation_cfg.training.disable_compile_for_validation = False
        validation_policy = TrainingCompilePolicyManager(
            model=self._train_compile_policy.eager_model,
            logger=logger,
        )
        validation_policy.setup(validation_cfg)
        log_if_rank_zero(
            logger,
            "Using separate torch.compile mode for validation: "
            f"train={train_mode!r}, val={validation_mode!r}.",
            level="warning",
        )
        return validation_policy

    def _sync_model_with_active_compile_policy(self) -> None:
        if (
            self._compile_policy.torch_compile_full_model
            and self._compile_policy.compile_enabled_for_current_stage
            and self._compile_policy.compiled_model is not None
        ):
            self.model = cast(SpladeModel, self._compile_policy.compiled_model)
            return
        self.model = cast(SpladeModel, self._compile_policy.eager_model)

    def _activate_compile_policy(
        self,
        *,
        policy: TrainingCompilePolicyManager,
        use_compiled: bool,
        compiled_loss_computer: Any | None,
    ) -> None:
        policy.prepare_for_device(device=self.device, use_compiled=use_compiled)
        selected_loss_computer: Any | None = policy.set_compile_state(
            use_compiled=use_compiled,
            eager_loss_computer=self._eager_loss_computer,
            compiled_loss_computer=compiled_loss_computer,
        )
        self._compile_policy = policy
        if selected_loss_computer is not None:
            self.loss_computer = selected_loss_computer
        self._sync_model_with_active_compile_policy()

    def _encode_docs_in_chunks(
        self,
        flat_docs: torch.Tensor,
        flat_masks: torch.Tensor,
        *,
        chunk_size: int,
        use_compile: bool,
    ) -> torch.Tensor:
        total_docs: int = int(flat_docs.shape[0])
        if total_docs <= chunk_size:
            if use_compile:
                self._compile_policy.maybe_mark_step()
            return self.model.encode_docs(flat_docs, flat_masks)

        doc_rep_chunks: list[torch.Tensor] = []
        start: int
        for start in range(0, total_docs, chunk_size):
            end: int = min(start + chunk_size, total_docs)
            if use_compile:
                self._compile_policy.maybe_mark_step()
            doc_rep_chunks.append(
                self.model.encode_docs(flat_docs[start:end], flat_masks[start:end])
            )
        return torch.cat(doc_rep_chunks, dim=0)

    def _setup_eval_metrics(self, cfg: DictConfig) -> None:
        self.val_metrics_cfg = cfg.training.validation_metrics
        self._validation_metrics = ValidationMetricsAccumulator(
            dataset_name="",
            metrics_cfg=self.val_metrics_cfg,
        )
        self.val_metrics_enabled = self._validation_metrics.enabled
        self._nanobeir_runner = NanoBEIREvaluationRunner(
            cfg=cfg,
            logger=logger,
            doc_only_enabled=bool(self._doc_only_flag),
        )

    def _compute_rep_magnitude(
        self, reps: torch.Tensor, row_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Track L2 norm to capture sparse output scale.
        reps_fp32: torch.Tensor = reps.float()
        per_row_norm: torch.Tensor = torch.linalg.vector_norm(reps_fp32, ord=2, dim=-1)
        if row_mask is None:
            return per_row_norm.mean()
        mask: torch.Tensor = row_mask.to(dtype=torch.bool)
        mask_float: torch.Tensor = mask.to(dtype=per_row_norm.dtype)
        denom: torch.Tensor = mask_float.sum().clamp(min=1.0)
        masked_sum: torch.Tensor = (per_row_norm * mask_float).sum()
        return masked_sum / denom

    def _compute_active_dims_and_sparsity(
        self, reps: torch.Tensor, row_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reps_fp32: torch.Tensor = reps.float()
        per_row_active_dims: torch.Tensor = (reps_fp32 > 0).sum(dim=-1).float()
        if row_mask is None:
            mean_active_dims: torch.Tensor = per_row_active_dims.mean()
        else:
            mask: torch.Tensor = row_mask.to(dtype=torch.bool)
            mask_float: torch.Tensor = mask.to(dtype=per_row_active_dims.dtype)
            denom: torch.Tensor = mask_float.sum().clamp(min=1.0)
            mean_active_dims = (per_row_active_dims * mask_float).sum() / denom
        vocab_dim: float = float(reps.shape[-1])
        sparsity_ratio: torch.Tensor = 1.0 - (mean_active_dims / vocab_dim)
        return mean_active_dims, sparsity_ratio

    def _add_sparsity_metrics(
        self,
        *,
        metrics: dict[str, torch.Tensor],
        q_reps: torch.Tensor,
        flat_doc_reps: torch.Tensor,
        flat_doc_mask: torch.Tensor,
    ) -> None:
        q_active_dims: torch.Tensor
        q_sparsity_ratio: torch.Tensor
        q_active_dims, q_sparsity_ratio = self._compute_active_dims_and_sparsity(q_reps)
        doc_active_dims: torch.Tensor
        doc_sparsity_ratio: torch.Tensor
        doc_active_dims, doc_sparsity_ratio = self._compute_active_dims_and_sparsity(
            flat_doc_reps, flat_doc_mask
        )
        metrics["q_active_dims"] = q_active_dims
        metrics["doc_active_dims"] = doc_active_dims
        metrics["q_sparsity_ratio"] = q_sparsity_ratio
        metrics["doc_sparsity_ratio"] = doc_sparsity_ratio

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
        doc_input_ids: torch.Tensor = batch["doc_input_ids"]
        doc_attention_mask: torch.Tensor = batch["doc_attention_mask"]

        bsz: int
        doc_count: int
        seq_len: int
        bsz, doc_count, seq_len = doc_input_ids.shape
        flat_docs: torch.Tensor = doc_input_ids.view(bsz * doc_count, seq_len)
        flat_masks: torch.Tensor = doc_attention_mask.view(bsz * doc_count, seq_len)

        q_reps: torch.Tensor
        flat_doc_reps: torch.Tensor
        use_compile: bool = bool(
            self._compile_policy.compile_enabled_for_current_stage
        )
        use_chunked_validation_doc_encoding: bool = (
            stage == "val"
            and int(flat_docs.shape[0]) > self._validation_doc_encode_chunk_size
        )
        if use_chunked_validation_doc_encoding:
            if use_compile:
                self._compile_policy.maybe_mark_step()
            q_reps = self.model.encode_queries(
                batch["query_input_ids"], batch["query_attention_mask"]
            )
            flat_doc_reps = self._encode_docs_in_chunks(
                flat_docs,
                flat_masks,
                chunk_size=self._validation_doc_encode_chunk_size,
                use_compile=use_compile,
            )
        elif self._compile_policy.torch_compile_full_model:
            active_model: torch.nn.Module = (
                self._compile_policy.resolve_active_model_for_train_step()
            )
            if use_compile:
                self._compile_policy.maybe_mark_step()
            q_reps, flat_doc_reps = active_model(
                batch["query_input_ids"],
                batch["query_attention_mask"],
                flat_docs,
                flat_masks,
            )
        else:
            if use_compile:
                self._compile_policy.maybe_mark_step()
            q_reps = self.model.encode_queries(
                batch["query_input_ids"], batch["query_attention_mask"]
            )
            if use_compile:
                self._compile_policy.maybe_mark_step()
            flat_doc_reps = self.model.encode_docs(flat_docs, flat_masks)

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
        loss_outputs = self._loss_service.compute_loss(
            loss_computer=self.loss_computer,
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            stage=stage,
            global_step=int(self.global_step),
        )
        loss: torch.Tensor
        pairwise_loss: torch.Tensor
        in_batch_loss: torch.Tensor
        distill_loss: torch.Tensor
        distill_losses: dict[str, torch.Tensor]
        q_reg: torch.Tensor
        d_reg: torch.Tensor
        lambda_scale_value: float = loss_outputs.lambda_scale_value
        loss = loss_outputs.loss
        pairwise_loss = loss_outputs.pairwise_loss
        in_batch_loss = loss_outputs.in_batch_loss
        distill_loss = loss_outputs.distill_loss
        distill_losses = loss_outputs.distill_losses
        q_reg = loss_outputs.q_reg
        d_reg = loss_outputs.d_reg
        reg_query_lambda: torch.Tensor = loss_outputs.reg_query_lambda
        reg_doc_lambda: torch.Tensor = loss_outputs.reg_doc_lambda

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
            for loss_key, loss_value in distill_losses.items():
                metrics[f"distill_{loss_key}"] = loss_value
        if self.reg_cfg.query_weight > 0:
            metrics["q_reg"] = q_reg
        if self.reg_cfg.doc_weight > 0:
            metrics["d_reg"] = d_reg

        if stage == "train":
            with torch.no_grad():
                vocab_dim: float = float(q_reps.shape[-1])
                vocab_dim_tensor: torch.Tensor = q_reps.new_tensor(
                    vocab_dim, dtype=torch.float32
                )
                q_reg_fp32: torch.Tensor = q_reg.float()
                d_reg_fp32: torch.Tensor = d_reg.float()
                reg_query_lambda_fp32: torch.Tensor = reg_query_lambda.float()
                reg_doc_lambda_fp32: torch.Tensor = reg_doc_lambda.float()
                loss_fp32: torch.Tensor = loss.float()
                q_reg_contrib: torch.Tensor = reg_query_lambda_fp32 * q_reg_fp32
                d_reg_contrib: torch.Tensor = reg_doc_lambda_fp32 * d_reg_fp32
                reg_total_contrib: torch.Tensor = q_reg_contrib + d_reg_contrib
                pre_reg_loss: torch.Tensor = loss_fp32 - reg_total_contrib
                eps: torch.Tensor = q_reps.new_tensor(1e-12, dtype=torch.float32)
                reg_total_frac_loss: torch.Tensor = reg_total_contrib / (
                    loss_fp32.abs().clamp(min=eps)
                )
                reg_total_frac_pre_reg: torch.Tensor = reg_total_contrib / (
                    pre_reg_loss.abs().clamp(min=eps)
                )
                q_reg_sum_equiv: torch.Tensor = q_reg_fp32
                d_reg_sum_equiv: torch.Tensor = d_reg_fp32
                q_reg_mean_equiv: torch.Tensor = q_reg_fp32 / vocab_dim_tensor
                d_reg_mean_equiv: torch.Tensor = d_reg_fp32 / vocab_dim_tensor
                if not bool(self.reg_cfg.paper_faithful):
                    q_reg_sum_equiv = q_reg_fp32 * vocab_dim_tensor
                    d_reg_sum_equiv = d_reg_fp32 * vocab_dim_tensor
                    q_reg_mean_equiv = q_reg_fp32
                    d_reg_mean_equiv = d_reg_fp32
                metrics["reg_lambda_scale"] = q_reps.new_tensor(
                    lambda_scale_value, dtype=torch.float32
                )
                metrics["reg_q_contrib"] = q_reg_contrib
                metrics["reg_d_contrib"] = d_reg_contrib
                metrics["reg_total_contrib"] = reg_total_contrib
                metrics["pre_reg_loss"] = pre_reg_loss
                metrics["reg_total_frac_loss"] = reg_total_frac_loss
                metrics["reg_total_frac_pre_reg"] = reg_total_frac_pre_reg
                metrics["vocab_size"] = vocab_dim_tensor
                self._add_sparsity_metrics(
                    metrics=metrics,
                    q_reps=q_reps,
                    flat_doc_reps=flat_doc_reps_for_mag,
                    flat_doc_mask=flat_doc_mask_for_mag,
                )
                if str(self.reg_cfg.type).lower() == "flops":
                    metrics["q_flops_proxy_sum_equiv"] = q_reg_sum_equiv
                    metrics["d_flops_proxy_sum_equiv"] = d_reg_sum_equiv
                    metrics["q_flops_proxy_mean_equiv"] = q_reg_mean_equiv
                    metrics["d_flops_proxy_mean_equiv"] = d_reg_mean_equiv
        elif stage == "val":
            with torch.no_grad():
                self._add_sparsity_metrics(
                    metrics=metrics,
                    q_reps=q_reps,
                    flat_doc_reps=flat_doc_reps_for_mag,
                    flat_doc_mask=flat_doc_mask_for_mag,
                )

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
        self._metrics_service.log_training_metrics(self, metrics)

    def on_validation_start(self) -> None:
        if self._train_compile_policy.torch_compile_enabled:
            if bool(self._train_compile_policy.disable_compile_for_validation):
                self._activate_compile_policy(
                    policy=self._train_compile_policy,
                    use_compiled=False,
                    compiled_loss_computer=self._compiled_loss_computer,
                )
            else:
                validation_policy: TrainingCompilePolicyManager = (
                    self._validation_compile_policy or self._train_compile_policy
                )
                validation_compiled_loss: Any | None = (
                    self._compiled_validation_loss_computer
                    if validation_policy is not self._train_compile_policy
                    else self._compiled_loss_computer
                )
                self._activate_compile_policy(
                    policy=validation_policy,
                    use_compiled=True,
                    compiled_loss_computer=validation_compiled_loss,
                )
        else:
            self._compile_policy = self._train_compile_policy
            self._sync_model_with_active_compile_policy()
        self._validation_metrics.on_validation_start(self.device)

    def on_validation_end(self) -> None:
        if self._train_compile_policy.torch_compile_enabled:
            self._activate_compile_policy(
                policy=self._train_compile_policy,
                use_compiled=True,
                compiled_loss_computer=self._compiled_loss_computer,
            )
        else:
            self._compile_policy = self._train_compile_policy
            self._sync_model_with_active_compile_policy()

    def _append_validation_metrics(
        self,
        q_reps: torch.Tensor,
        doc_reps: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> None:
        self._validation_metrics.append_batch(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            world_size=int(self.trainer.world_size),
            global_rank=int(self.trainer.global_rank),
        )

    def _log_validation_metric(
        self,
        *,
        name: str,
        value: torch.Tensor | None,
        batch_size: int,
    ) -> None:
        if value is None:
            return
        self.log(
            name,
            value.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def _resolve_nanobeir_device(self) -> torch.device:
        return self._nanobeir_runner.resolve_device(self.device)

    def _should_run_nanobeir_eval(self) -> bool:
        return self._nanobeir_runner.should_run_eval(
            sanity_checking=bool(self.trainer.sanity_checking)
        )

    def _nanobeir_barrier(self) -> None:
        world_size: int = int(self.trainer.world_size)
        if world_size <= 1:
            return
        self._nanobeir_runner.barrier(self.trainer.strategy)

    def _reset_nanobeir_runtime_state(self) -> None:
        self._nanobeir_runner.reset_runtime_state()

    def _cleanup_after_nanobeir_failure(self) -> None:
        self._nanobeir_runner.cleanup_after_failure()

    def _offload_nanobeir_cache_to_cpu(self) -> None:
        self._nanobeir_runner.offload_cache_to_cpu()

    def _run_nanobeir_eval(self) -> None:
        eval_model: torch.nn.Module = (
            self._compile_policy.eager_model
            if self._compile_policy.torch_compile_full_model
            else self.model
        )
        self._nanobeir_runner.run_eval(
            eval_model=eval_model,
            training_device=self.device,
            global_step=int(self.global_step),
            log_dir=str(self.cfg.log_dir),
            log_dict_fn=lambda logged_metrics: self.log_dict(
                logged_metrics,
                sync_dist=True,
                prog_bar=False,
                rank_zero_only=True,
            ),
            masked_lm_incompatibility_predicate=_is_masked_lm_incompatibility_error,
        )

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
        if self._compile_policy is not self._train_compile_policy:
            self._activate_compile_policy(
                policy=self._train_compile_policy,
                use_compiled=True,
                compiled_loss_computer=self._compiled_loss_computer,
            )
        if (
            self._compile_policy.torch_compile_enabled
            and not self._compile_policy.compile_enabled_for_current_stage
        ):
            self._activate_compile_policy(
                policy=self._compile_policy,
                use_compiled=True,
                compiled_loss_computer=self._compiled_loss_computer,
            )
        metrics: dict[str, torch.Tensor] = self._training_step_shared(
            batch, stage="train"
        )
        self._log_metrics(metrics)
        return metrics["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        if not self._validation_metrics.has_collection:
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
        metric_names: tuple[tuple[str, str], ...] = (
            ("val_q_rep_magnitude", "q_rep_magnitude"),
            ("val_doc_rep_magnitude", "doc_rep_magnitude"),
            ("val_q_active_dims", "q_active_dims"),
            ("val_doc_active_dims", "doc_active_dims"),
            ("val_q_sparsity_ratio", "q_sparsity_ratio"),
            ("val_doc_sparsity_ratio", "doc_sparsity_ratio"),
        )
        log_name: str
        metric_key: str
        for log_name, metric_key in metric_names:
            self._log_validation_metric(
                name=log_name,
                value=metrics.get(metric_key),
                batch_size=batch_size,
            )

    def on_validation_epoch_end(self) -> None:
        should_run_nanobeir: bool = self._should_run_nanobeir_eval()
        if self._validation_metrics.has_collection:
            has_data: bool
            filtered_metrics: dict[str, torch.Tensor]
            has_data, filtered_metrics = self._validation_metrics.finalize_epoch(
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
                if filtered_metrics:
                    self.log_dict(
                        filtered_metrics,
                        sync_dist=True,
                        prog_bar=False,
                        rank_zero_only=True,
                    )

        if should_run_nanobeir:
            if self.trainer.is_global_zero:
                nanobeir_error: Exception | None = None
                try:
                    self._run_nanobeir_eval()
                except Exception as exc:
                    nanobeir_error = exc
                    # If NanoBEIR OOMs on GPU, retry on CPU and keep future
                    # NanoBEIR evals on CPU for this run.
                    should_retry_on_cpu: bool = (
                        _is_cuda_oom_error(exc)
                        and not self._nanobeir_runner.doc_only_enabled
                        and not bool(self._nanobeir_runner.use_cpu)
                    )
                    if should_retry_on_cpu:
                        log_if_rank_zero(
                            logger,
                            "NanoBEIR hit CUDA OOM; retrying NanoBEIR on CPU for "
                            "this and subsequent validations.",
                            level="warning",
                        )
                        self._cleanup_after_nanobeir_failure()
                        self._nanobeir_runner.use_cpu = True
                        try:
                            self._run_nanobeir_eval()
                            nanobeir_error = None
                        except Exception as cpu_exc:
                            nanobeir_error = cpu_exc
                            self._cleanup_after_nanobeir_failure()
                    else:
                        self._cleanup_after_nanobeir_failure()
                    if nanobeir_error is not None:
                        log_if_rank_zero(
                            logger,
                            f"NanoBEIR evaluation failed: {nanobeir_error}",
                            level="warning",
                        )
                finally:
                    self._offload_nanobeir_cache_to_cpu()
            self._nanobeir_barrier()

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        optimizer_name: str = str(self.cfg.training.optimizer).lower()
        raw_betas: Any = self.cfg.training.get("adam_betas", (0.9, 0.999))
        if not isinstance(raw_betas, (list, tuple)) or len(raw_betas) != 2:
            raise ValueError(
                "training.adam_betas must be a list/tuple with 2 elements, "
                f"got: {raw_betas!r}"
            )
        beta1: float = float(raw_betas[0])
        beta2: float = float(raw_betas[1])
        adam_betas: tuple[float, float] = (beta1, beta2)
        adam_eps: float = float(self.cfg.training.get("adam_eps", 1e-8))
        optimizer_kwargs: dict[str, Any] = {
            "params": self.parameters(),
            "lr": self.cfg.training.lr,
            "weight_decay": self.cfg.training.weight_decay,
            "betas": adam_betas,
            "eps": adam_eps,
        }

        optimizer: torch.optim.Optimizer
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(**optimizer_kwargs)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(**optimizer_kwargs)
        else:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}. "
                "Supported optimizers are: adam, adamw."
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
