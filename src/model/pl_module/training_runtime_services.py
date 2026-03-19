from typing import Any, Callable, cast

import torch

from src.model.losses import LossComputer
from src.model.pl_module.compile_policy import TrainingCompilePolicyManager
from src.model.pl_module.validation_service import ValidationMetricsAccumulator
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import log_if_rank_zero

_VALIDATION_METRIC_NAMES: tuple[tuple[str, str], ...] = (
    ("val_q_rep_magnitude", "q_rep_magnitude"),
    ("val_doc_rep_magnitude", "doc_rep_magnitude"),
    ("val_q_active_dims", "q_active_dims"),
    ("val_doc_active_dims", "doc_active_dims"),
    ("val_q_sparsity_ratio", "q_sparsity_ratio"),
    ("val_doc_sparsity_ratio", "doc_sparsity_ratio"),
    ("val_sigmoid_pos_loss", "sigmoid_pos_loss"),
    ("val_sigmoid_neg_loss", "sigmoid_neg_loss"),
    ("val_sigmoid_logit_scale", "sigmoid_logit_scale"),
    ("val_sigmoid_bias", "sigmoid_bias"),
    ("val_sigmoid_pos_score_mean", "sigmoid_pos_score_mean"),
    ("val_sigmoid_neg_score_mean", "sigmoid_neg_score_mean"),
    ("val_sigmoid_pos_margin_mean", "sigmoid_pos_margin_mean"),
    ("val_sigmoid_neg_margin_mean", "sigmoid_neg_margin_mean"),
)


class CompileRuntimeService:
    """Own compile-policy stage switching and stage-specific loss selection."""

    def __init__(
        self,
        *,
        module: Any,
        train_policy: TrainingCompilePolicyManager,
        validation_policy: TrainingCompilePolicyManager | None,
    ) -> None:
        self._module: Any = module
        self.train_policy: TrainingCompilePolicyManager = train_policy
        self.validation_policy: TrainingCompilePolicyManager | None = validation_policy

    def _sync_model_with_active_compile_policy(self) -> None:
        self._module.model = cast(SpladeModel, self._module._compile_policy.eager_model)

    def resolve_stage_loss_computer(self, *, stage: str, use_compiled: bool) -> Any:
        eager_loss_computer: LossComputer | None
        compiled_loss_computer: Any | None
        if stage == "train":
            eager_loss_computer = self._module._eager_train_loss_computer
            compiled_loss_computer = self._module._compiled_train_loss_computer
        elif stage == "val":
            eager_loss_computer = (
                self._module._eager_train_loss_computer
                if self._module._eager_validation_loss_computer is None
                else self._module._eager_validation_loss_computer
            )
            compiled_loss_computer = (
                self._module._compiled_train_loss_computer
                if self._module._compiled_validation_loss_computer is None
                else self._module._compiled_validation_loss_computer
            )
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        if use_compiled and compiled_loss_computer is not None:
            return compiled_loss_computer
        if eager_loss_computer is None:
            raise RuntimeError(
                f"Eager loss computer is not initialized for stage={stage!r}."
            )
        return eager_loss_computer

    def activate(
        self,
        *,
        policy: TrainingCompilePolicyManager,
        use_compiled: bool,
        stage: str,
    ) -> None:
        policy.prepare_for_device(device=self._module.device, use_compiled=use_compiled)
        policy.set_compile_state(use_compiled=use_compiled)
        self._module._compile_policy = policy
        self._module.loss_computer = self.resolve_stage_loss_computer(
            stage=stage,
            use_compiled=bool(policy.compile_enabled_for_current_stage),
        )
        self._sync_model_with_active_compile_policy()

    def on_validation_start(self) -> None:
        if self.train_policy.torch_compile_enabled:
            if bool(self.train_policy.disable_compile_for_validation):
                self.activate(
                    policy=self.train_policy,
                    use_compiled=False,
                    stage="val",
                )
                return
            validation_policy: TrainingCompilePolicyManager = (
                self.validation_policy or self.train_policy
            )
            self.activate(
                policy=validation_policy,
                use_compiled=True,
                stage="val",
            )
            return
        self.activate(
            policy=self.train_policy,
            use_compiled=False,
            stage="val",
        )

    def on_validation_end(self) -> None:
        self.activate(
            policy=self.train_policy,
            use_compiled=bool(self.train_policy.torch_compile_enabled),
            stage="train",
        )

    def ensure_train_policy_active(self) -> None:
        if self._module._compile_policy is not self.train_policy:
            self.activate(
                policy=self.train_policy,
                use_compiled=True,
                stage="train",
            )
        if (
            self._module._compile_policy.torch_compile_enabled
            and not self._module._compile_policy.compile_enabled_for_current_stage
        ):
            self.activate(
                policy=self._module._compile_policy,
                use_compiled=True,
                stage="train",
            )


class ValidationRuntimeService:
    """Own validation-step logging and epoch-level metric accumulation."""

    def __init__(
        self,
        *,
        module: Any,
        metrics_accumulator: ValidationMetricsAccumulator,
        logger: Any,
    ) -> None:
        self._module: Any = module
        self._metrics: ValidationMetricsAccumulator = metrics_accumulator
        self._logger: Any = logger

    @property
    def has_collection(self) -> bool:
        return self._metrics.has_collection

    def on_validation_start(self) -> None:
        self._metrics.on_validation_start(self._module.device)

    def append_batch(
        self,
        pairwise_scores: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> None:
        self._metrics.append_batch(
            pairwise_scores=pairwise_scores,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            world_size=int(self._module.trainer.world_size),
            global_rank=int(self._module.trainer.global_rank),
        )

    def log_metric(
        self,
        *,
        name: str,
        value: torch.Tensor | None,
        batch_size: int,
    ) -> None:
        if value is None:
            return
        self._module.log(
            name,
            value.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def log_step_metrics(
        self,
        *,
        metrics: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        self._module.log(
            "val_loss",
            metrics["loss"].detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        log_name: str
        metric_key: str
        for log_name, metric_key in _VALIDATION_METRIC_NAMES:
            self.log_metric(
                name=log_name,
                value=metrics.get(metric_key),
                batch_size=batch_size,
            )

    def finalize_epoch(self) -> None:
        if not self._metrics.has_collection:
            return
        has_data: bool
        filtered_metrics: dict[str, torch.Tensor]
        has_data, filtered_metrics = self._metrics.finalize_epoch(
            world_size=self._module.trainer.world_size,
            all_gather_fn=(
                self._module.all_gather if self._module.trainer.world_size > 1 else None
            ),
        )
        if not has_data:
            log_if_rank_zero(
                self._logger,
                "No predictions accumulated during validation.",
                level="warning",
            )
            return
        if filtered_metrics:
            self._module.log_dict(
                filtered_metrics,
                sync_dist=True,
                prog_bar=False,
                rank_zero_only=True,
            )


class BenchmarkRuntimeService:
    """Own NanoBEIR runtime orchestration and retry/offload policy."""

    def __init__(
        self,
        *,
        module: Any,
        runner: Any,
        logger: Any,
        masked_lm_incompatibility_predicate: Callable[[Exception], bool],
        cuda_oom_predicate: Callable[[Exception], bool],
    ) -> None:
        self._module: Any = module
        self._runner: Any = runner
        self._logger: Any = logger
        self._masked_lm_incompatibility_predicate = masked_lm_incompatibility_predicate
        self._cuda_oom_predicate = cuda_oom_predicate

    def resolve_device(self) -> torch.device:
        return self._runner.resolve_device(self._module.device)

    def should_run_eval(self) -> bool:
        return self._runner.should_run_eval(
            sanity_checking=bool(self._module.trainer.sanity_checking)
        )

    def barrier(self) -> None:
        if int(self._module.trainer.world_size) <= 1:
            return
        self._runner.barrier(self._module.trainer.strategy)

    def reset_runtime_state(self) -> None:
        self._runner.reset_runtime_state()

    def cleanup_after_failure(self) -> None:
        self._runner.cleanup_after_failure()

    def offload_cache_to_cpu(self) -> None:
        self._runner.offload_cache_to_cpu()

    def run_eval_once(self) -> None:
        eval_model: torch.nn.Module = (
            self._module._compile_policy.eager_model
            if self._module._compile_policy.torch_compile_full_model
            else self._module.model
        )
        self._runner.run_eval(
            eval_model=eval_model,
            training_device=self._module.device,
            global_step=int(self._module.global_step),
            log_dir=str(self._module.cfg.log_dir),
            log_dict_fn=lambda logged_metrics: self._module.log_dict(
                logged_metrics,
                sync_dist=True,
                prog_bar=False,
                rank_zero_only=True,
            ),
            masked_lm_incompatibility_predicate=(
                self._masked_lm_incompatibility_predicate
            ),
        )

    def run_validation_epoch_end(self) -> None:
        should_run_nanobeir: bool = self.should_run_eval()
        if not should_run_nanobeir:
            return
        if self._module.trainer.is_global_zero:
            nanobeir_error: Exception | None = None
            try:
                self.run_eval_once()
            except Exception as exc:
                nanobeir_error = exc
                should_retry_on_cpu: bool = (
                    self._cuda_oom_predicate(exc)
                    and not self._runner.doc_only_enabled
                    and not bool(self._runner.use_cpu)
                )
                if should_retry_on_cpu:
                    log_if_rank_zero(
                        self._logger,
                        "NanoBEIR hit CUDA OOM; retrying NanoBEIR on CPU for "
                        "this and subsequent validations.",
                        level="warning",
                    )
                    self.cleanup_after_failure()
                    self._runner.use_cpu = True
                    try:
                        self.run_eval_once()
                        nanobeir_error = None
                    except Exception as cpu_exc:
                        nanobeir_error = cpu_exc
                        self.cleanup_after_failure()
                else:
                    self.cleanup_after_failure()
                if nanobeir_error is not None:
                    log_if_rank_zero(
                        self._logger,
                        f"NanoBEIR evaluation failed: {nanobeir_error}",
                        level="warning",
                    )
            finally:
                self.offload_cache_to_cpu()
        self.barrier()
