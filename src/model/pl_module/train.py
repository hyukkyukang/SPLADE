import inspect
import math
from typing import Any, Callable, TYPE_CHECKING, TypeVar, cast

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from src.model.losses import LossComputer
from src.model.pl_module.compile_policy import TrainingCompilePolicyManager
from src.model.pl_module.loss_service import (
    LossComputationOutputs,
    LossRegularizationService,
)
from src.model.pl_module.metrics_service import TrainingMetricsService
from src.model.pl_module.training_runtime_services import (
    BenchmarkRuntimeService,
    CompileRuntimeService,
    ValidationRuntimeService,
)
from src.model.pl_module.validation_sparse_probe import ValidationSparseProbeLogger
from src.model.pl_module.utils import validate_torch_compile_mode
from src.model.pl_module.validation_service import ValidationMetricsAccumulator
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.model_utils import build_splade_model, load_splade_checkpoint
from src.utils.script_setup import normalize_optional_str
from src.utils.windowed_encoding import encode_in_chunks

if TYPE_CHECKING:
    from src.model.pl_module.nanobeir_runner import NanoBEIREvaluationRunner

logger = get_logger("SPLADETrainingModule")
_TCallable = TypeVar("_TCallable", bound=Callable[..., Any])
_VALIDATION_DOC_ENCODE_CHUNK_SIZE = 10
_TRI_STATE_CONFIG_MODES = {"auto", "true", "false"}
NanoBEIREvaluationRunner: type[Any] | None = None


class _DisabledNanoBEIREvaluationRunner:
    """No-op NanoBEIR runner used when benchmark evaluation is disabled."""

    def __init__(self, *, use_cpu: bool, doc_only_enabled: bool) -> None:
        self.enabled: bool = False
        self.use_cpu: bool = bool(use_cpu)
        self.doc_only_enabled: bool = bool(doc_only_enabled)

    def should_run_eval(self, *, sanity_checking: bool) -> bool:
        _ = sanity_checking
        return False

    def barrier(self, strategy: Any) -> None:
        _ = strategy

    def reset_runtime_state(self) -> None:
        return

    def cleanup_after_failure(self) -> None:
        return

    def offload_cache_to_cpu(self) -> None:
        return

    def resolve_device(self, training_device: torch.device) -> torch.device:
        if self.use_cpu:
            return torch.device("cpu")
        return training_device

    def run_eval(
        self,
        *,
        eval_model: torch.nn.Module,
        training_device: torch.device,
        global_step: int,
        log_dir: str,
        log_dict_fn: Callable[[dict[str, float]], None],
        masked_lm_incompatibility_predicate: Callable[[Exception], bool],
    ) -> None:
        _ = (
            eval_model,
            training_device,
            global_step,
            log_dir,
            log_dict_fn,
            masked_lm_incompatibility_predicate,
        )
        return


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


def _normalize_tri_state_config(value: Any, *, field_name: str) -> str:
    """Parse bool/None/string config knobs into auto/true/false."""
    normalized: str
    if isinstance(value, bool):
        normalized = "true" if value else "false"
    elif value is None:
        normalized = "auto"
    else:
        normalized = str(value).strip().lower()
    if normalized not in _TRI_STATE_CONFIG_MODES:
        raise ValueError(
            f"{field_name} must be one of: auto, true, false. Got: {value!r}"
        )
    return normalized


def _resolve_regularization_type(
    explicit_type: Any | None, fallback_type: Any
) -> str:
    if explicit_type is None:
        return str(fallback_type).strip().lower()
    return str(explicit_type).strip().lower()


def _optimizer_supports_kwarg(
    optimizer_cls: type[torch.optim.Optimizer], kwarg_name: str
) -> bool:
    return kwarg_name in inspect.signature(optimizer_cls).parameters


def _optimizer_grad_parameters(
    optimizer: torch.optim.Optimizer,
) -> list[torch.nn.Parameter]:
    return [
        parameter
        for group in optimizer.param_groups
        for parameter in cast(list[torch.nn.Parameter], group["params"])
        if parameter.grad is not None
    ]


def _normalize_gradient_clip_algorithm(
    gradient_clip_algorithm: str | None,
) -> str:
    return str(
        gradient_clip_algorithm if gradient_clip_algorithm is not None else "norm"
    ).split(".")[-1].lower()


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
        self._eager_train_loss_computer: LossComputer | None = None
        self._eager_validation_loss_computer: LossComputer | None = None
        self._compiled_train_loss_computer: Any | None = None
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
                cfg.training.get("step_only_metric_log_interval", 100)
            ),
            validation_diagnostics_enabled=bool(
                cfg.training.validation_metrics.get("diagnostics_enabled", True)
            ),
            validation_diagnostics_log_interval=int(
                cfg.training.validation_metrics.get("diagnostics_log_interval", 1)
            ),
        )
        self._gradient_norm_monitor_interval: int = max(
            int(cfg.training.get("gradient_norm_monitor_interval", 0)), 0
        )
        self._gradient_norm_monitor_count: int = 0
        self._gradient_norm_nonfinite_count: int = 0
        self._gradient_norm_over_clip_count: int = 0
        self._gradient_norm_monitor_sum: float = 0.0
        self._gradient_norm_monitor_max: float = 0.0
        self._gradient_norm_monitor_max_step: int = -1
        self._ddp_comm_hook_mode: str = str(
            cfg.training.get("ddp_comm_hook", "none")
        ).strip().lower()
        if self._ddp_comm_hook_mode not in {"none", "bf16", "fp16"}:
            raise ValueError(
                "training.ddp_comm_hook must be one of: none, bf16, fp16"
            )
        self._ddp_comm_hook_registered: bool = False
        ordered_mask_slot_cfg: DictConfig | None = cfg.training.get(
            "ordered_mask_slots"
        )
        self._ordered_mask_slot_cfg: DictConfig | None = ordered_mask_slot_cfg
        self._ordered_mask_slot_enabled: bool = bool(
            ordered_mask_slot_cfg is not None
            and bool(ordered_mask_slot_cfg.get("enabled", False))
            and bool(getattr(self.model, "supports_ordered_mask_slot_loss", False))
        )
        self._ordered_mask_query_weight: float = (
            0.0
            if ordered_mask_slot_cfg is None
            else float(ordered_mask_slot_cfg.get("query_term_weight", 0.0))
        )
        self._ordered_mask_doc_weight: float = (
            0.0
            if ordered_mask_slot_cfg is None
            else float(ordered_mask_slot_cfg.get("doc_term_weight", 0.0))
        )
        self._ordered_mask_ignore_index: int = (
            -100
            if ordered_mask_slot_cfg is None
            else int(ordered_mask_slot_cfg.get("ignore_index", -100))
        )
        mdlm_cfg: DictConfig = cfg.training.get("mdlm")
        self._mdlm_cfg: DictConfig | None = mdlm_cfg
        self._mdlm_enabled: bool = bool(
            mdlm_cfg is not None and bool(mdlm_cfg.get("enabled", False))
        )
        self._mdlm_weight: float = (
            0.0 if mdlm_cfg is None else float(mdlm_cfg.get("weight", 0.0))
        )
        self._mdlm_eps: float = (
            1e-3 if mdlm_cfg is None else float(mdlm_cfg.get("eps", 1e-3))
        )
        self._mdlm_force_mask_at_least_one: bool = bool(
            True
            if mdlm_cfg is None
            else mdlm_cfg.get("force_mask_at_least_one", True)
        )
        self._mdlm_doc_chunk_size: int = max(
            0,
            int(0 if mdlm_cfg is None else (mdlm_cfg.get("doc_chunk_size", 0) or 0)),
        )
        raw_mdlm_doc_selection: str = str(
            "all" if mdlm_cfg is None else mdlm_cfg.get("doc_selection", "all")
        ).strip().lower()
        if raw_mdlm_doc_selection not in {"all", "positives"}:
            raise ValueError(
                "training.mdlm.doc_selection must be one of: all, positives"
            )
        self._mdlm_doc_selection: str = raw_mdlm_doc_selection
        if self._mdlm_enabled and not bool(
            getattr(self.model, "supports_mdlm_aux_loss", False)
        ):
            raise ValueError(
                "training.mdlm.enabled=true requires a model class that implements "
                "MDLM auxiliary loss support."
            )
        # Keep compatibility with existing call sites while service extraction is ongoing.
        self.temperature = self._loss_service.temperature
        self.distill_cfg = self._loss_service.distill_cfg
        self.reg_cfg = self._loss_service.reg_cfg
        self.loss_cfg = self._loss_service.loss_cfg
        self.loss_type = self._loss_service.loss_type
        self.validation_loss_type: str = self._loss_service.resolve_validation_loss_type()
        self.reg_query_weight = self._loss_service.reg_query_weight
        self.reg_doc_weight = self._loss_service.reg_doc_weight
        self.reg_query_type = self._loss_service.reg_query_type
        self.reg_doc_type = self._loss_service.reg_doc_type

        self._eager_train_loss_computer = self._loss_service.build_loss_computer()
        if self.validation_loss_type != self.loss_type:
            self._eager_validation_loss_computer = self._loss_service.build_loss_computer(
                loss_type=self.validation_loss_type
            )
        self.loss_computer: Any = self._eager_train_loss_computer
        self._train_compile_policy.finalize_train_core_compile(
            loss_computer=self._eager_train_loss_computer,
            mdlm_enabled=bool(self._mdlm_enabled and self._mdlm_weight > 0.0),
            mdlm_doc_selection=self._mdlm_doc_selection,
            mdlm_doc_chunk_size=self._mdlm_doc_chunk_size,
            mdlm_eps=self._mdlm_eps,
            mdlm_force_mask_at_least_one=self._mdlm_force_mask_at_least_one,
            mdlm_single_positive_assumption=bool(
                int(getattr(cfg.train_dataset, "num_positives", 0) or 0) == 1
            ),
            ordered_mask_slot_enabled=bool(self._ordered_mask_slot_enabled),
            ordered_mask_query_weight=self._ordered_mask_query_weight,
            ordered_mask_doc_weight=self._ordered_mask_doc_weight,
            ordered_mask_ignore_index=self._ordered_mask_ignore_index,
        )
        if self._train_compile_policy.torch_compile_enabled and compile_loss:
            if not self._train_compile_policy.compiled_train_core_available():
                self._compiled_train_loss_computer = torch.compile(
                    self._eager_train_loss_computer,
                    **self._train_compile_policy.loss_compile_mode_kwargs,
                )
        validation_policy: TrainingCompilePolicyManager | None = (
            self._validation_compile_policy
        )
        if (
            validation_policy is not None
            and validation_policy.torch_compile_enabled
            and compile_loss
        ):
            validation_eager_loss: LossComputer = (
                self._eager_train_loss_computer
                if self._eager_validation_loss_computer is None
                else self._eager_validation_loss_computer
            )
            if (
                validation_policy is self._train_compile_policy
                and validation_eager_loss is self._eager_train_loss_computer
            ):
                self._compiled_validation_loss_computer = None
            else:
                self._compiled_validation_loss_computer = torch.compile(
                    validation_eager_loss,
                    **validation_policy.loss_compile_mode_kwargs,
                )
        self._compile_runtime = CompileRuntimeService(
            module=self,
            train_policy=self._train_compile_policy,
            validation_policy=self._validation_compile_policy,
        )
        self._activate_compile_policy(
            policy=self._train_compile_policy,
            use_compiled=self._train_compile_policy.compile_enabled_for_current_stage,
            stage="train",
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

    def _maybe_register_ddp_comm_hook(self) -> None:
        if self._ddp_comm_hook_registered or self._ddp_comm_hook_mode == "none":
            return
        trainer: Any | None = getattr(self, "trainer", None)
        if trainer is None:
            return
        strategy: Any | None = getattr(trainer, "strategy", None)
        if strategy is None:
            return
        wrapped_model: Any | None = getattr(strategy, "model", None)
        if not isinstance(wrapped_model, DistributedDataParallel):
            return
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        hook_name: str = self._ddp_comm_hook_mode
        if hook_name == "bf16":
            if bool(self.cfg.training.use_cpu) or not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "training.ddp_comm_hook=bf16 requires CUDA bf16 support."
                )
            hook = default_hooks.bf16_compress_hook
        else:
            hook = default_hooks.fp16_compress_hook
        wrapped_model.register_comm_hook(state=wrapped_model.process_group, hook=hook)
        self._ddp_comm_hook_registered = True
        log_if_rank_zero(
            logger,
            f"Registered DDP communication hook: {hook_name}.",
            level="warning",
        )

    def on_fit_start(self) -> None:
        self._maybe_register_ddp_comm_hook()

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
        # Keep the eager model as the canonical module object. Full-model compile
        # is only entered via resolve_active_model_for_train_step().
        self.model = cast(SpladeModel, self._compile_policy.eager_model)

    def _resolve_stage_loss_computer(self, *, stage: str, use_compiled: bool) -> Any:
        return self._compile_runtime.resolve_stage_loss_computer(
            stage=stage,
            use_compiled=use_compiled,
        )

    def _activate_compile_policy(
        self,
        *,
        policy: TrainingCompilePolicyManager,
        use_compiled: bool,
        stage: str,
    ) -> None:
        self._compile_runtime.activate(
            policy=policy,
            use_compiled=use_compiled,
            stage=stage,
        )

    def _encode_docs_in_chunks(
        self,
        flat_docs: torch.Tensor,
        flat_masks: torch.Tensor,
        flat_pooling_masks: torch.Tensor | None,
        *,
        chunk_size: int,
        use_compile: bool,
    ) -> torch.Tensor:
        return encode_in_chunks(
            flat_docs,
            flat_masks,
            flat_pooling_masks,
            encode_fn=lambda chunk_docs, chunk_masks, chunk_pooling_masks: (
                self.model.encode_docs(
                    chunk_docs,
                    chunk_masks,
                    pooling_mask=chunk_pooling_masks,
                )
            ),
            chunk_size=chunk_size,
            mark_step=self._compile_policy.maybe_mark_step if use_compile else None,
        )

    def _setup_eval_metrics(self, cfg: DictConfig) -> None:
        self.val_metrics_cfg = cfg.training.validation_metrics
        self._validation_metrics = ValidationMetricsAccumulator(
            dataset_name="",
            metrics_cfg=self.val_metrics_cfg,
        )
        self.val_metrics_enabled = self._validation_metrics.enabled
        nanobeir_cfg: DictConfig | None = (
            cfg.nanobeir if "nanobeir" in cfg else None
        )
        nanobeir_enabled: bool = bool(
            nanobeir_cfg is not None and nanobeir_cfg.get("enabled", False)
        )
        if not nanobeir_enabled:
            self._nanobeir_runner = _DisabledNanoBEIREvaluationRunner(
                use_cpu=bool(
                    False if nanobeir_cfg is None else nanobeir_cfg.get("use_cpu", False)
                ),
                doc_only_enabled=bool(self._doc_only_flag),
            )
        else:
            global NanoBEIREvaluationRunner
            runner_cls: type[Any] | None = NanoBEIREvaluationRunner
            if runner_cls is None:
                from src.model.pl_module.nanobeir_runner import (
                    NanoBEIREvaluationRunner as _NanoBEIREvaluationRunner,
                )

                runner_cls = _NanoBEIREvaluationRunner
                NanoBEIREvaluationRunner = _NanoBEIREvaluationRunner
            self._nanobeir_runner = runner_cls(
                cfg=cfg,
                logger=logger,
                doc_only_enabled=bool(self._doc_only_flag),
            )
        self._validation_runtime = ValidationRuntimeService(
            module=self,
            metrics_accumulator=self._validation_metrics,
            logger=logger,
        )
        self._validation_sparse_probe_runtime = ValidationSparseProbeLogger(
            module=self,
            cfg=cfg,
            logger=logger,
        )
        self._benchmark_runtime = BenchmarkRuntimeService(
            module=self,
            runner=self._nanobeir_runner,
            logger=logger,
            masked_lm_incompatibility_predicate=_is_masked_lm_incompatibility_error,
            cuda_oom_predicate=_is_cuda_oom_error,
        )

    def _compute_ordered_mask_slot_loss(
        self,
        *,
        slot_logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        if slot_logits.ndim != 3 or target_ids.ndim != 2:
            raise ValueError(
                "Ordered mask-slot loss expects logits [B, K, V] and targets [B, K]."
            )
        batch_size: int = int(slot_logits.shape[0])
        num_slots: int = int(slot_logits.shape[1])
        vocab_size: int = int(slot_logits.shape[2])
        flattened_logits: torch.Tensor = slot_logits.reshape(
            batch_size * num_slots,
            vocab_size,
        )
        flattened_targets: torch.Tensor = target_ids.reshape(batch_size * num_slots)
        return F.cross_entropy(
            flattened_logits,
            flattened_targets,
            ignore_index=self._ordered_mask_ignore_index,
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

    def _compute_mdlm_auxiliary_metrics(
        self,
        *,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        flat_doc_input_ids: torch.Tensor,
        flat_doc_attention_mask: torch.Tensor,
        pos_mask: torch.Tensor | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mdlm_enabled: bool = bool(getattr(self, "_mdlm_enabled", False))
        mdlm_weight: float = float(getattr(self, "_mdlm_weight", 0.0))
        mdlm_eps: float = float(getattr(self, "_mdlm_eps", 1e-3))
        mdlm_force_mask_at_least_one: bool = bool(
            getattr(self, "_mdlm_force_mask_at_least_one", True)
        )
        mdlm_doc_chunk_size: int = max(
            0,
            int(getattr(self, "_mdlm_doc_chunk_size", 0) or 0),
        )
        mdlm_doc_selection: str = str(
            getattr(self, "_mdlm_doc_selection", "all")
        ).strip().lower()
        if not mdlm_enabled or mdlm_weight <= 0.0:
            zero: torch.Tensor = query_input_ids.new_zeros((), dtype=torch.float32)
            return zero, zero, zero
        mdlm_model: Any = self.model
        compile_policy: Any | None = getattr(self, "_compile_policy", None)
        has_compiled_query_mdlm_aux: bool = bool(
            compile_policy is not None
            and callable(getattr(compile_policy, "has_compiled_query_mdlm_aux", None))
            and compile_policy.has_compiled_query_mdlm_aux()
        )
        has_compiled_doc_mdlm_aux: bool = bool(
            compile_policy is not None
            and callable(getattr(compile_policy, "has_compiled_doc_mdlm_aux", None))
            and compile_policy.has_compiled_doc_mdlm_aux()
        )
        selected_doc_input_ids: torch.Tensor = flat_doc_input_ids
        selected_doc_attention_mask: torch.Tensor = flat_doc_attention_mask
        if mdlm_doc_selection == "positives":
            if pos_mask is None:
                raise ValueError(
                    "training.mdlm.doc_selection=positives requires pos_mask."
                )
            selected_doc_mask: torch.Tensor = pos_mask.to(
                device=flat_doc_input_ids.device,
                dtype=torch.bool,
            ).reshape(-1)
            if doc_mask is not None:
                selected_doc_mask = selected_doc_mask & doc_mask.to(
                    device=flat_doc_input_ids.device,
                    dtype=torch.bool,
                ).reshape(-1)
            selected_doc_input_ids = flat_doc_input_ids[selected_doc_mask]
            selected_doc_attention_mask = flat_doc_attention_mask[selected_doc_mask]

        selected_doc_count: int = int(selected_doc_input_ids.shape[0])
        query_batch_size: int = int(query_input_ids.shape[0])
        can_fuse_grouped_mdlm: bool = (
            callable(getattr(mdlm_model, "compute_grouped_mdlm_aux_losses", None))
            and mdlm_doc_chunk_size <= 0
            and mdlm_doc_selection == "positives"
            and selected_doc_count > 0
            and selected_doc_count <= query_batch_size
            and int(query_input_ids.shape[1]) == int(selected_doc_input_ids.shape[1])
        )
        mdlm_q_loss: torch.Tensor
        if can_fuse_grouped_mdlm:
            with torch.autograd.profiler.record_function("splade.compute_mdlm_aux_fused"):
                mdlm_q_loss, mdlm_d_loss = mdlm_model.compute_grouped_mdlm_aux_losses(
                    input_id_groups=(query_input_ids, selected_doc_input_ids),
                    attention_mask_groups=(
                        query_attention_mask,
                        selected_doc_attention_mask,
                    ),
                    mask_probability_eps=mdlm_eps,
                    force_mask_at_least_one=mdlm_force_mask_at_least_one,
                )
        else:
            query_record_name: str = (
                "splade.compute_mdlm_aux_query_compiled"
                if has_compiled_query_mdlm_aux
                else "splade.compute_mdlm_aux_query"
            )
            with torch.autograd.profiler.record_function(query_record_name):
                if has_compiled_query_mdlm_aux:
                    mdlm_q_loss = compile_policy.run_compiled_query_mdlm_aux(
                        input_ids=query_input_ids,
                        attention_mask=query_attention_mask,
                    )
                else:
                    mdlm_q_loss = mdlm_model.compute_mdlm_aux_loss(
                        query_input_ids,
                        query_attention_mask,
                        mask_probability_eps=mdlm_eps,
                        force_mask_at_least_one=mdlm_force_mask_at_least_one,
                    )
            if selected_doc_count == 0:
                mdlm_d_loss = query_input_ids.new_zeros((), dtype=torch.float32)
            elif mdlm_doc_chunk_size > 0 and selected_doc_count > mdlm_doc_chunk_size:
                total_docs: int = selected_doc_count
                mdlm_d_loss = query_input_ids.new_zeros((), dtype=torch.float32)
                with torch.autograd.profiler.record_function(
                    "splade.compute_mdlm_aux_docs_chunked"
                ):
                    chunk_start: int
                    for chunk_start in range(0, total_docs, mdlm_doc_chunk_size):
                        chunk_end: int = min(chunk_start + mdlm_doc_chunk_size, total_docs)
                        if has_compiled_doc_mdlm_aux:
                            chunk_loss = compile_policy.run_compiled_doc_mdlm_aux(
                                input_ids=selected_doc_input_ids[chunk_start:chunk_end],
                                attention_mask=selected_doc_attention_mask[
                                    chunk_start:chunk_end
                                ],
                            )
                        else:
                            chunk_loss = mdlm_model.compute_mdlm_aux_loss(
                                selected_doc_input_ids[chunk_start:chunk_end],
                                selected_doc_attention_mask[chunk_start:chunk_end],
                                mask_probability_eps=mdlm_eps,
                                force_mask_at_least_one=mdlm_force_mask_at_least_one,
                            )
                        chunk_weight: float = float(chunk_end - chunk_start) / float(
                            total_docs
                        )
                        mdlm_d_loss = mdlm_d_loss + (chunk_loss * chunk_weight)
            else:
                doc_record_name: str = (
                    "splade.compute_mdlm_aux_docs_compiled"
                    if has_compiled_doc_mdlm_aux
                    else "splade.compute_mdlm_aux_docs"
                )
                with torch.autograd.profiler.record_function(doc_record_name):
                    if has_compiled_doc_mdlm_aux:
                        mdlm_d_loss = compile_policy.run_compiled_doc_mdlm_aux(
                            input_ids=selected_doc_input_ids,
                            attention_mask=selected_doc_attention_mask,
                        )
                    else:
                        mdlm_d_loss = mdlm_model.compute_mdlm_aux_loss(
                            selected_doc_input_ids,
                            selected_doc_attention_mask,
                            mask_probability_eps=mdlm_eps,
                            force_mask_at_least_one=mdlm_force_mask_at_least_one,
                        )
        mdlm_total: torch.Tensor = mdlm_q_loss + mdlm_d_loss
        return mdlm_q_loss, mdlm_d_loss, mdlm_total

    def _training_step_shared(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
        *,
        return_reps: bool = False,
        compute_validation_diagnostics: bool | None = None,
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
        query_pooling_mask: torch.Tensor | None = batch.get("query_pooling_mask")
        doc_pooling_mask: torch.Tensor | None = batch.get("doc_pooling_mask")
        flat_doc_pooling_masks: torch.Tensor | None = (
            None
            if doc_pooling_mask is None
            else doc_pooling_mask.view(bsz * doc_count, seq_len)
        )
        pos_mask: torch.Tensor = batch["pos_mask"]
        doc_mask: torch.Tensor = batch["doc_mask"]
        teacher_scores: torch.Tensor = batch["teacher_scores"]

        q_reps: torch.Tensor
        flat_doc_reps: torch.Tensor
        loss_outputs: LossComputationOutputs
        use_compile: bool = bool(
            self._compile_policy.compile_enabled_for_current_stage
        )
        use_compiled_train_core: bool = (
            stage == "train" and self._compile_policy.has_compiled_train_core()
        )
        use_chunked_validation_doc_encoding: bool = (
            stage == "val"
            and int(flat_docs.shape[0]) > self._validation_doc_encode_chunk_size
        )
        training_cfg = getattr(getattr(self, "cfg", None), "training", None)
        fuse_query_doc_encoding_when_possible = bool(
            training_cfg.get("fuse_query_doc_encoding_when_possible", False)
        ) if training_cfg is not None else False
        use_fused_query_doc_encoding: bool = (
            fuse_query_doc_encoding_when_possible
            and not use_chunked_validation_doc_encoding
            and not self._compile_policy.torch_compile_full_model
            and int(batch["query_input_ids"].shape[1]) == int(flat_docs.shape[1])
            and self._compile_policy.can_fuse_query_doc_encoding()
        )
        ordered_mask_slot_enabled: bool = bool(
            getattr(self, "_ordered_mask_slot_enabled", False)
            and batch.get("query_slot_target_ids") is not None
            and batch.get("doc_slot_target_ids") is not None
        )
        if ordered_mask_slot_enabled:
            use_fused_query_doc_encoding = False
        compiled_mdlm_q_loss: torch.Tensor | None = None
        compiled_mdlm_d_loss: torch.Tensor | None = None
        compiled_mdlm_total_loss: torch.Tensor | None = None
        compiled_mdlm_applied: bool = False
        compiled_mdlm_apply_mode: str = "never"
        compiled_ordered_query_slot_loss: torch.Tensor | None = None
        compiled_ordered_doc_slot_loss: torch.Tensor | None = None
        compiled_ordered_total_loss: torch.Tensor | None = None
        query_slot_logits: torch.Tensor | None = None
        flat_doc_slot_logits: torch.Tensor | None = None
        if use_chunked_validation_doc_encoding:
            with torch.autograd.profiler.record_function("splade.encode_queries"):
                if use_compile:
                    self._compile_policy.maybe_mark_step()
                q_reps = self.model.encode_queries(
                    batch["query_input_ids"],
                    batch["query_attention_mask"],
                    pooling_mask=query_pooling_mask,
                )
            with torch.autograd.profiler.record_function(
                "splade.encode_docs_chunked"
            ):
                flat_doc_reps = self._encode_docs_in_chunks(
                    flat_docs,
                    flat_masks,
                    flat_doc_pooling_masks,
                    chunk_size=self._validation_doc_encode_chunk_size,
                    use_compile=use_compile,
                )
        elif self._compile_policy.torch_compile_full_model:
            active_model: torch.nn.Module = (
                self._compile_policy.resolve_active_model_for_train_step()
            )
            with torch.autograd.profiler.record_function(
                "splade.encode_full_model"
            ):
                if use_compile:
                    self._compile_policy.maybe_mark_step()
                q_reps, flat_doc_reps = active_model(
                    batch["query_input_ids"],
                    batch["query_attention_mask"],
                    flat_docs,
                    flat_masks,
                    query_pooling_mask=query_pooling_mask,
                    doc_pooling_mask=flat_doc_pooling_masks,
                )
        elif use_compiled_train_core:
            compiled_mdlm_apply_mode = (
                self._compile_policy.compiled_train_core_mdlm_apply_mode(
                    query_seq_len=int(batch["query_input_ids"].shape[1]),
                    doc_seq_len=int(flat_docs.shape[1]),
                )
            )
            lambda_scale_value: float = self._loss_service.lambda_schedule_multiplier(
                int(self.global_step)
            )
            lambda_scale: torch.Tensor = teacher_scores.new_tensor(
                lambda_scale_value, dtype=torch.float32
            )
            with torch.autograd.profiler.record_function("splade.train_core_compiled"):
                if use_compile:
                    self._compile_policy.maybe_mark_step()
                compiled_outputs: tuple[torch.Tensor, ...] = (
                    self._compile_policy.run_compiled_train_core(
                        query_input_ids=batch["query_input_ids"],
                        query_attention_mask=batch["query_attention_mask"],
                        doc_input_ids=flat_docs,
                        doc_attention_mask=flat_masks,
                        pos_mask=pos_mask,
                        doc_mask=doc_mask,
                        teacher_scores=teacher_scores,
                        lambda_scale=lambda_scale,
                        query_pooling_mask=query_pooling_mask,
                        doc_pooling_mask=flat_doc_pooling_masks,
                        query_slot_target_ids=batch.get("query_slot_target_ids"),
                        doc_slot_target_ids=batch.get("doc_slot_target_ids"),
                    )
                )
            if len(compiled_outputs) == 24:
                (
                    q_reps,
                    flat_doc_reps,
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
                    compiled_mdlm_q_loss,
                    compiled_mdlm_d_loss,
                    compiled_mdlm_total_loss,
                    compiled_mdlm_applied_flag,
                ) = compiled_outputs
            elif len(compiled_outputs) == 27:
                (
                    q_reps,
                    flat_doc_reps,
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
                    compiled_ordered_query_slot_loss,
                    compiled_ordered_doc_slot_loss,
                    compiled_ordered_total_loss,
                    compiled_mdlm_q_loss,
                    compiled_mdlm_d_loss,
                    compiled_mdlm_total_loss,
                    compiled_mdlm_applied_flag,
                ) = compiled_outputs
            else:
                raise ValueError(
                    "Compiled train-core returned an unexpected output tuple size: "
                    f"{len(compiled_outputs)}"
                )
            if compiled_mdlm_apply_mode == "always":
                compiled_mdlm_applied = True
            elif compiled_mdlm_apply_mode == "runtime_flag":
                compiled_mdlm_applied = bool(
                    bool(compiled_mdlm_applied_flag is not None)
                    and float(compiled_mdlm_applied_flag.item()) > 0.0
                )
            lambda_scale = lambda_scale.to(dtype=q_reps.dtype, device=q_reps.device)
            reg_query_lambda: torch.Tensor = lambda_scale * float(
                getattr(self, "reg_query_weight", self.reg_cfg.query_weight)
            )
            reg_doc_lambda: torch.Tensor = lambda_scale * float(
                getattr(self, "reg_doc_weight", self.reg_cfg.doc_weight)
            )
            loss_outputs = LossComputationOutputs(
                loss=loss,
                pairwise_scores=pairwise_scores,
                pairwise_loss=pairwise_loss,
                in_batch_loss=in_batch_loss,
                distill_loss=distill_loss,
                distill_mse_loss=distill_mse_loss,
                distill_kl_loss=distill_kl_loss,
                distill_margin_mse_loss=distill_margin_mse_loss,
                q_reg=q_reg,
                d_reg=d_reg,
                lambda_scale_value=lambda_scale_value,
                lambda_scale=lambda_scale,
                reg_query_lambda=reg_query_lambda,
                reg_doc_lambda=reg_doc_lambda,
                sigmoid_pos_loss=sigmoid_pos_loss,
                sigmoid_neg_loss=sigmoid_neg_loss,
                sigmoid_logit_scale=sigmoid_logit_scale,
                sigmoid_bias=sigmoid_bias,
                sigmoid_pos_score_mean=sigmoid_pos_score_mean,
                sigmoid_neg_score_mean=sigmoid_neg_score_mean,
                sigmoid_pos_margin_mean=sigmoid_pos_margin_mean,
                sigmoid_neg_margin_mean=sigmoid_neg_margin_mean,
            )
        elif use_fused_query_doc_encoding:
            with torch.autograd.profiler.record_function(
                "splade.encode_queries_docs_fused"
            ):
                if use_compile:
                    self._compile_policy.maybe_mark_step()
                q_reps, flat_doc_reps = self._compile_policy.encode_queries_and_docs(
                    query_input_ids=batch["query_input_ids"],
                    query_attention_mask=batch["query_attention_mask"],
                    doc_input_ids=flat_docs,
                    doc_attention_mask=flat_masks,
                    query_pooling_mask=query_pooling_mask,
                    doc_pooling_mask=flat_doc_pooling_masks,
                )
        elif ordered_mask_slot_enabled:
            with torch.autograd.profiler.record_function(
                "splade.encode_queries_with_slots"
            ):
                if use_compile:
                    self._compile_policy.maybe_mark_step()
                q_reps, query_slot_logits = self.model.encode_queries_with_slot_logits(
                    batch["query_input_ids"],
                    batch["query_attention_mask"],
                    pooling_mask=query_pooling_mask,
                )
            with torch.autograd.profiler.record_function(
                "splade.encode_docs_with_slots"
            ):
                if use_compile:
                    self._compile_policy.maybe_mark_step()
                flat_doc_reps, flat_doc_slot_logits = (
                    self.model.encode_docs_with_slot_logits(
                        flat_docs,
                        flat_masks,
                        pooling_mask=flat_doc_pooling_masks,
                    )
                )
        else:
            with torch.autograd.profiler.record_function("splade.encode_queries"):
                if use_compile:
                    self._compile_policy.maybe_mark_step()
                q_reps = self.model.encode_queries(
                    batch["query_input_ids"],
                    batch["query_attention_mask"],
                    pooling_mask=query_pooling_mask,
                )
            with torch.autograd.profiler.record_function("splade.encode_docs"):
                if use_compile:
                    self._compile_policy.maybe_mark_step()
                flat_doc_reps = self.model.encode_docs(
                    flat_docs,
                    flat_masks,
                    pooling_mask=flat_doc_pooling_masks,
                )

        doc_reps: torch.Tensor = flat_doc_reps.view(bsz, doc_count, -1)

        should_compute_expensive_metrics: bool
        if stage == "train":
            should_compute_expensive_metrics = (
                self._metrics_service.should_compute_step_only_metrics(self)
            )
        elif stage == "val":
            should_compute_expensive_metrics = bool(
                True
                if compute_validation_diagnostics is None
                else compute_validation_diagnostics
            )
        else:
            should_compute_expensive_metrics = True
        if not use_compiled_train_core:
            with torch.autograd.profiler.record_function("splade.compute_loss"):
                loss_outputs = self._loss_service.compute_loss(
                    loss_computer=self.loss_computer,
                    q_reps=q_reps,
                    doc_reps=doc_reps,
                    pos_mask=pos_mask,
                    doc_mask=doc_mask,
                    teacher_scores=teacher_scores,
                    global_step=int(self.global_step),
                )
        loss: torch.Tensor
        pairwise_loss: torch.Tensor
        in_batch_loss: torch.Tensor
        distill_loss: torch.Tensor
        q_reg: torch.Tensor
        d_reg: torch.Tensor
        lambda_scale_value: float = loss_outputs.lambda_scale_value
        loss = loss_outputs.loss
        pairwise_loss = loss_outputs.pairwise_loss
        in_batch_loss = loss_outputs.in_batch_loss
        distill_loss = loss_outputs.distill_loss
        q_reg = loss_outputs.q_reg
        d_reg = loss_outputs.d_reg
        reg_query_lambda: torch.Tensor = loss_outputs.reg_query_lambda
        reg_doc_lambda: torch.Tensor = loss_outputs.reg_doc_lambda
        mdlm_enabled: bool = bool(getattr(self, "_mdlm_enabled", False))
        mdlm_weight: float = float(getattr(self, "_mdlm_weight", 0.0))
        mdlm_q_loss: torch.Tensor | None = None
        mdlm_d_loss: torch.Tensor | None = None
        mdlm_total_loss: torch.Tensor | None = None
        if stage == "train" and mdlm_enabled and mdlm_weight > 0.0:
            if (
                use_compiled_train_core
                and compiled_mdlm_applied
                and compiled_mdlm_q_loss is not None
                and compiled_mdlm_d_loss is not None
                and compiled_mdlm_total_loss is not None
            ):
                mdlm_q_loss = compiled_mdlm_q_loss
                mdlm_d_loss = compiled_mdlm_d_loss
                mdlm_total_loss = compiled_mdlm_total_loss
            else:
                mdlm_q_loss, mdlm_d_loss, mdlm_total_loss = (
                    self._compute_mdlm_auxiliary_metrics(
                        query_input_ids=batch["query_input_ids"],
                        query_attention_mask=batch["query_attention_mask"],
                        flat_doc_input_ids=flat_docs,
                        flat_doc_attention_mask=flat_masks,
                        pos_mask=pos_mask,
                        doc_mask=doc_mask,
                    )
                )
            loss = loss + (mdlm_weight * mdlm_total_loss)

        ordered_query_slot_loss: torch.Tensor | None = None
        ordered_doc_slot_loss: torch.Tensor | None = None
        ordered_mask_slot_loss: torch.Tensor | None = None
        if ordered_mask_slot_enabled:
            if (
                use_compiled_train_core
                and compiled_ordered_query_slot_loss is not None
                and compiled_ordered_doc_slot_loss is not None
                and compiled_ordered_total_loss is not None
            ):
                ordered_query_slot_loss = compiled_ordered_query_slot_loss
                ordered_doc_slot_loss = compiled_ordered_doc_slot_loss
                ordered_mask_slot_loss = compiled_ordered_total_loss
            elif query_slot_logits is not None and flat_doc_slot_logits is not None:
                query_slot_target_ids: torch.Tensor = batch["query_slot_target_ids"]
                doc_slot_target_ids: torch.Tensor = batch["doc_slot_target_ids"]
                ordered_query_slot_loss = self._compute_ordered_mask_slot_loss(
                    slot_logits=query_slot_logits,
                    target_ids=query_slot_target_ids,
                )
                ordered_doc_slot_loss = self._compute_ordered_mask_slot_loss(
                    slot_logits=flat_doc_slot_logits,
                    target_ids=doc_slot_target_ids.view(bsz * doc_count, -1),
                )
                ordered_mask_slot_loss = (
                    (
                        float(getattr(self, "_ordered_mask_query_weight", 0.0))
                        * ordered_query_slot_loss
                    )
                    + (
                        float(getattr(self, "_ordered_mask_doc_weight", 0.0))
                        * ordered_doc_slot_loss
                    )
                )
                loss = loss + ordered_mask_slot_loss

        metrics: dict[str, torch.Tensor] = {
            "loss": loss,
            "reg_query_lambda": reg_query_lambda,
            "reg_doc_lambda": reg_doc_lambda,
        }
        if should_compute_expensive_metrics:
            flat_doc_reps_for_metrics: torch.Tensor = doc_reps.view(-1, doc_reps.shape[-1])
            flat_doc_mask_for_metrics: torch.Tensor = doc_mask.view(-1)
            metrics["q_rep_magnitude"] = self._compute_rep_magnitude(q_reps)
            metrics["doc_rep_magnitude"] = self._compute_rep_magnitude(
                flat_doc_reps_for_metrics, flat_doc_mask_for_metrics
            )
        if self.loss_type in {
            "pairwise",
            "in_batch_plus_pairwise",
            "sigmoid_pairwise_hard",
        }:
            metrics["pairwise_loss"] = pairwise_loss
        if self.loss_type in {"in_batch", "in_batch_plus_pairwise"}:
            metrics["in_batch_loss"] = in_batch_loss
        if self.loss_type == "sigmoid_pairwise_hard":
            metrics["sigmoid_pos_loss"] = loss_outputs.sigmoid_pos_loss
            metrics["sigmoid_neg_loss"] = loss_outputs.sigmoid_neg_loss
            metrics["sigmoid_logit_scale"] = loss_outputs.sigmoid_logit_scale
            metrics["sigmoid_bias"] = loss_outputs.sigmoid_bias
            metrics["sigmoid_pos_score_mean"] = loss_outputs.sigmoid_pos_score_mean
            metrics["sigmoid_neg_score_mean"] = loss_outputs.sigmoid_neg_score_mean
            metrics["sigmoid_pos_margin_mean"] = loss_outputs.sigmoid_pos_margin_mean
            metrics["sigmoid_neg_margin_mean"] = loss_outputs.sigmoid_neg_margin_mean
        if self.distill_cfg.enabled:
            metrics["distill_loss"] = distill_loss
            for (
                loss_key,
                loss_value,
            ) in self._loss_service.iter_enabled_distill_metric_tensors(loss_outputs):
                metrics[f"distill_{loss_key}"] = loss_value
        if float(getattr(self, "reg_query_weight", self.reg_cfg.query_weight)) > 0:
            metrics["q_reg"] = q_reg
        if float(getattr(self, "reg_doc_weight", self.reg_cfg.doc_weight)) > 0:
            metrics["d_reg"] = d_reg
        if mdlm_total_loss is not None:
            metrics["mdlm_q_loss"] = mdlm_q_loss
            metrics["mdlm_d_loss"] = mdlm_d_loss
            metrics["mdlm_loss"] = mdlm_total_loss
            metrics["mdlm_weight"] = loss.new_tensor(mdlm_weight, dtype=torch.float32)
        if ordered_mask_slot_loss is not None:
            metrics["ordered_query_slot_loss"] = ordered_query_slot_loss
            metrics["ordered_doc_slot_loss"] = ordered_doc_slot_loss
            metrics["ordered_mask_slot_loss"] = ordered_mask_slot_loss
            metrics["ordered_query_slot_weight"] = loss.new_tensor(
                float(getattr(self, "_ordered_mask_query_weight", 0.0)),
                dtype=torch.float32,
            )
            metrics["ordered_doc_slot_weight"] = loss.new_tensor(
                float(getattr(self, "_ordered_mask_doc_weight", 0.0)),
                dtype=torch.float32,
            )

        if stage == "train" and should_compute_expensive_metrics:
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
                reg_query_type: str = _resolve_regularization_type(
                    getattr(
                        self,
                        "reg_query_type",
                        getattr(self.reg_cfg, "query_type", None),
                    ),
                    getattr(self.reg_cfg, "type", "l1"),
                )
                reg_doc_type: str = _resolve_regularization_type(
                    getattr(
                        self,
                        "reg_doc_type",
                        getattr(self.reg_cfg, "doc_type", None),
                    ),
                    getattr(self.reg_cfg, "type", "l1"),
                )
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
                    flat_doc_reps=flat_doc_reps_for_metrics,
                    flat_doc_mask=flat_doc_mask_for_metrics,
                )
                if reg_query_type == "flops":
                    metrics["q_flops_proxy_sum_equiv"] = q_reg_sum_equiv
                    metrics["q_flops_proxy_mean_equiv"] = q_reg_mean_equiv
                if reg_doc_type == "flops":
                    metrics["d_flops_proxy_sum_equiv"] = d_reg_sum_equiv
                    metrics["d_flops_proxy_mean_equiv"] = d_reg_mean_equiv
        elif stage == "val" and should_compute_expensive_metrics:
            flat_doc_reps_for_metrics = doc_reps.view(-1, doc_reps.shape[-1])
            flat_doc_mask_for_metrics = doc_mask.view(-1)
            with torch.no_grad():
                self._add_sparsity_metrics(
                    metrics=metrics,
                    q_reps=q_reps,
                    flat_doc_reps=flat_doc_reps_for_metrics,
                    flat_doc_mask=flat_doc_mask_for_metrics,
                )

        if return_reps:
            return metrics, {
                "pairwise_scores": loss_outputs.pairwise_scores,
                "pos_mask": pos_mask,
                "doc_mask": doc_mask,
            }

        return metrics

    @_dynamo_disable
    def _log_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        self._metrics_service.log_training_metrics(self, metrics)

    def _compute_total_grad_norm(self) -> torch.Tensor:
        total_sq_norm: torch.Tensor | None = None
        parameter: torch.nn.Parameter
        for parameter in self.parameters():
            grad: torch.Tensor | None = parameter.grad
            if grad is None:
                continue
            detached_grad: torch.Tensor = grad.detach()
            if detached_grad.is_sparse:
                detached_grad = detached_grad.coalesce().values()
            grad_sq_norm: torch.Tensor = detached_grad.float().pow(2).sum()
            total_sq_norm = (
                grad_sq_norm if total_sq_norm is None else total_sq_norm + grad_sq_norm
            )
        if total_sq_norm is None:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        return total_sq_norm.sqrt()

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        super().on_before_optimizer_step(optimizer)
        interval: int = self._gradient_norm_monitor_interval
        if interval <= 0:
            return
        global_step: int = int(getattr(self, "global_step", 0))
        if global_step % interval != 0:
            return

        total_grad_norm: torch.Tensor = self._compute_total_grad_norm()
        total_grad_norm_value: float = float(total_grad_norm.detach().cpu().item())
        self._gradient_norm_monitor_count += 1
        self._gradient_norm_monitor_sum += total_grad_norm_value
        if not math.isfinite(total_grad_norm_value):
            self._gradient_norm_nonfinite_count += 1
        if total_grad_norm_value > self._gradient_norm_monitor_max:
            self._gradient_norm_monitor_max = total_grad_norm_value
            self._gradient_norm_monitor_max_step = global_step

        clip_threshold: float = float(self.cfg.training.get("max_grad_norm", 0.0))
        if clip_threshold > 0.0 and total_grad_norm_value > clip_threshold:
            self._gradient_norm_over_clip_count += 1

        self.log(
            "train_preclip_grad_norm",
            total_grad_norm.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )
        if clip_threshold > 0.0:
            self.log(
                "train_preclip_grad_over_clip_ratio",
                total_grad_norm.detach() / total_grad_norm.new_tensor(clip_threshold),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
            )

    def on_train_end(self) -> None:
        super().on_train_end()
        monitor_count: int = self._gradient_norm_monitor_count
        if monitor_count <= 0:
            return
        mean_grad_norm: float = self._gradient_norm_monitor_sum / float(monitor_count)
        clip_threshold: float = float(self.cfg.training.get("max_grad_norm", 0.0))
        summary_message: str = (
            "Observed pre-clip grad norms on rank 0: "
            f"samples={monitor_count}, mean={mean_grad_norm:.4f}, "
            f"max={self._gradient_norm_monitor_max:.4f}, "
            f"max_step={self._gradient_norm_monitor_max_step}, "
            f"nonfinite={self._gradient_norm_nonfinite_count}"
        )
        if clip_threshold > 0.0:
            over_clip_ratio: float = self._gradient_norm_over_clip_count / float(
                monitor_count
            )
            summary_message += (
                f", threshold={clip_threshold:.4f}, "
                f"over_threshold={self._gradient_norm_over_clip_count}/{monitor_count} "
                f"({over_clip_ratio:.1%})"
            )
        log_if_rank_zero(logger, summary_message)

    def on_validation_start(self) -> None:
        self._compile_runtime.on_validation_start()
        self._validation_runtime.on_validation_start()

    def on_validation_end(self) -> None:
        self._compile_runtime.on_validation_end()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        q: torch.Tensor = self.model.encode_queries(
            batch["query_input_ids"],
            batch["query_attention_mask"],
            pooling_mask=batch.get("query_pooling_mask"),
        )
        d: torch.Tensor = self.model.encode_docs(
            batch["doc_input_ids"],
            batch["doc_attention_mask"],
            pooling_mask=batch.get("doc_pooling_mask"),
        )
        return {"q": q, "d": d}

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        self._compile_runtime.ensure_train_policy_active()
        metrics: dict[str, torch.Tensor] = self._training_step_shared(
            batch, stage="train"
        )
        self._log_metrics(metrics)
        return metrics["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        compute_validation_diagnostics: bool = (
            self._metrics_service.should_compute_validation_diagnostics(
                batch_idx=batch_idx
            )
        )
        if not self._validation_metrics.has_collection:
            metrics: dict[str, torch.Tensor] = self._training_step_shared(
                batch,
                stage="val",
                compute_validation_diagnostics=compute_validation_diagnostics,
            )
        else:
            metrics, rep_cache = self._training_step_shared(
                batch,
                stage="val",
                return_reps=True,
                compute_validation_diagnostics=compute_validation_diagnostics,
            )
            self._validation_runtime.append_batch(
                rep_cache["pairwise_scores"],
                rep_cache["pos_mask"],
                rep_cache["doc_mask"],
            )
        batch_size: int = int(batch["query_input_ids"].shape[0])
        self._validation_runtime.log_step_metrics(
            metrics=metrics,
            batch_size=batch_size,
        )

    def on_validation_epoch_end(self) -> None:
        self._validation_runtime.finalize_epoch()
        self._validation_sparse_probe_runtime.run_validation_epoch_end()
        self._benchmark_runtime.run_validation_epoch_end()

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any | None = None,
    ) -> None:
        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure=optimizer_closure,
        )
        if self._eager_train_loss_computer is not None:
            self._eager_train_loss_computer.clamp_parameters()

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        clip_value: float = (
            0.0 if gradient_clip_val is None else float(gradient_clip_val)
        )
        if clip_value <= 0.0:
            return
        optimizer_defaults: dict[str, Any] = cast(
            dict[str, Any], getattr(optimizer, "defaults", {})
        )
        if not bool(optimizer_defaults.get("fused", False)):
            super().configure_gradient_clipping(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )
            return

        precision_plugin: Any | None = getattr(self.trainer, "precision_plugin", None)
        if getattr(precision_plugin, "scaler", None) is not None:
            raise RuntimeError(
                "Fused optimizer gradient clipping in SPLADETrainingModule only "
                "supports precision modes without GradScaler."
            )

        algorithm_name: str = _normalize_gradient_clip_algorithm(
            gradient_clip_algorithm
        )
        parameters: list[torch.nn.Parameter] = _optimizer_grad_parameters(optimizer)
        if not parameters:
            return
        if algorithm_name == "norm":
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=clip_value)
            return
        if algorithm_name == "value":
            torch.nn.utils.clip_grad_value_(parameters, clip_value=clip_value)
            return
        raise ValueError(
            "Unsupported gradient clip algorithm for fused optimizer path: "
            f"{gradient_clip_algorithm!r}"
        )

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
        optimizer_fused_value: Any = self.cfg.training.get("optimizer_fused", "auto")
        optimizer_fused_mode: str = _normalize_tri_state_config(
            optimizer_fused_value, field_name="training.optimizer_fused"
        )
        optimizer_foreach_value: Any = self.cfg.training.get("optimizer_foreach", False)
        optimizer_foreach_mode: str = _normalize_tri_state_config(
            optimizer_foreach_value, field_name="training.optimizer_foreach"
        )

        optimizer: torch.optim.Optimizer
        optimizer_cls: type[torch.optim.Optimizer]
        if optimizer_name == "adamw":
            optimizer_cls = torch.optim.AdamW
        elif optimizer_name == "adam":
            optimizer_cls = torch.optim.Adam
        else:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}. "
                "Supported optimizers are: adam, adamw."
            )
        gradient_clipping_enabled: bool = (
            float(self.cfg.training.get("max_grad_norm", 0.0)) > 0.0
        )
        precision_name: str = str(self.cfg.training.get("precision", "")).lower()
        fused_clipping_supported: bool = (
            gradient_clipping_enabled
            and "bf16" in precision_name
            and not bool(self.cfg.training.use_cpu)
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        fused_supported: bool = _optimizer_supports_kwarg(optimizer_cls, "fused")
        fused_requested: bool = optimizer_fused_mode in {"auto", "true"}
        if fused_requested and gradient_clipping_enabled and not fused_clipping_supported:
            if optimizer_fused_mode == "true":
                raise ValueError(
                    f"Fused {optimizer_cls.__name__} is incompatible with the "
                    "current Lightning AMP gradient clipping path. Set "
                    "training.max_grad_norm=0 to benchmark fused optimizer mode."
                )
            fused_requested = False
            log_if_rank_zero(
                logger,
                f"Auto-disabled fused {optimizer_cls.__name__} because "
                "training.max_grad_norm > 0 uses Lightning AMP gradient clipping.",
                level="warning",
            )
        fused_enabled: bool = (
            fused_requested
            and fused_supported
            and not bool(self.cfg.training.use_cpu)
            and torch.cuda.is_available()
        )
        foreach_supported: bool = _optimizer_supports_kwarg(
            optimizer_cls, "foreach"
        )
        foreach_requested: bool = optimizer_foreach_mode in {"auto", "true"}
        foreach_enabled: bool = (
            foreach_requested
            and foreach_supported
            and not fused_enabled
            and not bool(self.cfg.training.use_cpu)
            and torch.cuda.is_available()
        )
        if fused_enabled:
            optimizer_kwargs["fused"] = True
            log_if_rank_zero(
                logger,
                f"Enabled fused {optimizer_cls.__name__} optimizer on CUDA.",
            )
            if gradient_clipping_enabled:
                log_if_rank_zero(
                    logger,
                    f"Using SPLADETrainingModule custom gradient clipping for fused "
                    f"{optimizer_cls.__name__} in bf16 precision.",
                )
        elif optimizer_fused_mode == "true" and not fused_enabled:
            raise ValueError(
                f"Fused {optimizer_cls.__name__} requested but unsupported in the "
                "current runtime. Set training.optimizer_fused=auto or false."
            )
        if foreach_enabled:
            optimizer_kwargs["foreach"] = True
            log_if_rank_zero(
                logger,
                f"Enabled foreach {optimizer_cls.__name__} optimizer on CUDA.",
            )
        elif optimizer_foreach_mode == "true" and not foreach_enabled:
            if fused_enabled:
                log_if_rank_zero(
                    logger,
                    f"Ignoring foreach {optimizer_cls.__name__} request because "
                    "fused optimizer mode is enabled.",
                    level="warning",
                )
            else:
                raise ValueError(
                    f"Foreach {optimizer_cls.__name__} requested but unsupported in "
                    "the current runtime. Set training.optimizer_foreach=auto or "
                    "false."
                )
        optimizer = optimizer_cls(**optimizer_kwargs)

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
