import logging
from typing import Any, Callable, cast

import torch
from omegaconf import DictConfig

from src.model.pl_module.utils import (
    resolve_cudagraph_mark_step,
    validate_torch_compile_mode,
)
from src.utils.logging import log_if_rank_zero
from src.utils.normalize import normalize_optional_str


class _SharedCompiledEncoderAdapter(torch.nn.Module):
    """Route query/doc calls through one compiled shared encoder module."""

    def __init__(
        self,
        *,
        encoder_fn: Callable[..., torch.Tensor],
        pooling_mode: torch.Tensor,
    ) -> None:
        super().__init__()
        self._encoder_fn: Callable[..., torch.Tensor] = encoder_fn
        self.register_buffer("_pooling_mode", pooling_mode, persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._encoder_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mode=self._pooling_mode,
        )


class TrainingCompilePolicyManager:
    """Encapsulate torch.compile setup/state transitions for training."""

    def __init__(self, *, model: torch.nn.Module, logger: logging.Logger) -> None:
        self.model: torch.nn.Module = model
        self.logger: logging.Logger = logger

        self.disable_compile_for_validation: bool = False
        self.torch_compile_enabled: bool = False
        self.torch_compile_mark_step: Any | None = None
        self.torch_compile_full_model: bool = False
        self.compile_enabled_for_current_stage: bool = False
        self.eager_model: torch.nn.Module = model
        self.compiled_model: Any | None = None
        self._compiled_shared_encoder_module: Callable[..., torch.Tensor] | None = None
        self._eager_query_encoder_fn: Any | None = None
        self._eager_doc_encoder_fn: Any | None = None
        self._compiled_query_encoder_fn: Any | None = None
        self._compiled_doc_encoder_fn: Any | None = None
        self.loss_compile_mode_kwargs: dict[str, Any] = {}

    def _resolve_compile_mode_override(
        self,
        *,
        base_mode: str,
        base_kwargs: dict[str, Any],
        fallback_mode_value: str | None,
    ) -> tuple[str, dict[str, Any]]:
        if fallback_mode_value is None:
            return base_mode, dict(base_kwargs)
        return validate_torch_compile_mode(fallback_mode_value)

    def _compile_wrapper(
        self,
        wrapper: torch.nn.Module,
        *,
        compile_kwargs: dict[str, Any],
        skip_compile: bool,
        skip_message: str,
    ) -> Any:
        if skip_compile:
            log_if_rank_zero(self.logger, skip_message, level="warning")
            return wrapper
        return torch.compile(wrapper, **compile_kwargs)

    def setup(self, cfg: DictConfig) -> None:
        compile_enabled: bool = bool(cfg.training.torch_compile)
        compile_available: bool = hasattr(torch, "compile")
        self.disable_compile_for_validation = bool(
            cfg.training.disable_compile_for_validation
        )
        self.torch_compile_enabled = compile_enabled and compile_available
        self.torch_compile_mark_step = None
        self.torch_compile_full_model = False
        self.compile_enabled_for_current_stage = False
        self.eager_model = self.model
        self.compiled_model = None
        self._compiled_shared_encoder_module = None
        self._eager_query_encoder_fn = None
        self._eager_doc_encoder_fn = None
        self._compiled_query_encoder_fn = None
        self._compiled_doc_encoder_fn = None
        self.loss_compile_mode_kwargs = {}
        if compile_enabled and not compile_available:
            log_if_rank_zero(
                self.logger,
                "torch.compile is not available in this PyTorch build; continuing "
                "without compilation.",
                level="warning",
            )
            return
        if not compile_enabled or not compile_available:
            return

        compile_mode_value: Any = cfg.training.torch_compile_mode
        compile_mode, compile_mode_kwargs = validate_torch_compile_mode(
            compile_mode_value
        )
        self.loss_compile_mode_kwargs = dict(compile_mode_kwargs)

        strategy_name: str = str(cfg.training.get("strategy", "")).lower()
        raw_num_devices: Any = cfg.training.get("num_devices", 1)
        try:
            configured_num_devices = (
                1 if raw_num_devices is None else int(raw_num_devices)
            )
        except (TypeError, ValueError):
            configured_num_devices = 1
        ddp_enabled: bool = ("ddp" in strategy_name) or (configured_num_devices > 1)
        encoder_obj: Any = getattr(self.model, "encoder", None)
        freeze_backbone: bool = bool(getattr(encoder_obj, "freeze_backbone", False))
        # Query/doc encoder wrappers share one encoder module. Under unfrozen DDP,
        # compiling wrappers separately can be unstable; compile shared encoder once.
        if ddp_enabled and not freeze_backbone:
            shared_encoder_mode: str = compile_mode
            shared_encoder_kwargs: dict[str, Any] = dict(compile_mode_kwargs)
            if compile_mode == "max-autotune":
                (
                    shared_encoder_mode,
                    shared_encoder_kwargs,
                ) = validate_torch_compile_mode("max-autotune-no-cudagraphs")
                log_if_rank_zero(
                    self.logger,
                    "Using shared-encoder torch.compile fallback "
                    f"{compile_mode!r} -> {shared_encoder_mode!r} for unfrozen DDP "
                    "to avoid unstable cudagraph capture.",
                    level="warning",
                )
            self.loss_compile_mode_kwargs = dict(shared_encoder_kwargs)
            self._eager_query_encoder_fn = self.model._query_encoder_wrapper
            self._eager_doc_encoder_fn = self.model._doc_encoder_wrapper
            shared_encoder_module: torch.nn.Module = cast(torch.nn.Module, encoder_obj)
            query_pooling_mode: torch.Tensor = cast(
                torch.Tensor, self.model._query_pooling_mode
            )
            doc_pooling_mode: torch.Tensor = cast(
                torch.Tensor, self.model._doc_pooling_mode
            )
            try:
                self._compiled_shared_encoder_module = torch.compile(
                    shared_encoder_module, **shared_encoder_kwargs
                )
            except Exception as exc:
                _, safe_loss_compile_kwargs = validate_torch_compile_mode("default")
                self.loss_compile_mode_kwargs = dict(safe_loss_compile_kwargs)
                log_if_rank_zero(
                    self.logger,
                    "Shared-encoder torch.compile setup failed for unfrozen DDP "
                    f"({exc!r}); continuing with eager encoder wrappers and "
                    "loss-only compile fallback.",
                    level="warning",
                )
                return

            compiled_shared_encoder: Callable[..., torch.Tensor] = cast(
                Callable[..., torch.Tensor],
                self._compiled_shared_encoder_module,
            )
            compiled_encoder_fn: Callable[..., torch.Tensor] = compiled_shared_encoder
            self._compiled_query_encoder_fn = _SharedCompiledEncoderAdapter(
                encoder_fn=compiled_encoder_fn,
                pooling_mode=query_pooling_mode,
            )
            self._compiled_doc_encoder_fn = _SharedCompiledEncoderAdapter(
                encoder_fn=compiled_encoder_fn,
                pooling_mode=doc_pooling_mode,
            )
            log_if_rank_zero(
                self.logger,
                "Enabled shared-encoder torch.compile for unfrozen DDP and "
                "disabled dual-wrapper compilation; this avoids the unstable path "
                "where the same trainable encoder is compiled twice (can segfault).",
                level="warning",
            )
            self.set_compile_state(use_compiled=True)
            return

        query_compile_mode: str = compile_mode
        doc_compile_mode: str = compile_mode
        loss_compile_mode: str = compile_mode
        query_compile_mode_kwargs: dict[str, Any] = dict(compile_mode_kwargs)
        doc_compile_mode_kwargs: dict[str, Any] = dict(compile_mode_kwargs)
        loss_compile_mode_kwargs: dict[str, Any] = dict(compile_mode_kwargs)
        skip_query_compile_for_large_vocab: bool = False
        skip_doc_compile_for_large_vocab: bool = False
        # Large-vocab heads can trigger unstable Triton autotune kernels.
        if compile_mode in {"max-autotune", "max-autotune-no-cudagraphs"}:
            try:
                vocab_size: int = int(getattr(encoder_obj, "vocab_size", 0))
            except Exception:
                vocab_size = 0
            large_vocab_threshold: int = int(
                cfg.training.get("torch_compile_large_vocab_threshold", 100000)
            )
            force_aten_for_large_vocab: bool = bool(
                cfg.training.get("torch_compile_force_aten_gemm_for_large_vocab", True)
            )
            if force_aten_for_large_vocab and vocab_size >= large_vocab_threshold:
                query_fallback_mode_value: str | None = normalize_optional_str(
                    cfg.training.get("torch_compile_large_vocab_query_fallback_mode")
                )
                doc_fallback_mode_value: str | None = normalize_optional_str(
                    cfg.training.get(
                        "torch_compile_large_vocab_doc_fallback_mode", "default"
                    )
                )
                loss_fallback_mode_value: str | None = normalize_optional_str(
                    cfg.training.get(
                        "torch_compile_large_vocab_loss_fallback_mode", "default"
                    )
                )
                query_compile_mode, query_compile_mode_kwargs = (
                    self._resolve_compile_mode_override(
                        base_mode=compile_mode,
                        base_kwargs=compile_mode_kwargs,
                        fallback_mode_value=query_fallback_mode_value,
                    )
                )
                doc_compile_mode, doc_compile_mode_kwargs = (
                    self._resolve_compile_mode_override(
                        base_mode=compile_mode,
                        base_kwargs=compile_mode_kwargs,
                        fallback_mode_value=doc_fallback_mode_value,
                    )
                )
                loss_compile_mode, loss_compile_mode_kwargs = (
                    self._resolve_compile_mode_override(
                        base_mode=compile_mode,
                        base_kwargs=compile_mode_kwargs,
                        fallback_mode_value=loss_fallback_mode_value,
                    )
                )
                skip_doc_compile_for_large_vocab = bool(
                    cfg.training.get(
                        "torch_compile_skip_doc_encoder_for_large_vocab", False
                    )
                )
                skip_query_compile_for_large_vocab = bool(
                    cfg.training.get(
                        "torch_compile_skip_query_encoder_for_large_vocab", False
                    )
                )
                if query_compile_mode != compile_mode:
                    log_if_rank_zero(
                        self.logger,
                        "Using per-wrapper compile mode override for query encoder: "
                        f"{compile_mode!r} -> {query_compile_mode!r}.",
                        level="warning",
                    )
                if doc_compile_mode != compile_mode:
                    log_if_rank_zero(
                        self.logger,
                        "Using per-wrapper compile mode override for doc encoder: "
                        f"{compile_mode!r} -> {doc_compile_mode!r}.",
                        level="warning",
                    )
                if loss_compile_mode != compile_mode:
                    log_if_rank_zero(
                        self.logger,
                        "Using compile mode override for loss module: "
                        f"{compile_mode!r} -> {loss_compile_mode!r}.",
                        level="warning",
                    )
                try:
                    import torch._inductor.config as inductor_config

                    safe_gemm_backends: str = "ATEN"
                    current_gemm_backends: str = str(
                        getattr(inductor_config, "max_autotune_gemm_backends", "")
                    )
                    if current_gemm_backends.upper() != safe_gemm_backends:
                        inductor_config.max_autotune_gemm_backends = (
                            safe_gemm_backends
                        )
                        log_if_rank_zero(
                            self.logger,
                            "Forcing torch.compile max-autotune GEMM backends to "
                            f"{safe_gemm_backends!r} because vocab_size={vocab_size} "
                            f"(>= {large_vocab_threshold}) can trigger Triton "
                            "autotune illegal-address failures.",
                            level="warning",
                        )
                except Exception as exc:
                    log_if_rank_zero(
                        self.logger,
                        "Could not configure torch._inductor GEMM backends for "
                        f"large-vocab compile safety: {exc!r}",
                        level="warning",
                    )

        compile_full_model: bool = compile_mode in {"reduce-overhead", "max-autotune"}
        per_wrapper_mode_override: bool = (
            query_compile_mode != compile_mode or doc_compile_mode != compile_mode
        )
        if compile_full_model and per_wrapper_mode_override:
            compile_full_model = False
            log_if_rank_zero(
                self.logger,
                "Falling back to wrapper-only torch.compile because per-wrapper "
                "compile mode overrides are enabled.",
                level="warning",
            )
        if compile_full_model:
            static_graph_enabled: bool = bool(cfg.training.static_graph)
            find_unused_parameters: bool = bool(
                cfg.training.get("find_unused_parameters", False)
            )
            if (not static_graph_enabled) or find_unused_parameters:
                compile_full_model = False
                log_if_rank_zero(
                    self.logger,
                    "Falling back to wrapper-only torch.compile despite "
                    f"mode={compile_mode!r} because training.static_graph="
                    f"{static_graph_enabled} and "
                    "training.find_unused_parameters="
                    f"{find_unused_parameters}. This keeps max-autotune enabled "
                    "while avoiding the full-model cudagraph path that is unstable "
                    "under dynamic DDP settings.",
                    level="warning",
                )
        self.loss_compile_mode_kwargs = dict(loss_compile_mode_kwargs)
        if compile_full_model:
            self.torch_compile_mark_step = resolve_cudagraph_mark_step()
            # Compile full model to avoid repeated wrapper-level cudagraph issues.
            self.compiled_model = cast(
                Any, torch.compile(self.model, **compile_mode_kwargs)
            )
            self.torch_compile_full_model = True
            self.set_compile_state(use_compiled=True)
            return

        # Wrapper-only compile keeps Lightning/module bookkeeping eager.
        query_wrapper: torch.nn.Module = cast(
            torch.nn.Module, self.model._query_encoder_wrapper
        )
        doc_wrapper: torch.nn.Module = cast(
            torch.nn.Module, self.model._doc_encoder_wrapper
        )
        self._eager_query_encoder_fn = query_wrapper
        self._eager_doc_encoder_fn = doc_wrapper
        self._compiled_query_encoder_fn = self._compile_wrapper(
            query_wrapper,
            compile_kwargs=query_compile_mode_kwargs,
            skip_compile=skip_query_compile_for_large_vocab,
            skip_message=(
                "Skipping torch.compile for query encoder wrapper under "
                "large-vocab max-autotune safety mode."
            ),
        )
        self._compiled_doc_encoder_fn = self._compile_wrapper(
            doc_wrapper,
            compile_kwargs=doc_compile_mode_kwargs,
            skip_compile=skip_doc_compile_for_large_vocab,
            skip_message=(
                "Skipping torch.compile for doc encoder wrapper under large-vocab "
                "max-autotune safety mode; query encoder remains compiled."
            ),
        )
        self.set_compile_state(use_compiled=True)

    def set_compile_state(
        self,
        *,
        use_compiled: bool,
        eager_loss_computer: Any | None = None,
        compiled_loss_computer: Any | None = None,
    ) -> Any | None:
        active_loss_computer: Any | None = None
        if eager_loss_computer is not None:
            if use_compiled and compiled_loss_computer is not None:
                active_loss_computer = compiled_loss_computer
            else:
                active_loss_computer = eager_loss_computer

        if not self.torch_compile_enabled:
            self.compile_enabled_for_current_stage = False
            return active_loss_computer
        if self.torch_compile_full_model:
            self.compile_enabled_for_current_stage = bool(
                use_compiled and self.compiled_model is not None
            )
            return active_loss_computer

        if use_compiled:
            if (
                self._compiled_query_encoder_fn is None
                or self._compiled_doc_encoder_fn is None
            ):
                self.compile_enabled_for_current_stage = False
                return active_loss_computer
            self.model._query_encoder_fn = self._compiled_query_encoder_fn
            self.model._doc_encoder_fn = self._compiled_doc_encoder_fn
            self.compile_enabled_for_current_stage = True
            return active_loss_computer

        if self._eager_query_encoder_fn is None or self._eager_doc_encoder_fn is None:
            self.compile_enabled_for_current_stage = False
            return active_loss_computer
        self.model._query_encoder_fn = self._eager_query_encoder_fn
        self.model._doc_encoder_fn = self._eager_doc_encoder_fn
        self.compile_enabled_for_current_stage = False
        return active_loss_computer

    def maybe_mark_step(self) -> None:
        if (
            self.compile_enabled_for_current_stage
            and self.torch_compile_mark_step is not None
        ):
            self.torch_compile_mark_step()

    def resolve_active_model_for_train_step(self) -> torch.nn.Module:
        if (
            self.torch_compile_full_model
            and self.compile_enabled_for_current_stage
            and self.compiled_model is not None
        ):
            return self.compiled_model
        return self.eager_model
