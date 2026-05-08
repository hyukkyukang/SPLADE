import logging
from typing import Any, Callable, cast

import torch
from torch.nn import functional as F
from omegaconf import DictConfig

from src.model.losses import LossComputer
from src.model.pl_module.utils import (
    is_max_autotune_mode,
    resolve_cudagraph_mark_step,
    validate_torch_compile_mode,
)
from src.utils.logging import log_if_rank_zero
from src.utils.normalize import normalize_optional_str
from src.utils.trainer import resolve_effective_distributed_settings


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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._encoder_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mode=self._pooling_mode,
            pooling_mask=pooling_mask,
        )


class _CompiledTrainCoreAdapter(torch.nn.Module):
    """Compile encode + loss as one train-step core module."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        loss_computer: LossComputer,
        mdlm_enabled: bool = False,
        mdlm_doc_selection: str = "all",
        mdlm_doc_chunk_size: int = 0,
        mdlm_eps: float = 1e-3,
        mdlm_force_mask_at_least_one: bool = True,
        mdlm_single_positive_assumption: bool = False,
        ordered_mask_slot_enabled: bool = False,
        ordered_mask_query_weight: float = 0.0,
        ordered_mask_doc_weight: float = 0.0,
        ordered_mask_ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.model: torch.nn.Module = model
        self.loss_computer: LossComputer = loss_computer
        self._mdlm_enabled: bool = bool(mdlm_enabled)
        self._mdlm_doc_selection: str = str(mdlm_doc_selection).strip().lower()
        self._mdlm_doc_chunk_size: int = max(int(mdlm_doc_chunk_size), 0)
        self._mdlm_eps: float = max(float(mdlm_eps), 1e-6)
        self._mdlm_force_mask_at_least_one: bool = bool(
            mdlm_force_mask_at_least_one
        )
        self._mdlm_single_positive_assumption: bool = bool(
            mdlm_single_positive_assumption
        )
        splade_model: Any = model
        self._ordered_mask_slot_enabled: bool = bool(
            ordered_mask_slot_enabled
            and bool(getattr(splade_model, "supports_ordered_mask_slot_loss", False))
        )
        self._ordered_mask_query_weight: float = float(ordered_mask_query_weight)
        self._ordered_mask_doc_weight: float = float(ordered_mask_doc_weight)
        self._ordered_mask_ignore_index: int = int(ordered_mask_ignore_index)
        self._can_fuse_query_doc_encoding_static: bool = False
        if not bool(getattr(splade_model, "doc_only", False)):
            query_pooling_mode: Any = getattr(splade_model, "_query_pooling_mode", None)
            doc_pooling_mode: Any = getattr(splade_model, "_doc_pooling_mode", None)
            if (
                isinstance(query_pooling_mode, torch.Tensor)
                and isinstance(doc_pooling_mode, torch.Tensor)
                and query_pooling_mode.numel() == 1
                and doc_pooling_mode.numel() == 1
            ):
                self._can_fuse_query_doc_encoding_static = bool(
                    torch.equal(query_pooling_mode, doc_pooling_mode)
                )
        self._can_include_compiled_mdlm_static: bool = bool(
            self._mdlm_enabled
            and self._mdlm_doc_selection == "positives"
            and self._mdlm_doc_chunk_size <= 0
            and callable(
                getattr(splade_model, "compute_grouped_mdlm_aux_losses", None)
            )
        )

    def _encode_with_mode(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mode: torch.Tensor,
        pooling_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        splade_model: Any = self.model
        encoder: torch.nn.Module = cast(torch.nn.Module, splade_model.encoder)
        embeddings: torch.Tensor = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mode=pooling_mode,
            pooling_mask=pooling_mask,
        )
        return embeddings

    def _can_fuse_query_doc_encoding(
        self,
        *,
        query_input_ids: torch.Tensor,
        doc_input_ids: torch.Tensor,
    ) -> bool:
        if not self._can_fuse_query_doc_encoding_static:
            return False
        return int(query_input_ids.shape[1]) == int(doc_input_ids.shape[1])

    def resolve_compiled_mdlm_apply_mode(
        self,
        *,
        query_seq_len: int,
        doc_seq_len: int,
    ) -> str:
        if not self._can_include_compiled_mdlm_static:
            return "never"
        if int(query_seq_len) != int(doc_seq_len):
            return "never"
        if self._mdlm_single_positive_assumption:
            return "always"
        return "runtime_flag"

    def _compute_compiled_mdlm_aux_losses(
        self,
        *,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero: torch.Tensor = query_input_ids.new_zeros((), dtype=torch.float32)
        if not self._can_include_compiled_mdlm_static:
            return zero, zero, zero, zero
        if int(query_input_ids.shape[1]) != int(doc_input_ids.shape[1]):
            return zero, zero, zero, zero

        mdlm_model: Any = self.model
        batch_size: int = int(query_input_ids.shape[0])
        doc_count: int = int(doc_mask.shape[1])
        reshaped_doc_input_ids: torch.Tensor = doc_input_ids.view(
            batch_size, doc_count, -1
        )
        reshaped_doc_attention_mask: torch.Tensor = doc_attention_mask.view(
            batch_size, doc_count, -1
        )
        valid_positive_mask: torch.Tensor = pos_mask.to(dtype=torch.bool)
        valid_positive_mask = valid_positive_mask & doc_mask.to(dtype=torch.bool)
        positive_counts: torch.Tensor = valid_positive_mask.sum(dim=1)
        positive_row_mask: torch.Tensor = positive_counts.eq(1)
        exact_semantics_applied: torch.Tensor = positive_counts.le(1).all().to(
            dtype=zero.dtype
        )

        positive_indices: torch.Tensor = valid_positive_mask.to(
            dtype=torch.int64
        ).argmax(dim=1)
        batch_indices: torch.Tensor = torch.arange(
            batch_size,
            device=doc_input_ids.device,
            dtype=torch.long,
        )
        selected_doc_input_ids: torch.Tensor = reshaped_doc_input_ids[
            batch_indices, positive_indices
        ]
        selected_doc_attention_mask: torch.Tensor = reshaped_doc_attention_mask[
            batch_indices, positive_indices
        ]
        mdlm_q_loss: torch.Tensor
        mdlm_d_loss: torch.Tensor
        mdlm_q_loss, mdlm_d_loss = mdlm_model.compute_grouped_mdlm_aux_losses(
            input_id_groups=(query_input_ids, selected_doc_input_ids),
            attention_mask_groups=(
                query_attention_mask,
                selected_doc_attention_mask,
            ),
            reduction_mask_groups=(None, positive_row_mask),
            mask_probability_eps=self._mdlm_eps,
            force_mask_at_least_one=self._mdlm_force_mask_at_least_one,
        )
        mdlm_q_loss = mdlm_q_loss * exact_semantics_applied
        mdlm_d_loss = mdlm_d_loss * exact_semantics_applied
        mdlm_total_loss: torch.Tensor = mdlm_q_loss + mdlm_d_loss
        return mdlm_q_loss, mdlm_d_loss, mdlm_total_loss, exact_semantics_applied

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

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        teacher_scores: torch.Tensor,
        lambda_scale: torch.Tensor,
        query_pooling_mask: torch.Tensor | None = None,
        doc_pooling_mask: torch.Tensor | None = None,
        query_slot_target_ids: torch.Tensor | None = None,
        doc_slot_target_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        splade_model: Any = self.model
        query_pooling_mode: torch.Tensor = cast(
            torch.Tensor, splade_model._query_pooling_mode
        )
        doc_pooling_mode: torch.Tensor = cast(
            torch.Tensor, splade_model._doc_pooling_mode
        )
        query_slot_logits: torch.Tensor | None = None
        flat_doc_slot_logits: torch.Tensor | None = None
        ordered_mask_slot_enabled: bool = bool(
            self._ordered_mask_slot_enabled
            and query_slot_target_ids is not None
            and doc_slot_target_ids is not None
        )
        outputs_already_postprocessed: bool = False
        if ordered_mask_slot_enabled:
            q_reps, query_slot_logits = splade_model.encode_queries_with_slot_logits(
                query_input_ids,
                query_attention_mask,
                pooling_mask=query_pooling_mask,
            )
            flat_doc_reps, flat_doc_slot_logits = (
                splade_model.encode_docs_with_slot_logits(
                    doc_input_ids,
                    doc_attention_mask,
                    pooling_mask=doc_pooling_mask,
                )
            )
            outputs_already_postprocessed = True
        elif self._can_fuse_query_doc_encoding(
            query_input_ids=query_input_ids,
            doc_input_ids=doc_input_ids,
        ):
            resolved_query_pooling_mask: torch.Tensor = (
                query_attention_mask
                if query_pooling_mask is None
                else query_pooling_mask
            )
            resolved_doc_pooling_mask: torch.Tensor = (
                doc_attention_mask if doc_pooling_mask is None else doc_pooling_mask
            )
            combined_input_ids: torch.Tensor = torch.cat(
                (query_input_ids, doc_input_ids), dim=0
            )
            combined_attention_mask: torch.Tensor = torch.cat(
                (query_attention_mask, doc_attention_mask), dim=0
            )
            combined_pooling_mask: torch.Tensor = torch.cat(
                (resolved_query_pooling_mask, resolved_doc_pooling_mask), dim=0
            )
            combined_reps: torch.Tensor = self._encode_with_mode(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                pooling_mode=query_pooling_mode,
                pooling_mask=combined_pooling_mask,
            )
            query_batch_size: int = int(query_input_ids.shape[0])
            q_reps: torch.Tensor = combined_reps[:query_batch_size]
            flat_doc_reps: torch.Tensor = combined_reps[query_batch_size:]
        else:
            q_reps = self._encode_with_mode(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                pooling_mode=query_pooling_mode,
                pooling_mask=query_pooling_mask,
            )
            flat_doc_reps = self._encode_with_mode(
                input_ids=doc_input_ids,
                attention_mask=doc_attention_mask,
                pooling_mode=doc_pooling_mode,
                pooling_mask=doc_pooling_mask,
            )
        if not outputs_already_postprocessed:
            q_reps = splade_model.postprocess_query_embeddings(q_reps)
            flat_doc_reps = splade_model.postprocess_doc_embeddings(flat_doc_reps)
        batch_size: int = int(query_input_ids.shape[0])
        doc_count: int = int(doc_mask.shape[1])
        doc_reps: torch.Tensor = flat_doc_reps.view(batch_size, doc_count, -1)
        loss_outputs: tuple[torch.Tensor, ...] = self.loss_computer(
            q_reps=q_reps,
            doc_reps=doc_reps,
            pos_mask=pos_mask,
            doc_mask=doc_mask,
            teacher_scores=teacher_scores,
            lambda_scale=lambda_scale.to(dtype=q_reps.dtype, device=q_reps.device),
        )
        ordered_query_slot_loss: torch.Tensor = query_input_ids.new_zeros(
            (), dtype=torch.float32
        )
        ordered_doc_slot_loss: torch.Tensor = query_input_ids.new_zeros(
            (), dtype=torch.float32
        )
        ordered_total_loss: torch.Tensor = query_input_ids.new_zeros(
            (), dtype=torch.float32
        )
        if (
            ordered_mask_slot_enabled
            and query_slot_logits is not None
            and flat_doc_slot_logits is not None
            and query_slot_target_ids is not None
            and doc_slot_target_ids is not None
        ):
            ordered_query_slot_loss = self._compute_ordered_mask_slot_loss(
                slot_logits=query_slot_logits,
                target_ids=query_slot_target_ids,
            )
            ordered_doc_slot_loss = self._compute_ordered_mask_slot_loss(
                slot_logits=flat_doc_slot_logits,
                target_ids=doc_slot_target_ids.view(batch_size * doc_count, -1),
            )
            ordered_total_loss = (
                self._ordered_mask_query_weight * ordered_query_slot_loss
            ) + (self._ordered_mask_doc_weight * ordered_doc_slot_loss)
            loss_outputs = (
                loss_outputs[0] + ordered_total_loss,
                *loss_outputs[1:],
            )
        mdlm_q_loss: torch.Tensor
        mdlm_d_loss: torch.Tensor
        mdlm_total_loss: torch.Tensor
        mdlm_applied: torch.Tensor
        mdlm_q_loss, mdlm_d_loss, mdlm_total_loss, mdlm_applied = (
            self._compute_compiled_mdlm_aux_losses(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                doc_input_ids=doc_input_ids,
                doc_attention_mask=doc_attention_mask,
                pos_mask=pos_mask,
                doc_mask=doc_mask,
            )
        )
        return (
            q_reps,
            flat_doc_reps,
            *loss_outputs,
            ordered_query_slot_loss,
            ordered_doc_slot_loss,
            ordered_total_loss,
            mdlm_q_loss,
            mdlm_d_loss,
            mdlm_total_loss,
            mdlm_applied,
        )


class _CompiledMDLMAuxAdapter(torch.nn.Module):
    """Compile MDLM auxiliary loss for one fixed-shape input family."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        mdlm_eps: float = 1e-3,
        mdlm_force_mask_at_least_one: bool = True,
    ) -> None:
        super().__init__()
        self.model: torch.nn.Module = model
        self._mdlm_eps: float = max(float(mdlm_eps), 1e-6)
        self._mdlm_force_mask_at_least_one: bool = bool(
            mdlm_force_mask_at_least_one
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mdlm_model: Any = self.model
        return mdlm_model.compute_mdlm_aux_loss(
            input_ids,
            attention_mask,
            mask_probability_eps=self._mdlm_eps,
            force_mask_at_least_one=self._mdlm_force_mask_at_least_one,
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
        self._defer_train_core_compile: bool = False
        self._train_core_compile_kwargs: dict[str, Any] = {}
        self._eager_train_core_module: torch.nn.Module | None = None
        self._compiled_train_core_module: Any | None = None
        self.use_compiled_train_core_current_stage: bool = False
        self._compiled_query_mdlm_aux_module: Any | None = None
        self._compiled_doc_mdlm_aux_module: Any | None = None
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

    @staticmethod
    def _module_has_device_mismatch(
        module: torch.nn.Module, *, device: torch.device
    ) -> bool:
        parameter: torch.nn.Parameter
        for parameter in module.parameters(recurse=True):
            if parameter.device != device:
                return True
        buffer: torch.Tensor
        for buffer in module.buffers(recurse=True):
            if buffer.device != device:
                return True
        return False

    def _move_module_to_device_if_needed(
        self, maybe_module: Any | None, *, device: torch.device
    ) -> None:
        if not isinstance(maybe_module, torch.nn.Module):
            return
        if not self._module_has_device_mismatch(maybe_module, device=device):
            return
        maybe_module.to(device=device)

    def _setup_shared_encoder_compile(
        self,
        *,
        shared_encoder_kwargs: dict[str, Any],
    ) -> bool:
        encoder_obj: Any = getattr(self.model, "encoder", None)
        query_pooling_mode: torch.Tensor = cast(
            torch.Tensor, self.model._query_pooling_mode
        )
        doc_pooling_mode: torch.Tensor = cast(torch.Tensor, self.model._doc_pooling_mode)
        self._eager_query_encoder_fn = self.model._query_encoder_wrapper
        self._eager_doc_encoder_fn = self.model._doc_encoder_wrapper
        shared_encoder_module: torch.nn.Module = cast(torch.nn.Module, encoder_obj)
        self.loss_compile_mode_kwargs = dict(shared_encoder_kwargs)
        try:
            self._compiled_shared_encoder_module = torch.compile(
                shared_encoder_module, **shared_encoder_kwargs
            )
        except Exception as exc:
            _, safe_loss_compile_kwargs = validate_torch_compile_mode("default")
            self.loss_compile_mode_kwargs = dict(safe_loss_compile_kwargs)
            log_if_rank_zero(
                self.logger,
                "Shared-encoder torch.compile setup failed for unfrozen "
                "distributed training "
                f"({exc!r}); continuing with eager encoder wrappers and "
                "loss-only compile fallback.",
                level="warning",
            )
            return False

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
            "Enabled shared-encoder torch.compile for unfrozen distributed "
            "training and "
            "disabled dual-wrapper compilation; this avoids the unstable path "
            "where the same trainable encoder is compiled twice (can segfault).",
            level="warning",
        )
        self.set_compile_state(use_compiled=True)
        return True

    def finalize_train_core_compile(
        self,
        *,
        loss_computer: LossComputer,
        mdlm_enabled: bool = False,
        mdlm_doc_selection: str = "all",
        mdlm_doc_chunk_size: int = 0,
        mdlm_eps: float = 1e-3,
        mdlm_force_mask_at_least_one: bool = True,
        mdlm_single_positive_assumption: bool = False,
        ordered_mask_slot_enabled: bool = False,
        ordered_mask_query_weight: float = 0.0,
        ordered_mask_doc_weight: float = 0.0,
        ordered_mask_ignore_index: int = -100,
    ) -> bool:
        def _maybe_compile_mdlm_aux_modules() -> None:
            if not mdlm_enabled or not bool(
                getattr(self.eager_model, "supports_mdlm_aux_loss", False)
            ):
                return
            compiled_query_aux: Any
            compiled_doc_aux: Any
            try:
                compiled_query_aux = torch.compile(
                    _CompiledMDLMAuxAdapter(
                        model=self.eager_model,
                        mdlm_eps=mdlm_eps,
                        mdlm_force_mask_at_least_one=mdlm_force_mask_at_least_one,
                    ),
                    **self._train_core_compile_kwargs,
                )
                compiled_doc_aux = torch.compile(
                    _CompiledMDLMAuxAdapter(
                        model=self.eager_model,
                        mdlm_eps=mdlm_eps,
                        mdlm_force_mask_at_least_one=mdlm_force_mask_at_least_one,
                    ),
                    **self._train_core_compile_kwargs,
                )
            except Exception as exc:
                self._compiled_query_mdlm_aux_module = None
                self._compiled_doc_mdlm_aux_module = None
                log_if_rank_zero(
                    self.logger,
                    "MDLM auxiliary torch.compile setup failed "
                    f"({exc!r}); continuing with eager MDLM auxiliary loss.",
                    level="warning",
                )
                return
            self._compiled_query_mdlm_aux_module = compiled_query_aux
            self._compiled_doc_mdlm_aux_module = compiled_doc_aux

        if not self._defer_train_core_compile or not self.torch_compile_enabled:
            return False
        eager_train_core_module = _CompiledTrainCoreAdapter(
            model=self.eager_model,
            loss_computer=loss_computer,
            mdlm_enabled=mdlm_enabled,
            mdlm_doc_selection=mdlm_doc_selection,
            mdlm_doc_chunk_size=mdlm_doc_chunk_size,
            mdlm_eps=mdlm_eps,
            mdlm_force_mask_at_least_one=mdlm_force_mask_at_least_one,
            mdlm_single_positive_assumption=mdlm_single_positive_assumption,
            ordered_mask_slot_enabled=ordered_mask_slot_enabled,
            ordered_mask_query_weight=ordered_mask_query_weight,
            ordered_mask_doc_weight=ordered_mask_doc_weight,
            ordered_mask_ignore_index=ordered_mask_ignore_index,
        )
        self._eager_train_core_module = eager_train_core_module
        try:
            self._compiled_train_core_module = torch.compile(
                eager_train_core_module, **self._train_core_compile_kwargs
            )
        except Exception as exc:
            self._compiled_train_core_module = None
            self._eager_train_core_module = None
            self._defer_train_core_compile = False
            log_if_rank_zero(
                self.logger,
                "Train-core torch.compile setup failed "
                f"({exc!r}); falling back to shared-encoder compile.",
                level="warning",
            )
            setup_ok: bool = self._setup_shared_encoder_compile(
                shared_encoder_kwargs=self._train_core_compile_kwargs
            )
            if setup_ok:
                _maybe_compile_mdlm_aux_modules()
            return setup_ok
        _maybe_compile_mdlm_aux_modules()
        self.compile_enabled_for_current_stage = True
        return True

    def has_compiled_train_core(self) -> bool:
        return self.use_compiled_train_core_current_stage and (
            self._compiled_train_core_module is not None
        )

    def compiled_train_core_available(self) -> bool:
        return self._compiled_train_core_module is not None

    def compiled_train_core_mdlm_apply_mode(
        self,
        *,
        query_seq_len: int,
        doc_seq_len: int,
    ) -> str:
        if self._eager_train_core_module is None:
            return "never"
        if not isinstance(self._eager_train_core_module, _CompiledTrainCoreAdapter):
            return "never"
        return self._eager_train_core_module.resolve_compiled_mdlm_apply_mode(
            query_seq_len=query_seq_len,
            doc_seq_len=doc_seq_len,
        )

    def has_compiled_query_mdlm_aux(self) -> bool:
        return self.compile_enabled_for_current_stage and (
            self._compiled_query_mdlm_aux_module is not None
        )

    def has_compiled_doc_mdlm_aux(self) -> bool:
        return self.compile_enabled_for_current_stage and (
            self._compiled_doc_mdlm_aux_module is not None
        )

    def set_train_core_active(self, active: bool) -> None:
        self.use_compiled_train_core_current_stage = bool(
            active and self._compiled_train_core_module is not None
        )

    def run_compiled_train_core(self, **kwargs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self._compiled_train_core_module is None:
            raise RuntimeError("Compiled train-core module is not initialized.")
        compiled_train_core: Callable[..., tuple[torch.Tensor, ...]] = cast(
            Callable[..., tuple[torch.Tensor, ...]],
            self._compiled_train_core_module,
        )
        return compiled_train_core(**kwargs)

    def run_compiled_query_mdlm_aux(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._compiled_query_mdlm_aux_module is None:
            raise RuntimeError("Compiled query MDLM aux module is not initialized.")
        compiled_query_aux: Callable[..., torch.Tensor] = cast(
            Callable[..., torch.Tensor], self._compiled_query_mdlm_aux_module
        )
        return compiled_query_aux(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def run_compiled_doc_mdlm_aux(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._compiled_doc_mdlm_aux_module is None:
            raise RuntimeError("Compiled doc MDLM aux module is not initialized.")
        compiled_doc_aux: Callable[..., torch.Tensor] = cast(
            Callable[..., torch.Tensor], self._compiled_doc_mdlm_aux_module
        )
        return compiled_doc_aux(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

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
        self._defer_train_core_compile = False
        self._train_core_compile_kwargs = {}
        self._eager_train_core_module = None
        self._compiled_train_core_module = None
        self.use_compiled_train_core_current_stage = False
        self._compiled_query_mdlm_aux_module = None
        self._compiled_doc_mdlm_aux_module = None
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

        runtime_settings = resolve_effective_distributed_settings(cfg.training)
        encoder_obj: Any = getattr(self.model, "encoder", None)
        freeze_backbone: bool = bool(getattr(encoder_obj, "freeze_backbone", False))
        doc_only_enabled: bool = bool(getattr(self.model, "doc_only", False))
        peft_enabled: bool = bool(getattr(self.model, "peft_enabled", False)) or bool(
            getattr(encoder_obj, "peft_enabled", False)
        )
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
        if (
            is_max_autotune_mode(compile_mode)
            and force_aten_for_large_vocab
            and vocab_size >= large_vocab_threshold
        ):
            try:
                import torch._inductor.config as inductor_config

                safe_gemm_backends: str = "ATEN"
                current_gemm_backends: str = str(
                    getattr(inductor_config, "max_autotune_gemm_backends", "")
                )
                if current_gemm_backends.upper() != safe_gemm_backends:
                    inductor_config.max_autotune_gemm_backends = safe_gemm_backends
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
        enable_train_core_compile: bool = bool(
            cfg.training.get("torch_compile_train_core_when_possible", False)
        )
        # Query/doc encoder wrappers share one encoder module. Under unfrozen
        # distributed training, compiling wrappers separately can be unstable;
        # compile the shared encoder once.
        if runtime_settings.distributed_enabled and not freeze_backbone:
            shared_encoder_kwargs: dict[str, Any] = dict(compile_mode_kwargs)
            self.loss_compile_mode_kwargs = dict(shared_encoder_kwargs)
            if enable_train_core_compile and not doc_only_enabled and not peft_enabled:
                self._defer_train_core_compile = True
                self._train_core_compile_kwargs = dict(shared_encoder_kwargs)
                log_if_rank_zero(
                    self.logger,
                    "Deferring shared-encoder compile to build a larger compiled "
                    "train-core module after loss initialization.",
                    level="warning",
                )
                return
            self._setup_shared_encoder_compile(
                shared_encoder_kwargs=shared_encoder_kwargs
            )
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
        if is_max_autotune_mode(compile_mode):
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

        compile_full_model: bool = compile_mode == "reduce-overhead" or (
            is_max_autotune_mode(compile_mode)
        )
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
        if compile_full_model and doc_only_enabled:
            compile_full_model = False
            log_if_rank_zero(
                self.logger,
                "Falling back to wrapper-only torch.compile because model.doc_only "
                "uses a bag-of-words query path that is not safe for full-model "
                "compile.",
                level="warning",
            )
        if compile_full_model and peft_enabled:
            compile_full_model = False
            log_if_rank_zero(
                self.logger,
                "Falling back to wrapper-only torch.compile because PEFT-wrapped "
                "models are not enabled for the full-model compile path.",
                level="warning",
            )
        if compile_full_model:
            if not runtime_settings.full_model_compile_safe:
                compile_full_model = False
                log_if_rank_zero(
                    self.logger,
                    "Falling back to wrapper-only torch.compile despite "
                    f"mode={compile_mode!r} because the effective runtime settings "
                    "do not permit the full-model cudagraph path "
                    f"(static_graph={runtime_settings.static_graph}, "
                    "find_unused_parameters="
                    f"{runtime_settings.find_unused_parameters}). This keeps "
                    "max-autotune enabled while avoiding the unstable path under "
                    "dynamic distributed settings.",
                    level="warning",
                )
        skip_query_compile_for_doc_only: bool = doc_only_enabled
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
            skip_compile=(
                skip_query_compile_for_large_vocab or skip_query_compile_for_doc_only
            ),
            skip_message=(
                "Skipping torch.compile for query encoder wrapper because "
                "model.doc_only uses a bag-of-words query path."
                if skip_query_compile_for_doc_only
                else "Skipping torch.compile for query encoder wrapper under "
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

    def prepare_for_device(self, *, device: torch.device, use_compiled: bool) -> None:
        if not self.torch_compile_enabled or not use_compiled:
            return
        target_device: torch.device = torch.device(device)
        if self.torch_compile_full_model:
            self._move_module_to_device_if_needed(
                self.compiled_model, device=target_device
            )
            return
        self._move_module_to_device_if_needed(
            self._compiled_train_core_module, device=target_device
        )
        self._move_module_to_device_if_needed(
            self._compiled_query_mdlm_aux_module, device=target_device
        )
        self._move_module_to_device_if_needed(
            self._compiled_doc_mdlm_aux_module, device=target_device
        )
        self._move_module_to_device_if_needed(
            self._compiled_shared_encoder_module, device=target_device
        )
        self._move_module_to_device_if_needed(
            self._compiled_query_encoder_fn, device=target_device
        )
        self._move_module_to_device_if_needed(
            self._compiled_doc_encoder_fn, device=target_device
        )

    def set_compile_state(
        self,
        *,
        use_compiled: bool,
    ) -> None:
        if not self.torch_compile_enabled:
            self.compile_enabled_for_current_stage = False
            return
        if self.torch_compile_full_model:
            self.compile_enabled_for_current_stage = bool(
                use_compiled and self.compiled_model is not None
            )
            return
        if self._compiled_train_core_module is not None:
            if use_compiled and self.use_compiled_train_core_current_stage:
                self.model._query_encoder_fn = self.model._query_encoder_wrapper
                self.model._doc_encoder_fn = self.model._doc_encoder_wrapper
                self.compile_enabled_for_current_stage = True
                return

        if use_compiled:
            if (
                self._compiled_query_encoder_fn is None
                or self._compiled_doc_encoder_fn is None
            ):
                self.compile_enabled_for_current_stage = False
                return
            self.model._query_encoder_fn = self._compiled_query_encoder_fn
            self.model._doc_encoder_fn = self._compiled_doc_encoder_fn
            self.compile_enabled_for_current_stage = True
            return

        if self._eager_query_encoder_fn is None or self._eager_doc_encoder_fn is None:
            self.compile_enabled_for_current_stage = False
            return
        self.model._query_encoder_fn = self._eager_query_encoder_fn
        self.model._doc_encoder_fn = self._eager_doc_encoder_fn
        self.compile_enabled_for_current_stage = False
        return

    def can_fuse_query_doc_encoding(self) -> bool:
        model: torch.nn.Module = self.eager_model
        if bool(getattr(model, "doc_only", False)):
            return False
        query_pooling_mode: Any = getattr(model, "_query_pooling_mode", None)
        doc_pooling_mode: Any = getattr(model, "_doc_pooling_mode", None)
        if not isinstance(query_pooling_mode, torch.Tensor):
            return False
        if not isinstance(doc_pooling_mode, torch.Tensor):
            return False
        if query_pooling_mode.numel() != 1 or doc_pooling_mode.numel() != 1:
            return False
        return float(query_pooling_mode.item()) == float(doc_pooling_mode.item())

    def encode_queries_and_docs(
        self,
        *,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        query_pooling_mask: torch.Tensor | None = None,
        doc_pooling_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.can_fuse_query_doc_encoding():
            raise RuntimeError(
                "Query/doc fused encoding is not available for the active model."
            )

        model: torch.nn.Module = self.eager_model
        query_batch_size: int = int(query_input_ids.shape[0])
        combined_input_ids: torch.Tensor = torch.cat(
            (query_input_ids, doc_input_ids), dim=0
        )
        combined_attention_mask: torch.Tensor = torch.cat(
            (query_attention_mask, doc_attention_mask), dim=0
        )
        resolved_query_pooling_mask: torch.Tensor = (
            query_attention_mask if query_pooling_mask is None else query_pooling_mask
        )
        resolved_doc_pooling_mask: torch.Tensor = (
            doc_attention_mask if doc_pooling_mask is None else doc_pooling_mask
        )
        combined_pooling_mask: torch.Tensor = torch.cat(
            (resolved_query_pooling_mask, resolved_doc_pooling_mask), dim=0
        )
        pooling_mode: torch.Tensor = cast(torch.Tensor, model._query_pooling_mode)
        encoder_fn: Callable[..., torch.Tensor]
        if (
            self.compile_enabled_for_current_stage
            and self._compiled_shared_encoder_module is not None
        ):
            encoder_fn = cast(
                Callable[..., torch.Tensor], self._compiled_shared_encoder_module
            )
        else:
            encoder_fn = cast(Callable[..., torch.Tensor], model.encoder)
        combined_reps: torch.Tensor = encoder_fn(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            pooling_mode=pooling_mode,
            pooling_mask=combined_pooling_mask,
        )
        query_reps: torch.Tensor = combined_reps[:query_batch_size]
        doc_reps: torch.Tensor = combined_reps[query_batch_size:]
        postprocess_query_embeddings: Any = getattr(
            model, "postprocess_query_embeddings", None
        )
        if callable(postprocess_query_embeddings):
            query_reps = postprocess_query_embeddings(query_reps)
        elif bool(getattr(model, "normalize", False)):
            query_reps = torch.nn.functional.normalize(query_reps, p=2, dim=-1)
        postprocess_doc_embeddings: Any = getattr(
            model, "postprocess_doc_embeddings", None
        )
        if callable(postprocess_doc_embeddings):
            doc_reps = postprocess_doc_embeddings(doc_reps)
        elif bool(getattr(model, "normalize", False)):
            doc_reps = torch.nn.functional.normalize(doc_reps, p=2, dim=-1)
        return query_reps, doc_reps

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
