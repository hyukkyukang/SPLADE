from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as F
from omegaconf import DictConfig

from src.model.retriever.sparse.neural.splade import SpladeModel


class MDLMSpladeModel(SpladeModel):
    """Fixed-vocabulary MDLM-trained SPLADE variant.

    Retrieval stays identical to SPLADE at inference time: one clean forward,
    SPLADE pooling over raw MLM logits, and the existing sparse index/search
    pipeline. Training can optionally add an MDLM-style masked diffusion
    auxiliary loss through ``compute_mdlm_aux_loss``.
    """

    def __init__(
        self,
        family: str,
        model_name: str,
        huggingface_model_class: str,
        query_pooling: str,
        doc_pooling: str,
        sparse_activation: str,
        *,
        attn_implementation: str | None = None,
        dtype: torch.dtype | None = None,
        normalize: bool = False,
        doc_only: bool = False,
        tie_word_embeddings: bool = False,
        peft_cfg: DictConfig | None = None,
        freeze_backbone: bool = False,
        trust_remote_code: bool = False,
        model_revision: str | None = None,
        local_files_only: bool | None = None,
        exclude_token_ids: Sequence[int] | None = None,
        mask_token_id: int | None = None,
    ) -> None:
        super().__init__(
            family=family,
            model_name=model_name,
            huggingface_model_class=huggingface_model_class,
            query_pooling=query_pooling,
            doc_pooling=doc_pooling,
            sparse_activation=sparse_activation,
            attn_implementation=attn_implementation,
            dtype=dtype,
            normalize=normalize,
            doc_only=doc_only,
            tie_word_embeddings=tie_word_embeddings,
            peft_cfg=peft_cfg,
            freeze_backbone=freeze_backbone,
            trust_remote_code=trust_remote_code,
            model_revision=model_revision,
            local_files_only=local_files_only,
        )
        if not bool(self.encoder.output_token_aligned):
            raise ValueError(
                "MDLMSpladeModel requires token-aligned output dimensions."
            )

        resolved_mask_token_id: int | None = (
            None if mask_token_id is None else int(mask_token_id)
        )
        if resolved_mask_token_id is None or resolved_mask_token_id < 0:
            config_mask_token_id: Any = getattr(self.encoder.mlm.config, "mask_token_id", None)
            if config_mask_token_id is not None:
                resolved_mask_token_id = int(config_mask_token_id)
        if resolved_mask_token_id is None or resolved_mask_token_id < 0:
            raise ValueError(
                "MDLMSpladeModel requires a valid mask token id. Set model.mask_token_id "
                "in config when the Hugging Face config does not expose it."
            )
        self.mask_token_id: int = int(resolved_mask_token_id)

        token_ids: list[int] = self._resolve_special_token_ids(
            configured_ids=exclude_token_ids,
        )
        special_token_ids_tensor: torch.Tensor = torch.tensor(
            token_ids,
            dtype=torch.long,
        )
        self.register_buffer(
            "_special_token_ids",
            special_token_ids_tensor,
            persistent=False,
        )
        retrieval_exclude_output_ids: torch.Tensor = self.encoder.resolve_output_exclude_ids(
            special_token_ids_tensor
        )
        self.register_buffer(
            "_retrieval_exclude_output_ids",
            retrieval_exclude_output_ids.to(dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_mdlm_large_negative",
            torch.tensor(-1e9, dtype=torch.float32),
            persistent=False,
        )
        self._mdlm_large_negative_value: float = -1e9

    @property
    def supports_mdlm_aux_loss(self) -> bool:
        return True

    def _resolve_special_token_ids(
        self,
        *,
        configured_ids: Sequence[int] | None,
    ) -> list[int]:
        if configured_ids is not None and len(configured_ids) > 0:
            candidate_ids: list[int] = [int(token_id) for token_id in configured_ids]
        else:
            candidate_ids = []
            config: Any = self.encoder.mlm.config
            field_name: str
            for field_name in (
                "pad_token_id",
                "cls_token_id",
                "sep_token_id",
                "bos_token_id",
                "eos_token_id",
                "unk_token_id",
            ):
                value: Any = getattr(config, field_name, None)
                if value is None:
                    continue
                token_id: int = int(value)
                if token_id >= 0:
                    candidate_ids.append(token_id)
        candidate_ids.append(int(self.mask_token_id))
        return sorted(set(candidate_ids))

    def _apply_retrieval_output_mask(self, embeddings: torch.Tensor) -> torch.Tensor:
        exclude_output_ids: torch.Tensor = self._retrieval_exclude_output_ids
        if int(exclude_output_ids.numel()) == 0:
            return embeddings
        masked_embeddings: torch.Tensor = embeddings.clone()
        masked_embeddings.index_fill_(
            1,
            exclude_output_ids.to(device=masked_embeddings.device),
            0.0,
        )
        return masked_embeddings

    def postprocess_query_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self._apply_retrieval_output_mask(embeddings)
        return super().postprocess_query_embeddings(embeddings)

    def postprocess_doc_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self._apply_retrieval_output_mask(embeddings)
        return super().postprocess_doc_embeddings(embeddings)

    def denoiser_raw_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        return self.encoder.encode_raw_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **model_kwargs,
        )

    def build_special_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        special_token_ids: torch.Tensor = self._special_token_ids
        if int(special_token_ids.numel()) == 0:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        return torch.isin(
            input_ids,
            special_token_ids.to(device=input_ids.device),
        )

    def sample_noisy_view(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_token_mask: torch.Tensor | None = None,
        *,
        mask_probability_eps: float = 1e-3,
        force_mask_at_least_one: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("MDLM noisy-view sampling expects rank-2 tensors.")
        batch_size: int = int(input_ids.shape[0])
        device: torch.device = input_ids.device
        eps: float = max(float(mask_probability_eps), 1e-6)
        t: torch.Tensor = torch.rand(batch_size, device=device, dtype=torch.float32)
        t = t.clamp(min=eps, max=1.0)
        resolved_special_mask: torch.Tensor = (
            self.build_special_token_mask(input_ids)
            if special_token_mask is None
            else special_token_mask.to(dtype=torch.bool, device=device)
        )
        valid_mask: torch.Tensor = attention_mask.to(dtype=torch.bool, device=device)
        valid_mask = valid_mask & (~resolved_special_mask)

        bernoulli: torch.Tensor = torch.rand(
            input_ids.shape,
            device=device,
            dtype=torch.float32,
        )
        masked: torch.Tensor = bernoulli < t.unsqueeze(1)
        masked = masked & valid_mask

        if bool(force_mask_at_least_one):
            valid_counts: torch.Tensor = valid_mask.sum(dim=1)
            needs_forced_mask: torch.Tensor = (masked.sum(dim=1) == 0) & (valid_counts > 0)
            selection_scores: torch.Tensor = torch.rand(
                input_ids.shape,
                device=device,
                dtype=torch.float32,
            )
            selection_scores = selection_scores.masked_fill(~valid_mask, -1.0)
            forced_positions: torch.Tensor = selection_scores.argmax(dim=1)
            forced_rows: torch.Tensor = torch.nonzero(
                needs_forced_mask, as_tuple=False
            ).squeeze(1)
            masked[forced_rows, forced_positions[forced_rows]] = True

        xt: torch.Tensor = input_ids.clone()
        xt[masked] = int(self.mask_token_id)
        return xt, masked, t

    def subs_log_probs(
        self,
        raw_logits: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        logits: torch.Tensor = raw_logits.float().clone()
        logits[:, :, int(self.mask_token_id)] = self._mdlm_large_negative.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        log_probs: torch.Tensor = F.log_softmax(logits, dim=-1)

        unmasked: torch.Tensor = xt != int(self.mask_token_id)
        carry: torch.Tensor = torch.full_like(
            log_probs,
            self._mdlm_large_negative_value,
        )
        carry.scatter_(-1, xt.unsqueeze(-1), 0.0)
        return torch.where(unmasked.unsqueeze(-1), carry, log_probs)

    def _masked_target_nll(
        self,
        raw_logits: torch.Tensor,
        target_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked-token NLL without materializing a second dense [B, L, V] tensor.

        MDLM loss only supervises masked positions, so we do not need the full SUBS
        carry-over distribution for unmasked tokens during training. We only need the
        log-probability of the clean target token under the masked-position denoiser
        distribution with the `[MASK]` token forbidden.
        """

        logits: torch.Tensor = raw_logits.float()
        logits[:, :, int(self.mask_token_id)] = self._mdlm_large_negative.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        target_logits: torch.Tensor = torch.gather(
            logits,
            dim=-1,
            index=target_input_ids.unsqueeze(-1),
        ).squeeze(-1)
        return torch.logsumexp(logits, dim=-1) - target_logits

    def _compute_mdlm_aux_loss_per_example(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        special_token_mask: torch.Tensor | None = None,
        mask_probability_eps: float = 1e-3,
        force_mask_at_least_one: bool = True,
    ) -> torch.Tensor:
        resolved_special_mask: torch.Tensor = (
            self.build_special_token_mask(input_ids)
            if special_token_mask is None
            else special_token_mask.to(dtype=torch.bool, device=input_ids.device)
        )
        xt: torch.Tensor
        masked: torch.Tensor
        t: torch.Tensor
        xt, masked, t = self.sample_noisy_view(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_token_mask=resolved_special_mask,
            mask_probability_eps=mask_probability_eps,
            force_mask_at_least_one=force_mask_at_least_one,
        )
        raw_logits: torch.Tensor = self.denoiser_raw_logits(
            xt,
            attention_mask,
            timesteps=t,
        )
        token_nll: torch.Tensor = self._masked_target_nll(raw_logits, input_ids)
        masked_float: torch.Tensor = masked.to(dtype=token_nll.dtype)
        per_example_loss: torch.Tensor = (token_nll * masked_float).sum(dim=1)
        per_example_loss = per_example_loss / masked_float.sum(dim=1).clamp_min(1.0)
        per_example_loss = per_example_loss / t.clamp_min(
            max(float(mask_probability_eps), 1e-6)
        )
        return per_example_loss

    def compute_grouped_mdlm_aux_losses(
        self,
        *,
        input_id_groups: Sequence[torch.Tensor],
        attention_mask_groups: Sequence[torch.Tensor],
        special_token_mask_groups: Sequence[torch.Tensor | None] | None = None,
        reduction_mask_groups: Sequence[torch.Tensor | None] | None = None,
        mask_probability_eps: float = 1e-3,
        force_mask_at_least_one: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        if len(input_id_groups) != len(attention_mask_groups):
            raise ValueError(
                "input_id_groups and attention_mask_groups must have the same length."
            )
        if special_token_mask_groups is not None and len(special_token_mask_groups) != len(
            input_id_groups
        ):
            raise ValueError(
                "special_token_mask_groups must match input_id_groups length."
            )
        if reduction_mask_groups is not None and len(reduction_mask_groups) != len(
            input_id_groups
        ):
            raise ValueError(
                "reduction_mask_groups must match input_id_groups length."
            )
        if len(input_id_groups) == 0:
            raise ValueError("compute_grouped_mdlm_aux_losses requires at least one group.")

        batch_sizes: list[int] = []
        combined_input_groups: list[torch.Tensor] = []
        combined_attention_groups: list[torch.Tensor] = []
        combined_special_mask_groups: list[torch.Tensor] = []
        reference_input: torch.Tensor | None = None
        expected_seq_length: int | None = None

        for group_index, (group_input_ids, group_attention_mask) in enumerate(
            zip(input_id_groups, attention_mask_groups)
        ):
            if group_input_ids.ndim != 2 or group_attention_mask.ndim != 2:
                raise ValueError("Grouped MDLM loss expects rank-2 input tensors.")
            if group_input_ids.shape != group_attention_mask.shape:
                raise ValueError("Grouped MDLM inputs must share input/mask shapes.")
            group_batch_size: int = int(group_input_ids.shape[0])
            batch_sizes.append(group_batch_size)
            if group_batch_size == 0:
                continue
            if expected_seq_length is None:
                expected_seq_length = int(group_input_ids.shape[1])
            elif int(group_input_ids.shape[1]) != expected_seq_length:
                raise ValueError(
                    "Grouped MDLM loss requires all non-empty groups to share sequence length."
                )
            reference_input = group_input_ids
            combined_input_groups.append(group_input_ids)
            combined_attention_groups.append(group_attention_mask)
            special_group_mask: torch.Tensor | None = None
            if special_token_mask_groups is not None:
                special_group_mask = special_token_mask_groups[group_index]
            resolved_special_mask: torch.Tensor = (
                self.build_special_token_mask(group_input_ids)
                if special_group_mask is None
                else special_group_mask.to(
                    dtype=torch.bool, device=group_input_ids.device
                )
            )
            combined_special_mask_groups.append(resolved_special_mask)

        if reference_input is None:
            zero = torch.zeros((), dtype=torch.float32)
            return tuple(zero.clone() for _ in input_id_groups)

        combined_input_ids: torch.Tensor = torch.cat(combined_input_groups, dim=0)
        combined_attention_mask: torch.Tensor = torch.cat(combined_attention_groups, dim=0)
        combined_special_token_mask: torch.Tensor = torch.cat(
            combined_special_mask_groups, dim=0
        )
        combined_per_example_loss: torch.Tensor = self._compute_mdlm_aux_loss_per_example(
            combined_input_ids,
            combined_attention_mask,
            special_token_mask=combined_special_token_mask,
            mask_probability_eps=mask_probability_eps,
            force_mask_at_least_one=force_mask_at_least_one,
        )
        resolved_losses: list[torch.Tensor] = []
        offset: int = 0
        batch_size: int
        for group_index, batch_size in enumerate(batch_sizes):
            if batch_size == 0:
                resolved_losses.append(
                    reference_input.new_zeros((), dtype=torch.float32)
                )
                continue
            next_offset: int = offset + batch_size
            group_losses: torch.Tensor = combined_per_example_loss[offset:next_offset]
            reduction_mask: torch.Tensor | None = None
            if reduction_mask_groups is not None:
                reduction_mask = reduction_mask_groups[group_index]
            if reduction_mask is None:
                resolved_losses.append(group_losses.mean())
            else:
                resolved_mask: torch.Tensor = reduction_mask.to(
                    device=group_losses.device
                )
                if resolved_mask.ndim != 1 or int(resolved_mask.shape[0]) != batch_size:
                    raise ValueError(
                        "Each reduction mask must be rank-1 and match its group batch size."
                    )
                mask_float: torch.Tensor = resolved_mask.to(dtype=group_losses.dtype)
                denom: torch.Tensor = mask_float.sum().clamp_min(1.0)
                resolved_losses.append((group_losses * mask_float).sum() / denom)
            offset = next_offset
        return tuple(resolved_losses)

    def compute_mdlm_aux_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        special_token_mask: torch.Tensor | None = None,
        mask_probability_eps: float = 1e-3,
        force_mask_at_least_one: bool = True,
    ) -> torch.Tensor:
        return self._compute_mdlm_aux_loss_per_example(
            input_ids,
            attention_mask,
            special_token_mask=special_token_mask,
            mask_probability_eps=mask_probability_eps,
            force_mask_at_least_one=force_mask_at_least_one,
        ).mean()
