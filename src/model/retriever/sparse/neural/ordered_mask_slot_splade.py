from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from omegaconf import DictConfig

from src.model.retriever.sparse.neural.splade import SpladeModel


class OrderedMaskSlotSpladeModel(SpladeModel):
    """SPLADE variant that pools only over appended ordered mask slots."""

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
        num_mask_slots: int = 0,
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
                "OrderedMaskSlotSpladeModel requires token-aligned output dimensions."
            )
        self.num_mask_slots: int = max(int(num_mask_slots), 0)
        if self.num_mask_slots <= 0:
            raise ValueError("OrderedMaskSlotSpladeModel requires num_mask_slots > 0.")

        resolved_mask_token_id: int | None = (
            None if mask_token_id is None else int(mask_token_id)
        )
        if resolved_mask_token_id is None or resolved_mask_token_id < 0:
            config_mask_token_id: Any = getattr(
                self.encoder.mlm.config, "mask_token_id", None
            )
            if config_mask_token_id is not None:
                resolved_mask_token_id = int(config_mask_token_id)
        if resolved_mask_token_id is None or resolved_mask_token_id < 0:
            raise ValueError(
                "OrderedMaskSlotSpladeModel requires a valid mask token id."
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
        retrieval_exclude_output_ids: torch.Tensor = (
            self.encoder.resolve_output_exclude_ids(special_token_ids_tensor)
        )
        self.register_buffer(
            "_retrieval_exclude_output_ids",
            retrieval_exclude_output_ids.to(dtype=torch.long),
            persistent=False,
        )

    @property
    def supports_ordered_mask_slot_loss(self) -> bool:
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

    def extract_ordered_slot_logits(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 3 or attention_mask.ndim != 2:
            raise ValueError(
                "Ordered mask-slot extraction expects logits [B, L, V] and "
                "attention_mask [B, L]."
            )
        batch_size: int = int(logits.shape[0])
        vocab_size: int = int(logits.shape[-1])
        active_lengths: torch.Tensor = attention_mask.to(dtype=torch.long).sum(dim=1)
        device: torch.device = logits.device
        slot_offsets: torch.Tensor = torch.arange(
            self.num_mask_slots,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        slot_starts: torch.Tensor = active_lengths.unsqueeze(1) - int(self.num_mask_slots)
        slot_positions: torch.Tensor = slot_starts + slot_offsets
        gather_index: torch.Tensor = slot_positions.unsqueeze(-1).expand(
            batch_size, self.num_mask_slots, vocab_size
        )
        return logits.gather(1, gather_index)

    def _encode_with_slot_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None,
        pooling_mode: torch.Tensor,
        is_query: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits: torch.Tensor = self.encoder.encode_raw_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_scores: torch.Tensor = self.encoder.activation(logits)
        resolved_pooling_mask: torch.Tensor = (
            attention_mask if pooling_mask is None else pooling_mask
        )
        embeddings: torch.Tensor = self.encoder._pool_sparse(
            token_scores,
            resolved_pooling_mask,
            pooling_mode,
        )
        if is_query:
            embeddings = self.postprocess_query_embeddings(embeddings)
        else:
            embeddings = self.postprocess_doc_embeddings(embeddings)
        slot_logits: torch.Tensor = self.extract_ordered_slot_logits(
            logits,
            attention_mask,
        )
        return embeddings, slot_logits

    def encode_queries_with_slot_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_with_slot_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mask=pooling_mask,
            pooling_mode=self._query_pooling_mode,
            is_query=True,
        )

    def encode_docs_with_slot_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_with_slot_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mask=pooling_mask,
            pooling_mode=self._doc_pooling_mode,
            is_query=False,
        )
