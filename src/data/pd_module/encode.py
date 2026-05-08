import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.pd_module import PDModule
from src.data.collator import UniversalCollator
from src.data.lens_formatting import build_doc_pooling_mask
from src.data.dataclass import EncodingDataItem
from src.data.patent_text import (
    PATENT_DOCUMENT_TEMPLATE_NAME,
    format_patent_document_text_prefix,
)
from src.data.pd_module.utils import (
    resolve_num_mask_slots,
    tokenize_text_windows,
    tokenize_text_with_mask_slots,
    uses_ordered_mask_slot_pooling,
)


class EncodePDModule(PDModule):
    """Encoding dataset module for corpus-only batches."""

    # --- Special methods ---
    def __init__(
        self,
        cfg: DictConfig,
        encoding_cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        *,
        model_cfg: DictConfig | None = None,
        seed: int,
    ) -> None:
        super().__init__(
            cfg=cfg,
            tokenizer=tokenizer,
            model_cfg=model_cfg,
            seed=seed,
        )
        self.encoding_cfg: DictConfig = encoding_cfg
        self._collator: UniversalCollator | None = None
        strategy: str = str(
            self.encoding_cfg.get("long_doc_strategy", "truncate")
        ).strip().lower()
        if strategy not in {"truncate", "sliding_window"}:
            raise ValueError(
                "encoding.long_doc_strategy must be one of: truncate, sliding_window."
            )
        self._long_doc_strategy: str = strategy
        self._window_overlap_tokens: int = max(
            0, int(self.encoding_cfg.get("sliding_window_overlap_tokens", 0))
        )
        self._truncate_prefix_chars_per_token: int = max(
            0, int(self.encoding_cfg.get("truncate_prefix_chars_per_token", 0))
        )
        self._truncate_prefix_min_chars: int = max(
            0, int(self.encoding_cfg.get("truncate_prefix_min_chars", 4096))
        )

    def __len__(self) -> int:
        return int(len(self.dataset.corpus_dataset))

    def __getitem__(self, idx: int) -> EncodingDataItem:
        corpus_idx: int = int(idx)
        row: dict[str, object] = self.dataset.corpus_dataset[corpus_idx]
        doc_id: str = str(row[self.dataset.corpus_id_column_name])
        doc_group_id: str | None = None
        group_id_column_name: str | None = self.dataset.corpus_group_id_column_name
        if group_id_column_name is not None:
            raw_group_id: object | None = row.get(group_id_column_name)
            if raw_group_id is not None:
                resolved_group_id: str = str(raw_group_id).strip()
                if resolved_group_id:
                    doc_group_id = resolved_group_id
        prefix_builder = self._build_truncate_prefix_builder(row)
        doc_text: str = (
            ""
            if prefix_builder is not None
            else self.dataset._corpus_text_from_row(row)
        )
        doc_input_ids: torch.Tensor
        doc_attention_mask: torch.Tensor
        doc_pooling_mask: torch.Tensor
        doc_input_ids, doc_attention_mask, doc_pooling_mask = self._tokenize_doc_text(
            doc_text,
            prefix_builder=prefix_builder,
        )
        return EncodingDataItem(
            data_idx=int(idx),
            doc_id=doc_id,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            doc_pooling_mask=doc_pooling_mask,
            doc_group_id=doc_group_id,
        )

    # --- Property methods ---
    @property
    def collator(self) -> UniversalCollator:
        if self._collator is None:
            self._collator = UniversalCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                max_padding=self.max_padding,
                max_doc_length=self.max_doc_length,
            )
        return self._collator

    # --- Protected methods ---
    def _tokenize_doc_text(
        self,
        doc_text: str,
        *,
        prefix_builder=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if uses_ordered_mask_slot_pooling(self.model_cfg):
            if self._long_doc_strategy != "truncate":
                raise NotImplementedError(
                    "Sliding-window encoding is not yet supported for ordered "
                    "mask-slot models."
                )
            doc_input_ids: torch.Tensor
            doc_attention_mask: torch.Tensor
            doc_pooling_mask: torch.Tensor
            doc_input_ids, doc_attention_mask, doc_pooling_mask = (
                tokenize_text_with_mask_slots(
                    self.tokenizer,
                    doc_text,
                    max_length=self.max_doc_length,
                    num_mask_slots=resolve_num_mask_slots(self.model_cfg),
                    max_padding=self.max_padding,
                    model_cfg=self.model_cfg,
                )
            )
            return (
                doc_input_ids.unsqueeze(0),
                doc_attention_mask.unsqueeze(0),
                doc_pooling_mask.unsqueeze(0),
            )
        if self._long_doc_strategy == "truncate":
            doc_input_ids, doc_attention_mask = self._tokenize_text(
                doc_text,
                max_length=self.max_doc_length,
                fast_truncate_chars_per_token=self._truncate_prefix_chars_per_token,
                fast_truncate_min_chars=self._truncate_prefix_min_chars,
                prefix_builder=prefix_builder,
            )
            doc_pooling_mask = build_doc_pooling_mask(
                doc_attention_mask.unsqueeze(0),
                self.model_cfg,
            )
            return doc_input_ids.unsqueeze(0), doc_attention_mask.unsqueeze(0), doc_pooling_mask
        doc_input_ids, doc_attention_mask = tokenize_text_windows(
            self.tokenizer,
            doc_text,
            max_length=self.max_doc_length,
            max_padding=self.max_padding,
            overlap_tokens=self._window_overlap_tokens,
        )
        doc_pooling_mask = build_doc_pooling_mask(
            doc_attention_mask,
            self.model_cfg,
        )
        return doc_input_ids, doc_attention_mask, doc_pooling_mask

    def _build_truncate_prefix_builder(self, row: dict[str, object]):
        if self._truncate_prefix_chars_per_token <= 0:
            return None
        if self._long_doc_strategy != "truncate":
            return None
        template_name: str | None = self.dataset.corpus_text_template
        if (
            template_name is not None
            and str(template_name).strip().lower() == PATENT_DOCUMENT_TEMPLATE_NAME
        ):
            return lambda budget: format_patent_document_text_prefix(
                row,
                char_budget=int(budget),
            )
        return None

    def _requires_corpus_text_dataset(self) -> bool:
        return True

    # --- Public methods ---
    def setup(self) -> None:
        self._prepare_required_text_artifacts()

    def prepare_data(self) -> None:
        # Encode datasets can be BEIR-style and may not expose meta_dataset.
        # Keep prepare_data lightweight and defer text loading to setup().
        pass
