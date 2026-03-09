import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.pd_module import PDModule
from src.data.collator import UniversalCollator
from src.data.dataclass import EncodingDataItem
from src.data.pd_module.utils import tokenize_text_windows


class EncodePDModule(PDModule):
    """Encoding dataset module for corpus-only batches."""

    # --- Special methods ---
    def __init__(
        self,
        cfg: DictConfig,
        encoding_cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        *,
        seed: int,
    ) -> None:
        super().__init__(cfg=cfg, tokenizer=tokenizer, seed=seed)
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

    def __len__(self) -> int:
        return int(len(self.dataset.corpus_dataset))

    def __getitem__(self, idx: int) -> EncodingDataItem:
        corpus_idx: int = int(idx)
        row: dict[str, object] = self.dataset.corpus_dataset[corpus_idx]
        doc_id: str = str(row[self.dataset.corpus_id_column_name])
        doc_text: str = self.dataset._corpus_text_from_row(row)
        doc_input_ids: torch.Tensor
        doc_attention_mask: torch.Tensor
        doc_input_ids, doc_attention_mask = self._tokenize_doc_text(doc_text)
        return EncodingDataItem(
            data_idx=int(idx),
            doc_id=doc_id,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
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
    def _tokenize_doc_text(self, doc_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self._long_doc_strategy == "truncate":
            doc_input_ids, doc_attention_mask = self._tokenize_text(
                doc_text, max_length=self.max_doc_length
            )
            return doc_input_ids.unsqueeze(0), doc_attention_mask.unsqueeze(0)
        return tokenize_text_windows(
            self.tokenizer,
            doc_text,
            max_length=self.max_doc_length,
            max_padding=self.max_padding,
            overlap_tokens=self._window_overlap_tokens,
        )

    def _requires_corpus_text_dataset(self) -> bool:
        return True

    # --- Public methods ---
    def setup(self) -> None:
        self._prepare_required_text_artifacts()

    def prepare_data(self) -> None:
        # Encode datasets can be BEIR-style and may not expose meta_dataset.
        # Keep prepare_data lightweight and defer text loading to setup().
        pass
