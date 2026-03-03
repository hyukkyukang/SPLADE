import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.pd_module import PDModule
from src.data.collator import UniversalCollator
from src.data.dataclass import EncodingDataItem


class EncodePDModule(PDModule):
    """Encoding dataset module for corpus-only batches."""

    # --- Special methods ---
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        *,
        seed: int,
    ) -> None:
        super().__init__(cfg=cfg, tokenizer=tokenizer, seed=seed)
        self._corpus_indices: list[int] | None = None
        self._collator: UniversalCollator | None = None

    def __len__(self) -> int:
        return len(self._resolve_corpus_indices())

    def __getitem__(self, idx: int) -> EncodingDataItem:
        corpus_indices: list[int] = self._resolve_corpus_indices()
        corpus_idx: int = corpus_indices[int(idx)]
        doc_id: str = str(
            self.dataset.corpus_dataset[corpus_idx][self.dataset.corpus_id_column_name]
        )
        doc_text: str = self.dataset.corpus_text(corpus_idx)
        doc_input_ids: torch.Tensor
        doc_attention_mask: torch.Tensor
        doc_input_ids, doc_attention_mask = self._tokenize_text(
            doc_text, max_length=self.max_doc_length
        )
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
    def _resolve_corpus_indices(self) -> list[int]:
        if self._corpus_indices is not None:
            return self._corpus_indices
        dataset_len: int = int(len(self.dataset.corpus_dataset))
        self._corpus_indices = list(range(dataset_len))
        return self._corpus_indices

    def _requires_corpus_text_dataset(self) -> bool:
        return True

    # --- Public methods ---
    def setup(self) -> None:
        self._prepare_required_text_artifacts()
        _ = self._resolve_corpus_indices()

    def prepare_data(self) -> None:
        # Encode datasets can be BEIR-style and may not expose meta_dataset.
        # Keep prepare_data lightweight and defer text loading to setup().
        pass
