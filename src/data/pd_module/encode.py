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
        return EncodingDataItem(
            data_idx=int(idx),
            doc_id=doc_id,
            doc_text=doc_text,
        )

    # --- Property methods ---
    @property
    def collator(self) -> UniversalCollator:
        if self._collator is None:
            self._collator = UniversalCollator(
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self._collator

    # --- Protected methods ---
    def _resolve_corpus_indices(self) -> list[int]:
        if self._corpus_indices is not None:
            return self._corpus_indices
        dataset_len: int = int(len(self.dataset.corpus_dataset))
        skip_samples: int = int(self.cfg.hf_skip_samples)
        max_samples: int | None = (
            None if self.cfg.hf_max_samples is None else int(self.cfg.hf_max_samples)
        )
        start_index: int = min(skip_samples, dataset_len)
        end_index: int = dataset_len
        if max_samples is not None:
            end_index = min(start_index + max_samples, dataset_len)
        self._corpus_indices = list(range(start_index, end_index))
        return self._corpus_indices

    # --- Public methods ---
    def setup(self) -> None:
        _ = self._resolve_corpus_indices()

    def prepare_data(self) -> None:
        _ = self.dataset.corpus_dataset
