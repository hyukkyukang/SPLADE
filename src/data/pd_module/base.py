import random
from typing import Any

import torch
from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import Dataset as PyTorchDataset
from transformers import PreTrainedTokenizerBase

from src.data.dataclass import MetaItem
from src.data.dataset import BaseDataset
from src.data.pd_module.utils import tokenize_docs, tokenize_text
from src.data.registry import build_dataset


class PDModule(PyTorchDataset):
    """Base class for PyTorch dataset modules."""

    # --- Special methods ---
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        *,
        seed: int,
        load_teacher_scores: bool | None = None,
        require_teacher_scores: bool | None = None,
    ) -> None:
        self.cfg: DictConfig = cfg
        self.name: str = str(self.cfg.name)
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.seed: int = int(seed)
        self._rng: random.Random = random.Random(self.seed)

        self.max_query_length: int = int(self.cfg.max_query_length)
        self.max_doc_length: int = int(self.cfg.max_doc_length)
        self.max_padding: bool = bool(self.cfg.max_padding)
        self.num_positives: int = int(self.cfg.num_positives)
        self.num_negatives: int = int(self.cfg.num_negatives)
        self.use_hf: bool = bool(
            self.cfg.hf_name is not None or self.cfg.query_corpus_hf_name is not None
        )

        self.load_teacher_scores: bool = (
            False if load_teacher_scores is None else bool(load_teacher_scores)
        )
        self.require_teacher_scores: bool = (
            False if require_teacher_scores is None else bool(require_teacher_scores)
        )

        self._dataset: BaseDataset | None = None

    def __len__(self) -> int:
        return int(len(self.meta_dataset))

    def __getitem__(self, idx: int) -> Any:
        """Get dataset item based on task mode."""
        raise NotImplementedError("Implement this method in the subclass.")

    # --- Property methods ---
    @property
    def dataset(self) -> BaseDataset:
        if self._dataset is None:
            self._dataset = build_dataset(self.cfg)
        return self._dataset

    @property
    def meta_dataset(self) -> Dataset:
        return self.dataset.meta_dataset

    # --- Protected methods ---
    def _build_meta_item(self, idx: int) -> MetaItem:
        row: dict[str, Any] = dict(self.meta_dataset[int(idx)])
        return self.dataset.build_meta_item(
            row,
            int(idx),
            num_positives=self.num_positives,
            num_negatives=self.num_negatives,
            rng=self._rng,
            load_teacher_scores=self.load_teacher_scores,
            require_teacher_scores=self.require_teacher_scores,
        )

    def _tokenize_text(
        self, text: str, *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return tokenize_text(
            self.tokenizer,
            text,
            max_length=max_length,
            max_padding=self.max_padding,
        )

    def _tokenize_docs(
        self, docs: list[str], *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return tokenize_docs(
            self.tokenizer,
            docs,
            max_length=max_length,
            max_padding=self.max_padding,
        )

    def _requires_query_text_dataset(self) -> bool:
        """Whether this module needs the query text dataset during iteration."""
        return False

    def _requires_corpus_text_dataset(self) -> bool:
        """Whether this module needs the corpus text dataset during iteration."""
        return False

    def _requires_query_id_to_idx(self) -> bool:
        """Whether this module needs the query id->index cache."""
        return False

    def _requires_corpus_id_to_idx(self) -> bool:
        """Whether this module needs the corpus id->index cache."""
        return False

    def _prepare_required_text_artifacts(self) -> None:
        """
        Warm only the text artifacts required by the concrete PD module.

        This runs in setup() so heavy lazy loads do not happen inside worker
        processes during the first __getitem__ calls.
        """
        if not self.use_hf:
            return
        requires_query_text: bool = self._requires_query_text_dataset()
        requires_corpus_text: bool = self._requires_corpus_text_dataset()
        requires_query_id_to_idx: bool = self._requires_query_id_to_idx()
        requires_corpus_id_to_idx: bool = self._requires_corpus_id_to_idx()

        if requires_query_text or requires_query_id_to_idx:
            _ = self.dataset.query_dataset
        if requires_corpus_text or requires_corpus_id_to_idx:
            _ = self.dataset.corpus_dataset
        if requires_query_id_to_idx:
            _ = self.dataset.query_dataset_id_to_idx
        if requires_corpus_id_to_idx:
            _ = self.dataset.corpus_dataset_id_to_idx

    # --- Public methods ---
    def prepare_data(self) -> None:
        # Keep prepare_data lightweight: prepare metadata/downloads only.
        self.dataset.prepare_meta_dataset()

    def setup(self) -> None:
        _ = self.meta_dataset
        self._prepare_required_text_artifacts()