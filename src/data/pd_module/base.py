import random
from typing import Any

import torch
from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import Dataset as PyTorchDataset
from transformers import PreTrainedTokenizerBase

from src.data.dataclass import MetaItem
from src.data.dataset import BaseDataset
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
        padding: str | bool = "max_length" if self.max_padding else True
        tokens: dict[str, torch.Tensor] = self.tokenizer(
            text,
            padding=padding,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        input_ids: torch.Tensor = tokens["input_ids"].squeeze(0)
        attention_mask: torch.Tensor = tokens["attention_mask"].squeeze(0)
        return input_ids, attention_mask

    def _tokenize_docs(
        self, docs: list[str], *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding: str | bool = "max_length" if self.max_padding else True
        if not docs:
            empty_ids: torch.Tensor = torch.empty((0, max_length), dtype=torch.long)
            empty_mask: torch.Tensor = torch.empty((0, max_length), dtype=torch.long)
            return empty_ids, empty_mask
        tokens: dict[str, torch.Tensor] = self.tokenizer(
            list(docs),
            padding=padding,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        input_ids: torch.Tensor = tokens["input_ids"]
        attention_mask: torch.Tensor = tokens["attention_mask"]
        return input_ids, attention_mask

    # --- Public methods ---
    def prepare_data(self) -> None:
        self.dataset.prepare_meta_dataset()
        if self.use_hf:
            self.dataset.prepare_text_datasets()

    def setup(self) -> None:
        _ = self.meta_dataset
