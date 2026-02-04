from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase

from src.data.pd_module import RetrievalPDModule
from src.utils.transformers import build_tokenizer


class RetrievalDataModule(L.LightningDataModule):
    """LightningDataModule for BEIR retrieval evaluation."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.tokenizer: PreTrainedTokenizerBase = build_tokenizer(
            self.cfg.model.huggingface_name
        )
        self._dataset: RetrievalPDModule | None = None

    # --- Property methods ---
    @property
    def dataset(self) -> RetrievalPDModule:
        if self._dataset is None:
            self._dataset = RetrievalPDModule(
                cfg=self.cfg.dataset,
                tokenizer=self.tokenizer,
                seed=int(self.cfg.seed),
                load_teacher_scores=False,
                require_teacher_scores=False,
            )
        return self._dataset

    # --- Public methods ---
    def prepare_data(self) -> None:
        self.dataset.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.dataset.setup()

    def test_dataloader(self) -> DataLoader:
        num_workers: int = int(self.cfg.testing.num_workers)
        sampler: DistributedSampler | None = (
            DistributedSampler(self.dataset, shuffle=False)
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else None
        )
        batch_size: int = int(self.cfg.testing.batch_size)
        dataloader_kwargs: dict[str, Any] = {
            "dataset": self.dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": self.dataset.collator,
            "sampler": sampler,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": not bool(self.cfg.testing.use_cpu),
        }
        return DataLoader(**dataloader_kwargs)
