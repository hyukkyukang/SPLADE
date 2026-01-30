from __future__ import annotations

from functools import cached_property

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.dataset.corpus import CorpusDataset


class EncodeDataModule(L.LightningDataModule):
    """LightningDataModule for corpus encoding."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg

    # --- Public methods ---
    @cached_property
    def dataset(self) -> CorpusDataset:
        return CorpusDataset(cfg=self.cfg.dataset, global_cfg=self.cfg, tokenizer=None)

    def prepare_data(self) -> None:
        _ = self.dataset

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        _ = self.dataset

    def predict_dataloader(self) -> DataLoader:
        sampler: DistributedSampler | None = (
            DistributedSampler(self.dataset, shuffle=False)
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else None
        )
        batch_size: int = int(self.cfg.encoding.batch_size)
        dataloader: DataLoader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=int(self.cfg.encoding.num_workers),
            collate_fn=self.dataset.collator,
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            pin_memory=not bool(self.cfg.encoding.use_cpu),
        )
        return dataloader
