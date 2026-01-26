from __future__ import annotations

from functools import cached_property

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.datasets.beir import BEIRDataset


class EvalDataModule(L.LightningDataModule):
    """LightningDataModule for BEIR evaluation."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg

    @cached_property
    def test_dataset(self) -> BEIRDataset:
        dataset: BEIRDataset = BEIRDataset(cfg=self.cfg.dataset)
        return dataset

    def prepare_data(self) -> None:
        _ = self.test_dataset

    def setup(self, stage: str | None = None) -> None:
        _ = self.test_dataset

    def test_dataloader(self) -> DataLoader:
        dataset: BEIRDataset = self.test_dataset
        sampler: DistributedSampler | None = (
            DistributedSampler(dataset, shuffle=False)
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else None
        )
        shuffle: bool = False

        batch_size: int = int(
            getattr(self.cfg.testing, "per_device_batch_size", self.cfg.testing.batch_size)
        )

        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.cfg.testing.num_workers,
            collate_fn=dataset.collator,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=not self.cfg.testing.use_cpu,
        )
        return dataloader
