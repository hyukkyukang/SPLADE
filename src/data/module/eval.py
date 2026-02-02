from __future__ import annotations

import logging
from functools import cached_property

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.dataset.beir import BEIRDataset
from src.data.dataset.ir_beir import IRBEIRDataset

logger: logging.Logger = logging.getLogger("EvalDataModule")


class EvalDataModule(L.LightningDataModule):
    """LightningDataModule for BEIR evaluation."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg

    @cached_property
    def test_dataset(self) -> BEIRDataset | IRBEIRDataset:
        if self._use_ir_datasets():
            dataset: IRBEIRDataset = IRBEIRDataset(cfg=self.cfg.dataset)
            return dataset
        dataset: BEIRDataset = BEIRDataset(cfg=self.cfg.dataset)
        return dataset

    # --- Protected methods ---
    def _use_ir_datasets(self) -> bool:
        return bool(getattr(self.cfg.dataset, "use_ir_datasets", False))

    def prepare_data(self) -> None:
        _ = self.test_dataset

    def setup(self, stage: str | None = None) -> None:
        _ = self.test_dataset

    def test_dataloader(self) -> DataLoader:
        dataset: BEIRDataset | IRBEIRDataset = self.test_dataset
        num_workers: int = int(self.cfg.testing.num_workers)
        if isinstance(dataset, IRBEIRDataset) and num_workers > 0:
            logger.warning(
                "IRBEIRDataset is not picklable with multiprocessing workers; "
                "forcing testing.num_workers=0."
            )
            num_workers = 0
        sampler: DistributedSampler | None = (
            DistributedSampler(dataset, shuffle=False)
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else None
        )
        shuffle: bool = False

        per_device_batch_size: int | None = self.cfg.testing.per_device_batch_size
        batch_size: int = int(
            per_device_batch_size
            if per_device_batch_size is not None
            else self.cfg.testing.batch_size
        )

        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collator,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=not self.cfg.testing.use_cpu,
        )
        return dataloader
