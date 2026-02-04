from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase

from src.data.pd_module import EncodePDModule
from src.utils.transformers import build_tokenizer


class EncodeDataModule(L.LightningDataModule):
    """LightningDataModule for corpus encoding."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.tokenizer: PreTrainedTokenizerBase = build_tokenizer(
            self.cfg.model.huggingface_name
        )
        self._dataset: EncodePDModule | None = None

    # --- Property methods ---
    @property
    def dataset(self) -> EncodePDModule:
        if self._dataset is None:
            self._dataset = EncodePDModule(
                cfg=self.cfg.dataset,
                tokenizer=self.tokenizer,
                seed=int(self.cfg.seed),
            )
        return self._dataset

    # --- Public methods ---
    def prepare_data(self) -> None:
        self.dataset.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.dataset.setup()

    def predict_dataloader(self) -> DataLoader:
        num_workers: int = int(self.cfg.encoding.num_workers)
        prefetch_factor: int | None = None
        if num_workers > 0:
            prefetch_factor = int(self.cfg.encoding.prefetch_factor)
        sampler: DistributedSampler | None = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(self.dataset, shuffle=False)
        dataloader_kwargs: dict[str, Any] = {
            "dataset": self.dataset,
            "batch_size": int(self.cfg.encoding.batch_size),
            "num_workers": num_workers,
            "collate_fn": self.dataset.collator,
            "drop_last": False,
            "pin_memory": not bool(self.cfg.encoding.use_cpu),
        }
        if sampler is not None:
            dataloader_kwargs["sampler"] = sampler
            dataloader_kwargs["shuffle"] = False
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(**dataloader_kwargs)
