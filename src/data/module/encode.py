from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from src.data.dataset.corpus import CorpusDataset
from src.data.dataset.ir_corpus import IRCorpusDataset

logger: logging.Logger = logging.getLogger("EncodeDataModule")


class EncodeDataModule(L.LightningDataModule):
    """LightningDataModule for corpus encoding."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self._dataset: CorpusDataset | IRCorpusDataset | None = None

    # --- Public methods ---
    @property
    def dataset(self) -> CorpusDataset | IRCorpusDataset:
        if self._dataset is None:
            self._dataset = self._build_dataset()
        return self._dataset

    def prepare_data(self) -> None:
        dataset: CorpusDataset | IRCorpusDataset = self.dataset
        # Download/cache the corpus once on the main process.
        try:
            dataset.prepare_data()
        except RuntimeError as exc:
            if self._should_fallback_to_ir_datasets(exc):
                logger.warning(
                    "Falling back to ir_datasets because HF dataset scripts "
                    "are no longer supported for BEIR corpora."
                )
                self._dataset = self._build_ir_dataset()
                self._dataset.prepare_data()
            else:
                raise

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        dataset: CorpusDataset | IRCorpusDataset = self.dataset
        # Load the corpus on each process.
        dataset.setup()

    def predict_dataloader(self) -> DataLoader:
        dataset: CorpusDataset | IRCorpusDataset = self.dataset
        is_iterable: bool = isinstance(dataset, IterableDataset)
        num_workers: int = int(self.cfg.encoding.num_workers)
        if isinstance(dataset, IRCorpusDataset) and num_workers > 0:
            logger.warning(
                "IRCorpusDataset is not picklable with multiprocessing workers; "
                "forcing encoding.num_workers=0."
            )
            num_workers = 0
        sampler: DistributedSampler | None = None
        if (
            not is_iterable
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            sampler = DistributedSampler(dataset, shuffle=False)
        batch_size: int = int(self.cfg.encoding.batch_size)
        dataloader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": dataset.collator,
            "drop_last": False,
            "pin_memory": not bool(self.cfg.encoding.use_cpu),
        }
        if sampler is not None:
            dataloader_kwargs["sampler"] = sampler
        if not is_iterable:
            dataloader_kwargs["shuffle"] = False
        dataloader: DataLoader = DataLoader(**dataloader_kwargs)
        return dataloader

    # --- Protected methods ---
    def _build_dataset(self) -> CorpusDataset | IRCorpusDataset:
        if self._use_ir_datasets():
            return self._build_ir_dataset()
        return CorpusDataset(cfg=self.cfg.dataset, global_cfg=self.cfg, tokenizer=None)

    def _build_ir_dataset(self) -> IRCorpusDataset:
        return IRCorpusDataset(cfg=self.cfg.dataset, global_cfg=self.cfg)

    def _use_ir_datasets(self) -> bool:
        return bool(getattr(self.cfg.dataset, "use_ir_datasets", False))

    def _should_fallback_to_ir_datasets(self, exc: Exception) -> bool:
        if not self._is_beir_dataset():
            return False
        return self._is_hf_script_unsupported_error(exc)

    def _is_beir_dataset(self) -> bool:
        beir_dataset: str | None = getattr(self.cfg.dataset, "beir_dataset", None)
        if beir_dataset:
            return True
        hf_name: str | None = getattr(self.cfg.dataset, "hf_name", None)
        return bool(hf_name and str(hf_name).lower().startswith("beir/"))

    def _is_hf_script_unsupported_error(self, exc: Exception) -> bool:
        return isinstance(exc, RuntimeError) and (
            "Dataset scripts are no longer supported" in str(exc)
        )
