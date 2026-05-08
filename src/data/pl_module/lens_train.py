"""LightningDataModule for paper-faithful LENS training on ``cfli/bge-full-data``.

Builds:
- :class:`src.data.dataset.bge_full_data.BgeFullDataset`
- :class:`src.data.sampler.SameDatasetBatchSampler` (task-pure batches across DDP)
- :class:`src.data.lens_collator.LENSCollator`

and hands them to a DataLoader with ``batch_sampler=...``, ``num_workers``,
``prefetch_factor``, ``pin_memory`` sized for the 8×A100-40GB box.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, List

import lightning as L
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from src.data.dataset.bge_full_data import BgeFullDataset
from src.data.lens_collator import LENSCollator
from src.data.pl_module.common import build_model_tokenizer
from src.data.sampler import SameDatasetBatchSampler


class _RawRowDataset(Dataset[Any]):
    """Trivial adapter: integer index → raw row dict from the HF dataset."""

    def __init__(self, raw: BgeFullDataset) -> None:
        self._raw: BgeFullDataset = raw

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, idx: int) -> Any:
        return self._raw[int(idx)]


class LENSTrainDataModule(L.LightningDataModule):
    """LightningDataModule wrapping the LENS multi-task data pipeline."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.prepare_data_per_node = False
        self.tokenizer: PreTrainedTokenizerBase = build_model_tokenizer(self.cfg.model)

    # --- Public state --------------------------------------------------------

    @cached_property
    def raw_dataset(self) -> BgeFullDataset:
        train_cfg: DictConfig = self.cfg.train_dataset
        return BgeFullDataset(
            local_dir=str(train_cfg.path),
            hf_name=str(train_cfg.get("hf_name", "cfli/bge-full-data")),
            cache_dir=train_cfg.get("cache_dir"),
        )

    @cached_property
    def train_dataset(self) -> _RawRowDataset:
        return _RawRowDataset(self.raw_dataset)

    # --- Lightning hooks -----------------------------------------------------

    def train_dataloader(self) -> DataLoader[Any]:
        train_cfg: DictConfig = self.cfg.train_dataset
        lens_cfg: DictConfig = self.cfg.training.lens_multi_task

        # Per-task batch sizes (per rank).
        batch_size: int = int(self.cfg.training.batch_size)
        symmetric_batch: int = int(lens_cfg.symmetric_batch_size)
        num_replicas: int = self._num_replicas()
        per_task_batch: List[int] = []
        name: str
        for name in self.raw_dataset.meta.data_names:
            if "symmetric" in name:
                per_task_batch.append(max(symmetric_batch // num_replicas, 1))
            else:
                per_task_batch.append(batch_size)

        sampler = SameDatasetBatchSampler(
            row_ranges=self.raw_dataset.meta.row_ranges,
            per_task_batch_size=per_task_batch,
            seed=int(self.cfg.seed) if "seed" in self.cfg else 0,
            drop_last=True,
        )
        collator = LENSCollator(
            tokenizer=self.tokenizer,
            train_group_size=int(lens_cfg.train_group_size),
            symmetric_train_group_size=int(lens_cfg.symmetric_train_group_size),
            max_class_neg=int(lens_cfg.max_class_neg),
            query_max_len=int(lens_cfg.query_max_len),
            passage_max_len=int(lens_cfg.passage_max_len),
            sub_batch_size=int(lens_cfg.sub_batch_size),
            use_special_tokens=bool(train_cfg.get("use_special_tokens", True)),
            pad_to_multiple_of=int(lens_cfg.get("pad_to_multiple_of", 8)),
        )
        num_workers: int = int(self.cfg.training.num_workers)
        dl_kwargs: dict[str, Any] = {
            "batch_sampler": sampler,
            "collate_fn": collator,
            "num_workers": num_workers,
            "pin_memory": not bool(self.cfg.training.use_cpu),
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = int(self.cfg.training.prefetch_factor)
        return DataLoader(self.train_dataset, **dl_kwargs)

    def val_dataloader(self) -> DataLoader[Any]:
        """Single-batch placeholder loader.

        Lightning requires a val_dataloader to fire validation hooks; the actual
        evaluation work happens in
        ``LENSTrainingModule.on_validation_epoch_end`` (calls MTEB on the
        live model). The "batch" here is unused -- we just need Lightning to
        invoke the hook.
        """
        from torch.utils.data import TensorDataset
        import torch as _torch
        dummy = TensorDataset(_torch.zeros(1, dtype=_torch.long))
        return DataLoader(dummy, batch_size=1, num_workers=0)

    # --- Helpers -------------------------------------------------------------

    @staticmethod
    def _num_replicas() -> int:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
        return 1


__all__ = ["LENSTrainDataModule"]
