from __future__ import annotations

from functools import cached_property
from typing import Any

import lightning as L
import torch
from datasets import IterableDataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase

from src.data.registry import DATASET_REGISTRY
from src.utils.transformers import build_tokenizer


class TrainDataModule(L.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # Build a shared tokenizer for dataset construction.
        self.tokenizer: PreTrainedTokenizerBase = build_tokenizer(
            cfg.model.huggingface_name
        )

    @cached_property
    def train_dataset(self):
        require_teacher_scores = bool(
            self.cfg.training.distill.enabled
            and self.cfg.training.distill.fail_on_missing
        )
        return self._build_dataset(
            self.cfg.train_dataset,
            require_teacher_scores=require_teacher_scores,
            load_teacher_scores=bool(self.cfg.training.distill.enabled),
        )

    @cached_property
    def val_dataset(self):
        return self._build_dataset(
            self.cfg.val_dataset,
            require_teacher_scores=False,
            load_teacher_scores=False,
        )

    def prepare_data(self) -> None:
        train_dataset: Any = self.train_dataset
        prepare_integer_ids: Any = getattr(
            train_dataset, "prepare_integer_id_cache", None
        )
        if callable(prepare_integer_ids):
            # Precompute integer IDs once on the main process.
            prepare_integer_ids()
        val_dataset: Any = self.val_dataset
        prepare_val_integer_ids: Any = getattr(
            val_dataset, "prepare_integer_id_cache", None
        )
        if callable(prepare_val_integer_ids):
            prepare_val_integer_ids()
        train_dataset.prepare_data()
        self.val_dataset.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset.setup()
        self.val_dataset.setup()

    def _build_dataset(
        self,
        cfg: DictConfig,
        require_teacher_scores: bool | None,
        load_teacher_scores: bool | None = None,
    ):
        if not cfg.use_hf:
            dataset_name: str = str(cfg.name)
            local_dataset_names: set[str] = {"msmarco_local_triplets"}
            if dataset_name not in local_dataset_names:
                raise ValueError(
                    "Local dataset files are only supported for "
                    f"{sorted(local_dataset_names)}. "
                    "Please use HuggingFace datasets with dataset.use_hf=true."
                )
        if load_teacher_scores is None:
            load_teacher_scores = bool(self.cfg.training.distill.enabled)
        dataset_builder = DATASET_REGISTRY[cfg.name]
        return dataset_builder(
            cfg=cfg,
            global_cfg=self.cfg,
            tokenizer=self.tokenizer,
            load_teacher_scores=load_teacher_scores,
            require_teacher_scores=require_teacher_scores,
        )

    def _build_sampler(
        self,
        dataset: Any,
        shuffle: bool,
        drop_last: bool,
    ) -> DistributedSampler | None:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return None
        inner_dataset: Any = getattr(dataset, "dataset", None)
        if isinstance(inner_dataset, IterableDataset):
            return None
        return DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)

    def _make_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
    ) -> DataLoader:
        num_workers: int = self.cfg.training.num_workers
        pin_memory: bool = not bool(self.cfg.training.use_cpu)
        persistent_workers: bool = num_workers > 0
        prefetch_factor: int | None = None
        if num_workers > 0:
            # Use a small prefetch to overlap CPU preprocessing and GPU work.
            prefetch_factor = self.cfg.training.prefetch_factor
        sampler: DistributedSampler | None = self._build_sampler(
            dataset=dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        inner_dataset: Any = getattr(dataset, "dataset", None)
        is_iterable: bool = isinstance(dataset, IterableDataset) or isinstance(
            inner_dataset, IterableDataset
        )
        use_shuffle: bool = shuffle and sampler is None and not is_iterable

        dataloader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": use_shuffle,
            "num_workers": num_workers,
            "collate_fn": dataset.collator,
            "drop_last": drop_last,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
        if sampler is not None:
            dataloader_kwargs["sampler"] = sampler
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(**dataloader_kwargs)

    def train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        return self._make_dataloader(
            dataset=dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self.val_dataset
        return self._make_dataloader(
            dataset=dataset,
            batch_size=self.cfg.training.eval_batch_size,
            shuffle=False,
            drop_last=False,
        )
