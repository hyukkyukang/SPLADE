import os
from functools import cached_property
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase

from src.data.pl_module.common import build_model_tokenizer
from src.data.pd_module import TrainingPDModule

# Backward-compatible alias used by tests and external monkey patches.
build_tokenizer = build_model_tokenizer


class TrainDataModule(L.LightningDataModule):
    """LightningDataModule for SPLADE training/validation."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        # Run prepare_data once globally (rank 0) under DDP.
        self.prepare_data_per_node = False
        self.tokenizer: PreTrainedTokenizerBase = build_tokenizer(self.cfg.model)

    # --- Property methods ---
    @cached_property
    def train_dataset(self) -> TrainingPDModule:
        return self._build_dataset(
            self.cfg.train_dataset,
            load_teacher_scores=None,
            require_teacher_scores=None,
            cache_namespace="train",
        )

    @cached_property
    def val_dataset(self) -> TrainingPDModule:
        return self._build_dataset(
            self.cfg.val_dataset,
            load_teacher_scores=False,
            require_teacher_scores=False,
            cache_namespace="val",
        )

    # --- Protected methods ---
    def _validation_enabled(self) -> bool:
        raw_value: Any = self.cfg.training.get("limit_val_batches", 1.0)
        try:
            return float(raw_value) > 0.0
        except (TypeError, ValueError):
            return bool(raw_value)

    def _build_dataset(
        self,
        cfg: DictConfig,
        load_teacher_scores: bool | None,
        require_teacher_scores: bool | None,
        cache_namespace: str | None = None,
    ) -> TrainingPDModule:
        uses_hf: bool = cfg.hf_name is not None
        if not uses_hf:
            dataset_name: str = str(cfg.name)
            if dataset_name != "msmarco_local_triplets":
                raise ValueError(
                    "Local dataset files are only supported for msmarco_local_triplets. "
                    "Please set dataset.hf_name for HuggingFace datasets."
                )
        distill_cfg: DictConfig = self.cfg.training.distill
        resolved_load: bool = (
            bool(distill_cfg.enabled)
            if load_teacher_scores is None
            else bool(load_teacher_scores)
        )
        resolved_require: bool = (
            bool(resolved_load and distill_cfg.fail_on_missing)
            if require_teacher_scores is None
            else bool(require_teacher_scores)
        )
        return TrainingPDModule(
            cfg=cfg,
            tokenizer=self.tokenizer,
            model_cfg=self.cfg.model,
            seed=int(self.cfg.seed),
            load_teacher_scores=resolved_load,
            require_teacher_scores=resolved_require,
            cache_namespace=cache_namespace,
        )

    def _build_sampler(
        self,
        dataset: TrainingPDModule,
        shuffle: bool,
        drop_last: bool,
    ) -> DistributedSampler | None:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return None
        return DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)

    def _make_dataloader(
        self,
        dataset: TrainingPDModule,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
    ) -> DataLoader:
        num_workers: int = int(self.cfg.training.num_workers)
        pin_memory: bool = not bool(self.cfg.training.use_cpu)
        persistent_workers: bool = num_workers > 0
        prefetch_factor: int | None = None
        if num_workers > 0:
            # Use a small prefetch to overlap CPU preprocessing and GPU work.
            prefetch_factor = int(self.cfg.training.prefetch_factor)
        sampler: DistributedSampler | None = self._build_sampler(
            dataset=dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        use_shuffle: bool = shuffle and sampler is None
        dataloader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": int(batch_size),
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
        if num_workers > 0:
            raw_context_value: Any | None = self.cfg.training.get(
                "dataloader_multiprocessing_context"
            )
            context_value: str | None = None
            if raw_context_value is not None:
                normalized_context: str = str(raw_context_value).strip().lower()
                if normalized_context and normalized_context not in {"none", "null"}:
                    context_value = normalized_context
            elif os.name == "posix":
                # CUDA training must avoid fork: worker processes can inherit the
                # parent rank's CUDA context and show up as extra GPU consumers.
                if pin_memory:
                    context_value = "spawn"
                else:
                    # For CPU-only runs, keep fork for lower startup overhead and
                    # better page-cache sharing across workers.
                    context_value = "fork"
            if context_value is not None:
                dataloader_kwargs["multiprocessing_context"] = context_value
        return DataLoader(**dataloader_kwargs)

    # --- Public methods ---
    def prepare_data(self) -> None:
        train_dataset: TrainingPDModule = self.train_dataset
        train_dataset.prepare_data()
        if self._validation_enabled():
            val_dataset: TrainingPDModule = self.val_dataset
            val_dataset.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.train_dataset.setup()
        if self._validation_enabled():
            self.val_dataset.setup()

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(
            dataset=self.train_dataset,
            batch_size=int(self.cfg.training.batch_size),
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        if not self._validation_enabled():
            return []
        return self._make_dataloader(
            dataset=self.val_dataset,
            batch_size=int(self.cfg.training.eval_batch_size),
            shuffle=False,
            drop_last=False,
        )
