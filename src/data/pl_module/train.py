from functools import cached_property
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase

from src.data.pd_module import TrainingPDModule
from src.utils.transformers import build_tokenizer


class TrainDataModule(L.LightningDataModule):
    """LightningDataModule for SPLADE training/validation."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.tokenizer: PreTrainedTokenizerBase = build_tokenizer(
            self.cfg.model.huggingface_name
        )

    # --- Property methods ---
    @cached_property
    def train_dataset(self) -> TrainingPDModule:
        return self._build_dataset(
            self.cfg.train_dataset,
            load_teacher_scores=None,
            require_teacher_scores=None,
        )

    @cached_property
    def val_dataset(self) -> TrainingPDModule:
        return self._build_dataset(
            self.cfg.val_dataset,
            load_teacher_scores=False,
            require_teacher_scores=False,
        )

    # --- Protected methods ---
    def _build_dataset(
        self,
        cfg: DictConfig,
        load_teacher_scores: bool | None,
        require_teacher_scores: bool | None,
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
            seed=int(self.cfg.seed),
            load_teacher_scores=resolved_load,
            require_teacher_scores=resolved_require,
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
        return DataLoader(**dataloader_kwargs)

    # --- Public methods ---
    def prepare_data(self) -> None:
        train_dataset: TrainingPDModule = self.train_dataset
        train_dataset.prepare_data()
        val_dataset: TrainingPDModule = self.val_dataset
        val_dataset.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.train_dataset.setup()
        self.val_dataset.setup()

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(
            dataset=self.train_dataset,
            batch_size=int(self.cfg.training.batch_size),
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(
            dataset=self.val_dataset,
            batch_size=int(self.cfg.training.eval_batch_size),
            shuffle=False,
            drop_last=False,
        )
