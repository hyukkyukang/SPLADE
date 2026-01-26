from __future__ import annotations

from functools import cached_property

import lightning as L
from torch.utils.data import DataLoader

from src.dataset.registry import DATASET_REGISTRY
from src.tokenization.tokenizer import build_tokenizer


class TrainDataModule(L.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = build_tokenizer(cfg.model.huggingface_name)

    @cached_property
    def train_dataset(self):
        require_teacher_scores = bool(
            self.cfg.training.distill.enabled
            and self.cfg.training.distill.fail_on_missing
        )
        return self._build_dataset(
            self.cfg.train_dataset, require_teacher_scores=require_teacher_scores
        )

    @cached_property
    def val_dataset(self):
        return self._build_dataset(self.cfg.val_dataset, require_teacher_scores=False)

    def prepare_data(self) -> None:
        self.train_dataset.prepare_data()
        self.val_dataset.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset.setup()
        self.val_dataset.setup()

    def _build_dataset(self, cfg, require_teacher_scores: bool | None):
        if not getattr(cfg, "use_hf", False):
            raise ValueError(
                "Local dataset files are no longer supported. "
                "Please use HuggingFace datasets with dataset.use_hf=true."
            )
        load_teacher_scores = bool(self.cfg.training.distill.enabled)
        dataset_builder = DATASET_REGISTRY[cfg.name]
        return dataset_builder(
            cfg=cfg,
            global_cfg=self.cfg,
            tokenizer=self.tokenizer,
            load_teacher_scores=load_teacher_scores,
            require_teacher_scores=require_teacher_scores,
        )

    def train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=not getattr(dataset, "is_streaming", False),
            num_workers=self.cfg.training.num_workers,
            collate_fn=dataset.collator,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self.val_dataset
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            collate_fn=dataset.collator,
        )
