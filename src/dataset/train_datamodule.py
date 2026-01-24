from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader

from src.dataset.collators import TrainCollator
from src.dataset.datasets.train import TrainDataset
from src.dataset.datasets.train_hf import HFMSMarcoTrainDataset
from src.tokenization.tokenizer import build_tokenizer


class TrainDataModule(L.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = build_tokenizer(cfg.model.huggingface_name)

    def setup(self, stage: str | None = None) -> None:
        train_cfg = self.cfg.train_dataset
        val_cfg = self.cfg.val_dataset

        require_teacher = self.cfg.training.distill.enabled and self.cfg.training.distill.fail_on_missing
        teacher_key = self.cfg.training.distill.teacher_score_key
        self.train_dataset = self._build_dataset(
            train_cfg,
            require_teacher=require_teacher,
            teacher_key=teacher_key,
            fallback_path=train_cfg.train_path,
        )

        self.val_dataset = self._build_dataset(
            val_cfg,
            require_teacher=False,
            teacher_key=teacher_key,
            fallback_path=val_cfg.train_path
            if hasattr(val_cfg, "train_path") and val_cfg.train_path
            else train_cfg.train_path,
        )

        self.train_collator = TrainCollator(
            tokenizer=self.tokenizer,
            max_query_length=train_cfg.max_query_length,
            max_doc_length=train_cfg.max_doc_length,
            require_teacher_scores=require_teacher,
        )
        self.val_collator = TrainCollator(
            tokenizer=self.tokenizer,
            max_query_length=val_cfg.max_query_length
            if hasattr(val_cfg, "max_query_length")
            else train_cfg.max_query_length,
            max_doc_length=val_cfg.max_doc_length
            if hasattr(val_cfg, "max_doc_length")
            else train_cfg.max_doc_length,
            require_teacher_scores=False,
        )

    def _build_dataset(self, cfg, require_teacher: bool, teacher_key: str, fallback_path: str):
        if getattr(cfg, "use_hf", False):
            return HFMSMarcoTrainDataset(
                hf_name=cfg.hf_name,
                hf_subset=cfg.hf_subset,
                hf_split=cfg.hf_split,
                num_positives=cfg.num_positives,
                num_negatives=cfg.num_negatives,
                require_teacher_scores=require_teacher,
                teacher_score_key=teacher_key,
                hf_text_name=cfg.hf_text_name,
                hf_teacher_name=cfg.hf_teacher_name,
                hf_teacher_subset=cfg.hf_teacher_subset,
                hf_teacher_split=cfg.hf_teacher_split,
                hf_cache_dir=cfg.hf_cache_dir,
                hf_max_samples=cfg.hf_max_samples,
                hf_teacher_cache_dir=cfg.hf_teacher_cache_dir,
                hf_teacher_max_samples=cfg.hf_teacher_max_samples,
            )
        return TrainDataset(
            train_path=fallback_path,
            num_positives=cfg.num_positives,
            num_negatives=cfg.num_negatives,
            require_teacher_scores=require_teacher,
            teacher_score_key=teacher_key,
            teacher_scores_path=cfg.teacher_scores_path
            if hasattr(cfg, "teacher_scores_path")
            else None,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            collate_fn=self.train_collator,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            collate_fn=self.val_collator,
        )
