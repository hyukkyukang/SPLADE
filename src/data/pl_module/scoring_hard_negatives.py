from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Sampler

from src.data.pd_module.scoring_hard_negatives import HardNegativesScoringPDModule
from src.data.pl_module.scoring import ScoringCollator
from src.utils.script_setup import normalize_optional_str
from src.data.pd_module.scoring import _resolve_local_files_only
from src.data.sampler import NonPaddingDistributedSampler


class ScoringHardNegativesDataModule(L.LightningDataModule):
    """LightningDataModule for hard-negatives scoring."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self._dataset: HardNegativesScoringPDModule | None = None

    @property
    def dataset(self) -> HardNegativesScoringPDModule:
        if self._dataset is None:
            self._dataset = HardNegativesScoringPDModule(
                score_dataset_cfg=self.cfg.score_dataset,
                scoring_cfg=self.cfg.scoring,
                hard_negatives_cfg=self.cfg.hard_negatives,
            )
        return self._dataset

    def prepare_data(self) -> None:
        self.dataset.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.dataset.setup()

    def predict_dataloader(self) -> DataLoader:
        scoring_cfg: DictConfig = self.cfg.scoring
        num_workers: int = int(scoring_cfg.num_workers)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_primary = int(torch.distributed.get_rank()) == 0
        else:
            is_primary = True
        local_files_only = _resolve_local_files_only(
            scoring_cfg, is_primary=is_primary
        )
        collator = ScoringCollator(
            model_name=str(scoring_cfg.model_name),
            tokenizer_name=normalize_optional_str(scoring_cfg.tokenizer_name),
            max_length=int(scoring_cfg.max_length),
            tokenize_chunk_size=int(scoring_cfg.tokenize_chunk_size),
            local_files_only=local_files_only,
        )
        sampler: Sampler[int] | None = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = NonPaddingDistributedSampler(self.dataset, shuffle=False)
        prefetch_factor: int | None = None
        if num_workers > 0:
            prefetch_factor = int(scoring_cfg.prefetch_factor)
        batch_size: int = int(scoring_cfg.row_batch_size)
        if batch_size <= 0:
            raise ValueError("scoring.row_batch_size must be a positive integer.")
        use_cpu: bool = bool(scoring_cfg.use_cpu)
        pin_memory: bool = not use_cpu
        persistent_workers: bool = num_workers > 0
        dataloader_kwargs: dict[str, Any] = {
            "dataset": self.dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collator,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
        if sampler is not None:
            dataloader_kwargs["sampler"] = sampler
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(**dataloader_kwargs)
