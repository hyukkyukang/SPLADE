from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase

from src.data.pd_module import RerankingPDModule
from src.utils.transformers import build_tokenizer


class RerankingDataModule(L.LightningDataModule):
    """LightningDataModule for reranking datasets."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.tokenizer: PreTrainedTokenizerBase = build_tokenizer(
            self.cfg.model.huggingface_name
        )
        self._dataset: RerankingPDModule | None = None

    # --- Property methods ---
    @property
    def dataset(self) -> RerankingPDModule:
        if self._dataset is None:
            self._dataset = RerankingPDModule(
                cfg=self.cfg.dataset,
                tokenizer=self.tokenizer,
                seed=int(self.cfg.seed),
                load_teacher_scores=False,
                require_teacher_scores=False,
            )
        return self._dataset

    # --- Protected methods ---
    def _build_dataloader(self, *, shuffle: bool) -> DataLoader:
        num_workers: int = int(self.cfg.testing.num_workers)
        sampler: DistributedSampler | None = (
            DistributedSampler(self.dataset, shuffle=shuffle)
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else None
        )
        dataloader_kwargs: dict[str, Any] = {
            "dataset": self.dataset,
            "batch_size": int(self.cfg.testing.batch_size),
            "num_workers": num_workers,
            "collate_fn": self.dataset.collator,
            "shuffle": shuffle if sampler is None else False,
            "drop_last": False,
            "pin_memory": not bool(self.cfg.testing.use_cpu),
        }
        if sampler is not None:
            dataloader_kwargs["sampler"] = sampler
        return DataLoader(**dataloader_kwargs)

    # --- Public methods ---
    def prepare_data(self) -> None:
        self.dataset.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.dataset.setup()

    def test_dataloader(self) -> DataLoader:
        return self._build_dataloader(shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._build_dataloader(shuffle=False)
