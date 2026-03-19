import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.data.pl_module.common import (
    build_inference_dataloader,
    build_model_tokenizer,
)
from src.data.pd_module import RetrievalPDModule


class RetrievalSpeedDataModule(L.LightningDataModule):
    """LightningDataModule for SPLADE speed benchmarking."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.tokenizer: PreTrainedTokenizerBase = build_model_tokenizer(self.cfg.model)
        self._dataset: RetrievalPDModule | None = None

    # --- Property methods ---
    @property
    def dataset(self) -> RetrievalPDModule:
        if self._dataset is None:
            self._dataset = RetrievalPDModule(
                cfg=self.cfg.dataset,
                tokenizer=self.tokenizer,
                model_cfg=self.cfg.model,
                seed=int(self.cfg.seed),
                load_teacher_scores=False,
                require_teacher_scores=False,
            )
        return self._dataset

    # --- Public methods ---
    def prepare_data(self) -> None:
        self.dataset.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.dataset.setup()

    def test_dataloader(self) -> list[DataLoader]:
        per_query_batch_size: int = 1
        batch_sizes = [int(value) for value in self.cfg.speed.batch_sizes]
        if not batch_sizes:
            raise ValueError("speed.batch_sizes must contain at least one value.")
        if any(value <= 0 for value in batch_sizes):
            raise ValueError("speed.batch_sizes must be positive integers.")
        batch_size: int = batch_sizes[0]
        return [
            self._build_dataloader(per_query_batch_size),
            self._build_dataloader(batch_size),
        ]

    # --- Protected methods ---
    def _build_dataloader(self, batch_size: int) -> DataLoader:
        return build_inference_dataloader(
            dataset=self.dataset,
            batch_size=int(batch_size),
            num_workers=int(self.cfg.testing.num_workers),
            collate_fn=self.dataset.collator,
            use_cpu=bool(self.cfg.testing.use_cpu),
            shuffle=False,
            drop_last=False,
            distributed_shuffle=False,
        )
