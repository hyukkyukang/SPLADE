import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.data.pl_module.common import (
    build_inference_dataloader,
    build_model_tokenizer,
)
from src.data.pd_module import EncodePDModule


class EncodeDataModule(L.LightningDataModule):
    """LightningDataModule for corpus encoding."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.tokenizer: PreTrainedTokenizerBase = build_model_tokenizer(self.cfg.model)
        self._dataset: EncodePDModule | None = None

    # --- Property methods ---
    @property
    def dataset(self) -> EncodePDModule:
        if self._dataset is None:
            self._dataset = EncodePDModule(
                cfg=self.cfg.dataset,
                encoding_cfg=self.cfg.encoding,
                tokenizer=self.tokenizer,
                model_cfg=self.cfg.model,
                seed=int(self.cfg.seed),
            )
        return self._dataset

    # --- Public methods ---
    def prepare_data(self) -> None:
        self.dataset.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.dataset.setup()

    def predict_dataloader(self) -> DataLoader:
        num_workers: int = int(self.cfg.encoding.num_workers)
        prefetch_factor: int | None = (
            int(self.cfg.encoding.prefetch_factor) if num_workers > 0 else None
        )
        return build_inference_dataloader(
            dataset=self.dataset,
            batch_size=int(self.cfg.encoding.batch_size),
            num_workers=num_workers,
            collate_fn=self.dataset.collator,
            use_cpu=bool(self.cfg.encoding.use_cpu),
            shuffle=False,
            drop_last=False,
            distributed_shuffle=False,
            prefetch_factor=prefetch_factor,
        )
