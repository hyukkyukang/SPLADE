import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.data.pl_module.common import (
    build_inference_dataloader,
    build_model_tokenizer,
)
from src.data.pd_module import RerankingPDModule


class RerankingDataModule(L.LightningDataModule):
    """LightningDataModule for reranking datasets."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.tokenizer: PreTrainedTokenizerBase = build_model_tokenizer(self.cfg.model)
        self._dataset: RerankingPDModule | None = None

    # --- Property methods ---
    @property
    def dataset(self) -> RerankingPDModule:
        if self._dataset is None:
            self._dataset = RerankingPDModule(
                cfg=self.cfg.dataset,
                tokenizer=self.tokenizer,
                model_cfg=self.cfg.model,
                seed=int(self.cfg.seed),
                load_teacher_scores=False,
                require_teacher_scores=False,
            )
        return self._dataset

    # --- Protected methods ---
    def _build_dataloader(self, *, shuffle: bool) -> DataLoader:
        return build_inference_dataloader(
            dataset=self.dataset,
            batch_size=int(self.cfg.testing.batch_size),
            num_workers=int(self.cfg.testing.num_workers),
            collate_fn=self.dataset.collator,
            use_cpu=bool(self.cfg.testing.use_cpu),
            shuffle=bool(shuffle),
            drop_last=False,
            distributed_shuffle=bool(shuffle),
        )

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
