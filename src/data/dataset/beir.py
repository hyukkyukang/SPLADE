from datasets import Dataset
from omegaconf import DictConfig

from src.data.dataset import BaseDataset
from src.data.dataset.base import (
    CORPUS_ID_COLUMN_KEY,
    CORPUS_SPLIT_NAME_KEY,
    CORPUS_SUBSET_NAME_KEY,
    CORPUS_TEXT_COLUMN_KEY,
    CORPUS_TITLE_COLUMN_KEY,
    QUERY_ID_COLUMN_KEY,
    QUERY_SPLIT_NAME_KEY,
    QUERY_SUBSET_NAME_KEY,
    QUERY_TEXT_COLUMN_KEY,
)


class BEIRDataset(BaseDataset):
    """BEIR dataset configuration for query/corpus splits."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)

        split_name: str = str(cfg.split)
        corpus_split: str = str(cfg.corpus_split)

        # Column mapping for query dataset fields.
        self.query_column_names: dict[str, str] = {
            QUERY_SUBSET_NAME_KEY: "queries",
            QUERY_SPLIT_NAME_KEY: split_name,
            QUERY_ID_COLUMN_KEY: "_id",
            QUERY_TEXT_COLUMN_KEY: "text",
        }

        # Column mapping for corpus dataset fields.
        self.corpus_column_names: dict[str, str] = {
            CORPUS_SUBSET_NAME_KEY: "corpus",
            CORPUS_SPLIT_NAME_KEY: corpus_split,
            CORPUS_ID_COLUMN_KEY: "_id",
            CORPUS_TEXT_COLUMN_KEY: "text",
            CORPUS_TITLE_COLUMN_KEY: "title",
        }

    # --- Protected methods ---
    def _resolve_meta_dataset(self) -> Dataset:
        raise NotImplementedError("BEIRDataset does not provide training metadata.")
