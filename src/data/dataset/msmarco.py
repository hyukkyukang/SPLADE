from datasets import Dataset
from omegaconf import DictConfig

from src.data.dataset import BaseDataset
from src.data.dataset.base import (
    CORPUS_ID_COLUMN_KEY,
    CORPUS_SPLIT_NAME_KEY,
    CORPUS_SUBSET_NAME_KEY,
    CORPUS_TEXT_COLUMN_KEY,
    QUERY_ID_COLUMN_KEY,
    QUERY_SPLIT_NAME_KEY,
    QUERY_SUBSET_NAME_KEY,
    QUERY_TEXT_COLUMN_KEY,
)


class MSMARCODataset(BaseDataset):
    """MS MARCO dataset implementation.

    This dataset handles loading and processing of MS MARCO dataset,
    which is a large-scale dataset for information retrieval.

    MS MARCO dataset consists of:
    - Queries: Natural language questions or search queries
    - Corpus: Collection of documents/passages to search through
    - Qrels: Query-document relevance judgments (ground truth)
    """

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
        }

    # --- Protected methods ---
    def _resolve_meta_dataset(self) -> Dataset:
        if not self.use_hf:
            return self._load_local_triplets()
        if self.hf_name is None:
            raise ValueError("hf_name must be set for HuggingFace datasets.")
        meta_dataset: Dataset = self._load_hf_dataset(
            hf_name=self.hf_name,
            hf_subset=self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
            data_files=self.hf_data_files,
        )
        return self._apply_hf_sample_window(meta_dataset)
