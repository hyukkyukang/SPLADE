from datasets import Dataset

from src.data.dataset import BaseDataset


class MSMARCODataset(BaseDataset):
    """MS MARCO dataset implementation.

    This dataset handles loading and processing of MS MARCO dataset,
    which is a large-scale dataset for information retrieval.

    MS MARCO dataset consists of:
    - Queries: Natural language questions or search queries
    - Corpus: Collection of documents/passages to search through
    - Qrels: Query-document relevance judgments (ground truth)
    """

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
