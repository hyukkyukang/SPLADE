from datasets import Dataset

from src.data.dataset import BaseDataset


class BEIRDataset(BaseDataset):
    """BEIR dataset configuration for query/corpus splits."""

    # --- Protected methods ---
    def _resolve_meta_dataset(self) -> Dataset:
        raise NotImplementedError("BEIRDataset does not provide training metadata.")
