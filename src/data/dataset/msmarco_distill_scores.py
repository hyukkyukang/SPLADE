from datasets import Dataset

from src.data.dataset import BaseDataset


class MSMARCODistillScoresDataset(BaseDataset):
    """MS MARCO distillation dataset using scored rows as metadata."""

    # --- Protected methods ---
    def _resolve_meta_dataset(self) -> Dataset:
        if self.hf_name is None:
            raise ValueError("hf_name must be set for scored datasets.")
        meta_dataset: Dataset = self._load_hf_dataset(
            hf_name=self.hf_name,
            hf_subset=self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
            data_files=self.hf_data_files,
        )
        return self._apply_hf_sample_window(meta_dataset)
