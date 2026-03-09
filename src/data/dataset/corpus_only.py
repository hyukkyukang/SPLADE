from functools import cached_property

from datasets import Dataset

from src.data.dataset.base import BaseDataset
from src.data.dataset.parquet_view import ProjectedParquetDataset


class CorpusOnlyDataset(BaseDataset):
    """Dataset implementation for corpus-only Hugging Face/parquet encoding."""

    @cached_property
    def corpus_dataset(self) -> Dataset | ProjectedParquetDataset:
        """Get the corpus dataset, using direct parquet access when possible."""
        if (
            self.query_corpus_hf_name == "parquet"
            and self.query_corpus_hf_data_files is not None
        ):
            return ProjectedParquetDataset(
                data_files=self.query_corpus_hf_data_files,
                split=self.corpus_column_names["corpus_split_name"],
                columns=self.required_corpus_columns,
            )
        return super().corpus_dataset

    # --- Protected methods ---
    def _resolve_meta_dataset(self) -> Dataset:
        raise NotImplementedError(
            "CorpusOnlyDataset does not provide training metadata."
        )
