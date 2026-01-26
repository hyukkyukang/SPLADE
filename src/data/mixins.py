"""Mixins for dataset classes providing common functionality."""

from functools import cached_property
from typing import Set

from huggingface_hub import snapshot_download


class HuggingFaceDatasetMixin:
    """Mixin for datasets loaded from HuggingFace Hub.

    Provides common functionality for:
    - Computing all query IDs from the query dataset
    - Computing all document IDs from the corpus dataset
    - Downloading datasets from HuggingFace Hub

    Requires the following properties to be defined by the class using this mixin:
    - query_dataset: The query dataset
    - corpus_dataset: The corpus dataset
    - query_id_column: Column name for query IDs
    - corpus_id_column: Column name for corpus/document IDs
    """

    @cached_property
    def all_qids(self) -> Set[str]:
        """Get all query IDs in the dataset."""
        return set(self.query_dataset[self.query_id_column])

    @cached_property
    def all_doc_ids(self) -> Set[str]:
        """Get all document IDs in the corpus."""
        return set(self.corpus_dataset[self.corpus_id_column])

    def _download_from_hub(self, *repo_ids: str) -> None:
        """Download one or more datasets from HuggingFace Hub."""
        for repo_id in repo_ids:
            snapshot_download(repo_id=repo_id, repo_type="dataset")
