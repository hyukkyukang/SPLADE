"""Dataset configuration classes for SPLADE data pipelines."""

from src.data.dataset.base import BaseDataset
from src.data.dataset.beir import BEIRDataset
from src.data.dataset.msmarco import MSMARCODataset

__all__ = [
    "BaseDataset",
    "BEIRDataset",
    "MSMARCODataset",
]
