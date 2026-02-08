"""Dataset configuration classes for SPLADE data pipelines."""

from src.data.dataset.base import BaseDataset
from src.data.dataset.beir import BEIRDataset
from src.data.dataset.msmarco import MSMARCODataset
from src.data.dataset.msmarco_distill_scores import MSMARCODistillScoresDataset
from src.data.dataset.msmarco_triplet_scores import MSMARCOTripletScoresDataset

__all__ = [
    "BaseDataset",
    "BEIRDataset",
    "MSMARCODataset",
    "MSMARCODistillScoresDataset",
    "MSMARCOTripletScoresDataset",
]
