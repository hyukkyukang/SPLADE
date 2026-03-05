"""Dataset configuration classes for SPLADE data pipelines."""

from src.data.dataset.base import BaseDataset
from src.data.dataset.beir import BEIRDataset
from src.data.dataset.msmarco_dev_small_negatives import (
    MSMARCODevSmallNegativesDataset,
)
from src.data.dataset.msmarco import MSMARCODataset
from src.data.dataset.msmarco_hard_negatives import MSMARCOHardNegativesDataset
from src.data.dataset.msmarco_distill_scores import MSMARCODistillScoresDataset
from src.data.dataset.msmarco_triplet_scores import MSMARCOTripletScoresDataset

__all__ = [
    "BaseDataset",
    "BEIRDataset",
    "MSMARCODevSmallNegativesDataset",
    "MSMARCODataset",
    "MSMARCOHardNegativesDataset",
    "MSMARCODistillScoresDataset",
    "MSMARCOTripletScoresDataset",
]
