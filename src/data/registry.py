from typing import Callable

from omegaconf import DictConfig

from src.data.dataset import (
    BaseDataset,
    BEIRDataset,
    CorpusOnlyDataset,
    MSMARCODevSmallNegativesDataset,
    MSMARCODataset,
    MSMARCODistillScoresDataset,
    MSMARCOHardNegativesDataset,
    MSMARCOTripletScoresDataset,
    Patent10KHardNegativesDataset,
    PatentUsInBatchDataset,
)

DatasetBuilder = Callable[[DictConfig], BaseDataset]


_DATASET_BUILDERS: dict[str, DatasetBuilder] = {
    "corpus_only": CorpusOnlyDataset,
    "msmarco": MSMARCODataset,
    "msmarco_local_triplets": MSMARCODataset,
    "msmarco_hard_negatives": MSMARCOHardNegativesDataset,
    "msmarco_dev_small_negatives": MSMARCODevSmallNegativesDataset,
    "msmarco_distill_scores": MSMARCODistillScoresDataset,
    "msmarco_triplet_scores": MSMARCOTripletScoresDataset,
    "patent_10k_hard_negatives": Patent10KHardNegativesDataset,
    "patent_us_in_batch": PatentUsInBatchDataset,
    "beir": BEIRDataset,
}


def resolve_dataset_builder(dataset_cfg: DictConfig) -> DatasetBuilder:
    """Resolve a dataset builder from the dataset config."""
    dataset_type: str = str(dataset_cfg.get("type") or dataset_cfg.get("name"))
    builder: DatasetBuilder | None = _DATASET_BUILDERS.get(dataset_type)
    if builder is None:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return builder


def build_dataset(dataset_cfg: DictConfig) -> BaseDataset:
    """Instantiate a BaseDataset from config."""
    builder: DatasetBuilder = resolve_dataset_builder(dataset_cfg)
    return builder(cfg=dataset_cfg)
