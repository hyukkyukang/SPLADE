from typing import Callable

from omegaconf import DictConfig

from src.data.dataset import BaseDataset, BEIRDataset, MSMARCODataset

DatasetBuilder = Callable[[DictConfig], BaseDataset]


_DATASET_BUILDERS: dict[str, DatasetBuilder] = {
    "msmarco": MSMARCODataset,
    "msmarco_local_triplets": MSMARCODataset,
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
