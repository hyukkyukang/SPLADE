"""Dataset configuration classes for SPLADE data pipelines."""

from importlib import import_module
from typing import Any

__all__ = [
    "BaseDataset",
    "BEIRDataset",
    "CorpusOnlyDataset",
    "MSMARCODevSmallNegativesDataset",
    "MSMARCODataset",
    "MSMARCOHardNegativesDataset",
    "MSMARCODistillScoresDataset",
    "MSMARCOTripletScoresDataset",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseDataset": ("src.data.dataset.base", "BaseDataset"),
    "BEIRDataset": ("src.data.dataset.beir", "BEIRDataset"),
    "CorpusOnlyDataset": ("src.data.dataset.corpus_only", "CorpusOnlyDataset"),
    "MSMARCODevSmallNegativesDataset": (
        "src.data.dataset.msmarco_dev_small_negatives",
        "MSMARCODevSmallNegativesDataset",
    ),
    "MSMARCODataset": ("src.data.dataset.msmarco", "MSMARCODataset"),
    "MSMARCOHardNegativesDataset": (
        "src.data.dataset.msmarco_hard_negatives",
        "MSMARCOHardNegativesDataset",
    ),
    "MSMARCODistillScoresDataset": (
        "src.data.dataset.msmarco_distill_scores",
        "MSMARCODistillScoresDataset",
    ),
    "MSMARCOTripletScoresDataset": (
        "src.data.dataset.msmarco_triplet_scores",
        "MSMARCOTripletScoresDataset",
    ),
}


def __getattr__(name: str) -> Any:
    target: tuple[str, str] | None = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name: str
    attr_name: str
    module_name, attr_name = target
    module = import_module(module_name)
    value: Any = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
