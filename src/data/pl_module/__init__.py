"""Lightning data modules for SPLADE."""

from importlib import import_module
from typing import Any

__all__: list[str] = [
    "EncodeDataModule",
    "ScoringDataModule",
    "ScoringHardNegativesDataModule",
    "RetrievalSpeedDataModule",
    "RerankingDataModule",
    "RetrievalDataModule",
    "TrainDataModule",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EncodeDataModule": ("src.data.pl_module.encode", "EncodeDataModule"),
    "ScoringDataModule": ("src.data.pl_module.scoring", "ScoringDataModule"),
    "ScoringHardNegativesDataModule": (
        "src.data.pl_module.scoring_hard_negatives",
        "ScoringHardNegativesDataModule",
    ),
    "RetrievalSpeedDataModule": (
        "src.data.pl_module.speed",
        "RetrievalSpeedDataModule",
    ),
    "RerankingDataModule": ("src.data.pl_module.reranking", "RerankingDataModule"),
    "RetrievalDataModule": ("src.data.pl_module.retrieval", "RetrievalDataModule"),
    "TrainDataModule": ("src.data.pl_module.train", "TrainDataModule"),
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
