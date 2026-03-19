"""PyTorch dataset modules for SPLADE."""

from importlib import import_module
from typing import Any

__all__: list[str] = [
    "EncodePDModule",
    "PDModule",
    "RerankingPDModule",
    "RetrievalPDModule",
    "ScoringPDModule",
    "HardNegativesScoringPDModule",
    "TrainingPDModule",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "PDModule": ("src.data.pd_module.base", "PDModule"),
    "EncodePDModule": ("src.data.pd_module.encode", "EncodePDModule"),
    "RerankingPDModule": ("src.data.pd_module.reranking", "RerankingPDModule"),
    "RetrievalPDModule": ("src.data.pd_module.retrieval", "RetrievalPDModule"),
    "ScoringPDModule": ("src.data.pd_module.scoring", "ScoringPDModule"),
    "HardNegativesScoringPDModule": (
        "src.data.pd_module.scoring_hard_negatives",
        "HardNegativesScoringPDModule",
    ),
    "TrainingPDModule": ("src.data.pd_module.train", "TrainingPDModule"),
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
