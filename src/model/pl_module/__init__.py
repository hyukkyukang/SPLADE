"""Lightning model modules for SPLADE."""

from importlib import import_module
from typing import Any

__all__: list[str] = [
    "DenseEncodeModule",
    "DenseRetrievalEvalLightningModule",
    "SPLADEEncodeModule",
    "RetrievalEvalLightningModule",
    "RetrievalSearchLightningModule",
    "RetrievalSpeedLightningModule",
    "RerankingLightningModule",
    "CrossEncoderScoringModule",
    "SPLADETrainingModule",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DenseEncodeModule": ("src.model.pl_module.dense_encode", "DenseEncodeModule"),
    "DenseRetrievalEvalLightningModule": (
        "src.model.pl_module.dense_eval",
        "DenseRetrievalEvalLightningModule",
    ),
    "SPLADEEncodeModule": ("src.model.pl_module.encode", "SPLADEEncodeModule"),
    "RetrievalEvalLightningModule": (
        "src.model.pl_module.eval",
        "RetrievalEvalLightningModule",
    ),
    "RetrievalSearchLightningModule": (
        "src.model.pl_module.search",
        "RetrievalSearchLightningModule",
    ),
    "RetrievalSpeedLightningModule": (
        "src.model.pl_module.speed",
        "RetrievalSpeedLightningModule",
    ),
    "RerankingLightningModule": (
        "src.model.pl_module.reranking",
        "RerankingLightningModule",
    ),
    "CrossEncoderScoringModule": (
        "src.model.pl_module.scoring",
        "CrossEncoderScoringModule",
    ),
    "SPLADETrainingModule": (
        "src.model.pl_module.train",
        "SPLADETrainingModule",
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
