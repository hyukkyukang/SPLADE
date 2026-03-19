from importlib import import_module
from typing import Any

__all__: list[str] = [
    "EmbeddingGemmaLSRModel",
    "TextPair",
    "info_nce_in_batch",
    "flops_regularization",
    "compute_ranking_metrics",
    "resolve_boundary_token_ids",
    "discover_fragmented_terms",
    "build_semantic_projection_initialization",
    "apply_projection_initialization",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TextPair": ("src.prototype.embeddinggemma_lsr.data", "TextPair"),
    "compute_ranking_metrics": (
        "src.prototype.embeddinggemma_lsr.losses",
        "compute_ranking_metrics",
    ),
    "flops_regularization": (
        "src.prototype.embeddinggemma_lsr.losses",
        "flops_regularization",
    ),
    "info_nce_in_batch": (
        "src.prototype.embeddinggemma_lsr.losses",
        "info_nce_in_batch",
    ),
    "EmbeddingGemmaLSRModel": (
        "src.prototype.embeddinggemma_lsr.model",
        "EmbeddingGemmaLSRModel",
    ),
    "apply_projection_initialization": (
        "src.prototype.embeddinggemma_lsr.model",
        "apply_projection_initialization",
    ),
    "build_semantic_projection_initialization": (
        "src.prototype.embeddinggemma_lsr.model",
        "build_semantic_projection_initialization",
    ),
    "discover_fragmented_terms": (
        "src.prototype.embeddinggemma_lsr.model",
        "discover_fragmented_terms",
    ),
    "resolve_boundary_token_ids": (
        "src.prototype.embeddinggemma_lsr.model",
        "resolve_boundary_token_ids",
    ),
}


def __getattr__(name: str) -> Any:
    target: tuple[str, str] | None = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    module_name: str
    attr_name: str
    module_name, attr_name = target
    module = import_module(module_name)
    value: Any = getattr(module, attr_name)
    globals()[name] = value
    return value
