"""Search-related utilities for index-based retrieval."""

from importlib import import_module
from typing import Any

__all__ = [
    "IndexedRetrievalHelper",
    "InvertedIndex",
    "load_inverted_index",
    "prepare_score_buffers",
    "resolve_query_sparsify_config",
    "score_query_postings",
    "score_query_postings_bmw",
    "score_query_postings_wand",
    "sparsify_batch_gpu_csr",
    "sparsify_query_vector",
    "sparsify_vector_gpu",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "prepare_score_buffers": ("src.search.buffers", "prepare_score_buffers"),
    "resolve_query_sparsify_config": (
        "src.search.buffers",
        "resolve_query_sparsify_config",
    ),
    "InvertedIndex": ("src.search.index", "InvertedIndex"),
    "load_inverted_index": ("src.search.index", "load_inverted_index"),
    "IndexedRetrievalHelper": ("src.search.retrieval", "IndexedRetrievalHelper"),
    "score_query_postings": ("src.search.scoring", "score_query_postings"),
    "score_query_postings_bmw": ("src.search.scoring", "score_query_postings_bmw"),
    "score_query_postings_wand": ("src.search.scoring", "score_query_postings_wand"),
    "sparsify_batch_gpu_csr": ("src.search.sparsify", "sparsify_batch_gpu_csr"),
    "sparsify_query_vector": ("src.search.sparsify", "sparsify_query_vector"),
    "sparsify_vector_gpu": ("src.search.sparsify", "sparsify_vector_gpu"),
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
    return sorted(list(globals().keys()) + __all__)
