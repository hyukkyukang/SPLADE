"""Index utilities package."""

from importlib import import_module
from typing import Any

__all__ = [
    "AsyncSparseWriter",
    "SparseShardWriter",
    "SparseWriterConfig",
    "build_inverted_index_from_shards",
    "load_shard_manifest",
    "resolve_numpy_dtype",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AsyncSparseWriter": ("src.index.async_writer", "AsyncSparseWriter"),
    "SparseWriterConfig": ("src.index.async_writer", "SparseWriterConfig"),
    "SparseShardWriter": ("src.index.sparse", "SparseShardWriter"),
    "build_inverted_index_from_shards": (
        "src.index.sparse",
        "build_inverted_index_from_shards",
    ),
    "load_shard_manifest": ("src.index.sparse", "load_shard_manifest"),
    "resolve_numpy_dtype": ("src.index.sparse", "resolve_numpy_dtype"),
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
