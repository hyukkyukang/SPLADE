"""
Retriever models package.

To avoid circular imports, retriever classes are not imported at module level.
Import them explicitly when needed:
    from src.model.retriever.base import BaseRetriever
    from src.model.retriever.registry import RETRIEVER_REGISTRY
"""

from importlib import import_module
from typing import Any

__all__ = [
    "RETRIEVER_REGISTRY",
    "BaseRetriever",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseRetriever": ("src.model.retriever.base", "BaseRetriever"),
    "RETRIEVER_REGISTRY": ("src.model.retriever.registry", "RETRIEVER_REGISTRY"),
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
