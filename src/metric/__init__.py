from importlib import import_module
from typing import Any

__all__: list[str] = ["RetrievalMetrics", "ValidationRetrievalMetrics"]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "RetrievalMetrics": ("src.metric.retrieval", "RetrievalMetrics"),
    "ValidationRetrievalMetrics": (
        "src.metric.validation_retrieval",
        "ValidationRetrievalMetrics",
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
