from importlib import import_module
from typing import Any

__all__ = [
    "get_rank",
    "get_world_size",
    "is_rank_zero",
    "log_if_rank_zero",
    "maybe_barrier",
    "set_seed",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "get_rank": ("src.utils.dist", "get_rank"),
    "get_world_size": ("src.utils.dist", "get_world_size"),
    "is_rank_zero": ("src.utils.dist", "is_rank_zero"),
    "maybe_barrier": ("src.utils.dist", "maybe_barrier"),
    "log_if_rank_zero": ("src.utils.logging", "log_if_rank_zero"),
    "set_seed": ("src.utils.seed", "set_seed"),
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
