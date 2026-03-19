from importlib import import_module
from typing import Any

__all__ = ["AnnaTokenizer", "AnnaTokenizerFast"]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AnnaTokenizer": ("src.tokenization.anna_tokenizer", "AnnaTokenizer"),
    "AnnaTokenizerFast": ("src.tokenization.anna_tokenizer", "AnnaTokenizerFast"),
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
