from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class Registry:
    """
    A generic registry to register and retrieve objects (classes, functions) by name.
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable[[T], T]:
        """
        Decorator to register an object.
        Usage:
            @REGISTRY.register("my_obj")
            class MyClass: ...
        """

        def _register(obj: T) -> T:
            key = name if name is not None else obj.__name__
            if key in self._registry:
                raise ValueError(f"'{key}' is already registered in {self._name}")
            self._registry[key] = obj
            return obj

        return _register

    def get(self, name: str) -> Any:
        """Retrieve an object by name."""
        if name not in self._registry:
            raise KeyError(
                f"'{name}' is not registered in {self._name}. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list_available(self) -> List[str]:
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def __or__(self, other: "Registry") -> "Registry":
        """Combine two registries."""
        if not isinstance(other, Registry):
            return NotImplemented

        new_registry = Registry(f"{self._name}_{other._name}")
        new_registry._registry = {**self._registry, **other._registry}
        return new_registry
