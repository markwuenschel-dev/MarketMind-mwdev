"""ModelRegistry: versioned registry for trainable model factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelFactory(Protocol):
    """Returns a new, unfitted model instance given hyperparameters."""

    def __call__(self, params: dict[str, Any]) -> Any: ...


_registry: dict[str, dict[str, ModelFactory]] = {}


def register(name: str, version: str) -> Callable[[ModelFactory], ModelFactory]:
    """Decorator: register a ModelFactory under name@version."""

    def _inner(factory: ModelFactory) -> ModelFactory:
        _registry.setdefault(name, {})[version] = factory
        return factory

    return _inner


def get(name: str, version: str = "latest") -> ModelFactory:
    if name not in _registry:
        raise KeyError(f"No model factory registered as '{name}'")
    versions = _registry[name]
    key = max(versions) if version == "latest" else version
    if key not in versions:
        raise KeyError(f"Model '{name}' has no version '{version}'")
    return versions[key]


def list_registered() -> list[tuple[str, str]]:
    return [(n, v) for n, vs in _registry.items() for v in vs]


__all__ = ["ModelFactory", "register", "get", "list_registered"]
