"""ObjectiveRegistry: versioned registry for ObjectiveProtocol implementations."""

from __future__ import annotations

from collections.abc import Callable

from pysrc.tuning.core.contracts.objective import ObjectiveProtocol

_registry: dict[str, dict[str, type[ObjectiveProtocol]]] = {}


def register(
    name: str, version: str
) -> Callable[[type[ObjectiveProtocol]], type[ObjectiveProtocol]]:
    """Decorator: register an ObjectiveProtocol implementation under name@version."""

    def _inner(cls: type[ObjectiveProtocol]) -> type[ObjectiveProtocol]:
        _registry.setdefault(name, {})[version] = cls
        return cls

    return _inner


def get(name: str, version: str = "latest") -> type[ObjectiveProtocol]:
    if name not in _registry:
        raise KeyError(f"No objective registered as '{name}'")
    versions = _registry[name]
    key = max(versions) if version == "latest" else version
    if key not in versions:
        raise KeyError(f"Objective '{name}' has no version '{version}'")
    return versions[key]


def list_registered() -> list[tuple[str, str]]:
    return [(n, v) for n, vs in _registry.items() for v in vs]


__all__ = ["register", "get", "list_registered"]
