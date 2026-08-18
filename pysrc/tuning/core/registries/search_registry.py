"""SearchRegistry: versioned registry for SearchProtocol implementations."""

from __future__ import annotations

from collections.abc import Callable

from pysrc.tuning.core.contracts.search import SearchProtocol

_registry: dict[str, dict[str, type[SearchProtocol]]] = {}


def register(name: str, version: str) -> Callable[[type[SearchProtocol]], type[SearchProtocol]]:
    """Decorator: register a SearchProtocol implementation under name@version."""

    def _inner(cls: type[SearchProtocol]) -> type[SearchProtocol]:
        _registry.setdefault(name, {})[version] = cls
        return cls

    return _inner


def get(name: str, version: str = "latest") -> type[SearchProtocol]:
    if name not in _registry:
        raise KeyError(f"No search algorithm registered as '{name}'")
    versions = _registry[name]
    key = max(versions) if version == "latest" else version
    if key not in versions:
        raise KeyError(f"Search algorithm '{name}' has no version '{version}'")
    return versions[key]


def list_registered() -> list[tuple[str, str]]:
    return [(n, v) for n, vs in _registry.items() for v in vs]


__all__ = ["register", "get", "list_registered"]
