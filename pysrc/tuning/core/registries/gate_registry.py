"""GateRegistry: versioned registry for GateProtocol implementations."""

from __future__ import annotations

from collections.abc import Callable

from pysrc.tuning.core.contracts.gate import GateProtocol

_registry: dict[str, dict[str, type[GateProtocol]]] = {}


def register(name: str, version: str) -> Callable[[type[GateProtocol]], type[GateProtocol]]:
    """Decorator: register a GateProtocol implementation under name@version."""

    def _inner(cls: type[GateProtocol]) -> type[GateProtocol]:
        _registry.setdefault(name, {})[version] = cls
        return cls

    return _inner


def get(name: str, version: str = "latest") -> type[GateProtocol]:
    if name not in _registry:
        raise KeyError(f"No gate registered as '{name}'")
    versions = _registry[name]
    key = max(versions) if version == "latest" else version
    if key not in versions:
        raise KeyError(f"Gate '{name}' has no version '{version}'")
    return versions[key]


def list_registered() -> list[tuple[str, str]]:
    return [(n, v) for n, vs in _registry.items() for v in vs]


__all__ = ["register", "get", "list_registered"]
