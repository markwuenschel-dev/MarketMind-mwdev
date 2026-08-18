"""PlannerRegistry: versioned registry for PlannerProtocol implementations."""

from __future__ import annotations

from collections.abc import Callable

from pysrc.tuning.core.contracts.planner import PlannerProtocol

_registry: dict[str, dict[str, type[PlannerProtocol]]] = {}


def register(name: str, version: str) -> Callable[[type[PlannerProtocol]], type[PlannerProtocol]]:
    """Decorator: register a PlannerProtocol implementation under name@version."""

    def _inner(cls: type[PlannerProtocol]) -> type[PlannerProtocol]:
        _registry.setdefault(name, {})[version] = cls
        return cls

    return _inner


def get(name: str, version: str = "latest") -> type[PlannerProtocol]:
    if name not in _registry:
        raise KeyError(f"No planner registered as '{name}'")
    versions = _registry[name]
    key = max(versions) if version == "latest" else version
    if key not in versions:
        raise KeyError(f"Planner '{name}' has no version '{version}'")
    return versions[key]


def list_registered() -> list[tuple[str, str]]:
    return [(n, v) for n, vs in _registry.items() for v in vs]


__all__ = ["register", "get", "list_registered"]
