"""FeatureAdapterRegistry: versioned registry for feature pipeline adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FeatureAdapterProtocol(Protocol):
    """Adapts raw features to a model-specific feature matrix."""

    def transform(self, data: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame: ...


_registry: dict[str, dict[str, type[FeatureAdapterProtocol]]] = {}


def register(
    name: str, version: str
) -> Callable[[type[FeatureAdapterProtocol]], type[FeatureAdapterProtocol]]:
    """Decorator: register a FeatureAdapterProtocol implementation under name@version."""

    def _inner(cls: type[FeatureAdapterProtocol]) -> type[FeatureAdapterProtocol]:
        _registry.setdefault(name, {})[version] = cls
        return cls

    return _inner


def get(name: str, version: str = "latest") -> type[FeatureAdapterProtocol]:
    if name not in _registry:
        raise KeyError(f"No feature adapter registered as '{name}'")
    versions = _registry[name]
    key = max(versions) if version == "latest" else version
    if key not in versions:
        raise KeyError(f"Feature adapter '{name}' has no version '{version}'")
    return versions[key]


def list_registered() -> list[tuple[str, str]]:
    return [(n, v) for n, vs in _registry.items() for v in vs]


__all__ = ["FeatureAdapterProtocol", "register", "get", "list_registered"]
