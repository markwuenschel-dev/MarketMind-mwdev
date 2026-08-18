"""ValidationRegistry: versioned registry for ValidatorProtocol implementations."""

from __future__ import annotations

from collections.abc import Callable

from pysrc.tuning.core.contracts.validator import ValidatorProtocol

_registry: dict[str, dict[str, type[ValidatorProtocol]]] = {}


def register(
    name: str, version: str
) -> Callable[[type[ValidatorProtocol]], type[ValidatorProtocol]]:
    """Decorator: register a ValidatorProtocol implementation under name@version."""

    def _inner(cls: type[ValidatorProtocol]) -> type[ValidatorProtocol]:
        _registry.setdefault(name, {})[version] = cls
        return cls

    return _inner


def get(name: str, version: str = "latest") -> type[ValidatorProtocol]:
    if name not in _registry:
        raise KeyError(f"No validator registered as '{name}'")
    versions = _registry[name]
    key = max(versions) if version == "latest" else version
    if key not in versions:
        raise KeyError(f"Validator '{name}' has no version '{version}'")
    return versions[key]


def list_registered() -> list[tuple[str, str]]:
    return [(n, v) for n, vs in _registry.items() for v in vs]


__all__ = ["register", "get", "list_registered"]
