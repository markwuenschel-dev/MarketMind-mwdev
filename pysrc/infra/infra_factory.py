# py/infra/infra_factory.py
"""Lightweight, import-safe registry + factory for data sources.

- No import-time side effects (no top-level lookups or context managers).
- Thread-safe register/unregister.
- Exposes both a modern factory class and a legacy 'creator' callable.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable
from threading import RLock
from typing import Any

# Optional project logger; fall back to stdlib
try:
    from pysrc.ops.mm_logkit import get_logger  # type: ignore

    log = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Internal registry (protected by a lock)
# -----------------------------------------------------------------------------
_Source = Callable[..., Any]
_registry_lock = RLock()
_source_registry: dict[str, _Source] = {}


def register_source(name: str, creator: _Source) -> None:
    """Register a source constructor under a lowercase name."""
    if not isinstance(name, str):
        raise TypeError(f"name must be str, got {type(name)}")
    if not callable(creator):
        raise TypeError("creator must be callable")
    lname = name.lower()
    with _registry_lock:
        _source_registry[lname] = creator
        with contextlib.suppress(Exception):
            log.debug("Registered data source '%s' -> %r", lname, creator)


def unregister_source(name: str) -> None:
    """Remove a source by name; no error if missing."""
    lname = name.lower()
    with _registry_lock:
        _source_registry.pop(lname, None)
        with contextlib.suppress(Exception):
            log.debug("Unregistered data source '%s'", lname)


def get_creator(source_type: str) -> _Source | None:
    """Return the registered constructor for a given type (case-insensitive)."""
    if not isinstance(source_type, str):
        raise TypeError(f"source_type must be str, got {type(source_type)}")
    with _registry_lock:
        return _source_registry.get(source_type.lower())


def list_sources() -> list[str]:
    """List available registered source type names (sorted)."""
    with _registry_lock:
        return sorted(_source_registry.keys())


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
class DataSourceFactory:
    """Factory that instantiates registered data sources."""

    @staticmethod
    def create(source_type: str, /, **kwargs: Any) -> Any:
        """Instantiate a data source of the given type with **kwargs."""
        ctor = get_creator(source_type)
        if ctor is None:
            raise ValueError(
                f"Unknown data source type: {source_type!r}. "
                f"Available: {list_sources() or '(none)'}"
            )
        if inspect.isclass(ctor):
            return ctor(**kwargs)  # type: ignore[misc]
        return ctor(**kwargs)


# -----------------------------------------------------------------------------
# Legacy convenience: callable that returns ctor or instance
# -----------------------------------------------------------------------------
class _LegacyCreatorProxy:
    def __call__(self, source_type: str, /, **kwargs: Any):
        ctor = get_creator(source_type)
        if ctor is None:
            raise ValueError(
                f"Unknown data source type: {source_type!r}. "
                f"Available: {list_sources() or '(none)'}"
            )
        # If kwargs provided, instantiate now; else return the callable itself.
        return ctor(**kwargs) if kwargs else ctor


creator = _LegacyCreatorProxy()

__all__ = [
    "register_source",
    "unregister_source",
    "get_creator",
    "list_sources",
    "DataSourceFactory",
    "creator",
]
