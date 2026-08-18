"""SpecCache: keyed in-memory cache for Spec objects (keyed by spec hash)."""

from __future__ import annotations

from typing import Any


class SpecCache:
    """Simple LRU-style dict-backed cache for spec objects."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, spec_hash: str) -> Any | None:
        return self._store.get(spec_hash)

    def put(self, spec_hash: str, spec: Any) -> None:
        self._store[spec_hash] = spec

    def evict(self, spec_hash: str) -> None:
        self._store.pop(spec_hash, None)


__all__ = ["SpecCache"]
