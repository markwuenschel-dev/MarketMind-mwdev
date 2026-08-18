"""ArtifactCache: thin read-through cache for artifact payloads by CAS hash."""

from __future__ import annotations

from typing import Any


class ArtifactCache:
    """Read-through cache for artifact payloads keyed by CAS hash."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, cas_hash: str) -> Any | None:
        return self._store.get(cas_hash)

    def put(self, cas_hash: str, payload: Any) -> None:
        self._store[cas_hash] = payload


__all__ = ["ArtifactCache"]
