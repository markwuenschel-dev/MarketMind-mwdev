"""IRCache: keyed in-memory cache for IR objects."""

from __future__ import annotations

from typing import Any


class IRCache:
    """Dict-backed cache for IR objects keyed by ir hash."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, ir_hash: str) -> Any | None:
        return self._store.get(ir_hash)

    def put(self, ir_hash: str, ir: Any) -> None:
        self._store[ir_hash] = ir

    def evict(self, ir_hash: str) -> None:
        self._store.pop(ir_hash, None)


__all__ = ["IRCache"]
