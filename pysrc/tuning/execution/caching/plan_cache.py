"""PlanCache: keyed in-memory cache for plan objects."""

from __future__ import annotations

from typing import Any


class PlanCache:
    """Dict-backed cache for plan objects keyed by plan hash."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, plan_hash: str) -> Any | None:
        return self._store.get(plan_hash)

    def put(self, plan_hash: str, plan: Any) -> None:
        self._store[plan_hash] = plan

    def evict(self, plan_hash: str) -> None:
        self._store.pop(plan_hash, None)


__all__ = ["PlanCache"]
