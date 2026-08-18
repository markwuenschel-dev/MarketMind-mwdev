"""CachePlan: which intermediate results should be cached and at which granularity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["CacheEntry", "CachePlan"]


@dataclass(frozen=True)
class CacheEntry:
    """Describes a single cacheable unit within an execution plan."""

    key: str
    scope: Literal["spec", "ir", "plan", "feature", "artifact"]
    ttl_seconds: int | None = None


@dataclass(frozen=True)
class CachePlan:
    """Declares cache entries for a given execution plan."""

    job_id: str
    entries: tuple[CacheEntry, ...]
    default_ttl_seconds: int | None = 3600
