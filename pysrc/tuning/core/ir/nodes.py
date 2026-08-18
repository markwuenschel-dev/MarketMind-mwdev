"""Primitive IR node/value types shared across all IR objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Scalar", "HParam", "IRMetadata"]


@dataclass(frozen=True)
class Scalar:
    """A single named scalar value with optional units."""

    name: str
    value: float
    units: str = ""


@dataclass(frozen=True)
class HParam:
    """A single hyperparameter name+value pair."""

    name: str
    value: Any


@dataclass(frozen=True)
class IRMetadata:
    """Common provenance fields attached to every IR object."""

    spec_hash: str  # source spec hash; format: "cas.v1:b3-256:<hex>"
    created_at_ns: int  # monotonic nanoseconds at creation
    determinism_tier: str = "d1"
    tags: dict[str, str] = field(default_factory=dict)
