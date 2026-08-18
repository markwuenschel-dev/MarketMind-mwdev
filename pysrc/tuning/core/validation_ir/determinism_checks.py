"""Determinism tier checks: enforce that IR objects carry a valid determinism marker."""

from __future__ import annotations

from typing import Literal

from pysrc.tuning.core.ir.nodes import IRMetadata

__all__ = [
    "VALID_TIERS",
    "DeterminismViolationError",
    "validate_tier",
    "assert_tier_not_downgraded",
]

VALID_TIERS: frozenset[str] = frozenset({"d0", "d1", "d2", "d3"})
_TIER_ORDER: dict[str, int] = {"d0": 0, "d1": 1, "d2": 2, "d3": 3}


class DeterminismViolationError(ValueError):
    """Raised when an IR object's determinism tier is invalid or downgraded."""


def validate_tier(meta: IRMetadata) -> IRMetadata:
    """Assert that the determinism tier in *meta* is one of the four valid values."""
    if meta.determinism_tier not in VALID_TIERS:
        raise DeterminismViolationError(f"Unknown determinism tier: {meta.determinism_tier!r}")
    return meta


def assert_tier_not_downgraded(
    current: Literal["d0", "d1", "d2", "d3"],
    proposed: Literal["d0", "d1", "d2", "d3"],
) -> None:
    """Raise if proposed tier is weaker (higher index) than current."""
    if _TIER_ORDER[proposed] > _TIER_ORDER[current]:
        raise DeterminismViolationError(
            f"Downgrading determinism from {current!r} to {proposed!r} requires an ADR"
        )
