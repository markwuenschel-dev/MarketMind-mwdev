"""Dimension types and bounds utilities for typed search space navigation."""

from __future__ import annotations

from pysrc.tuning.core.specs.search_space_spec import DimensionSpec


def is_continuous(dim: DimensionSpec) -> bool:
    """Return True if the dimension is real-valued (real or log_real)."""
    return dim.kind in ("real", "log_real")


def is_discrete(dim: DimensionSpec) -> bool:
    """Return True if the dimension is discrete (int or categorical)."""
    return dim.kind in ("int", "categorical")


def bounds(dim: DimensionSpec) -> tuple[float, float]:
    """Return (low, high) for continuous/int dims; raises ValueError for categoricals."""
    if dim.low is None or dim.high is None:
        raise ValueError(f"Dimension {dim.name!r} has no bounds (categorical?)")
    return (dim.low, dim.high)


__all__ = ["is_continuous", "is_discrete", "bounds"]
