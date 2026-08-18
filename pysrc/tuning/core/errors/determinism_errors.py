"""Typed determinism tier errors."""

from __future__ import annotations


class DeterminismError(ValueError):
    """Base class for determinism tier errors."""


class TierDowngradeError(DeterminismError):
    """Raised when a proposed tier is weaker than the current tier without an ADR."""


class InvalidTierError(DeterminismError):
    """Raised when an unrecognised determinism tier string is encountered."""


__all__ = ["DeterminismError", "TierDowngradeError", "InvalidTierError"]
