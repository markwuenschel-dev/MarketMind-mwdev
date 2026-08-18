"""Typed point-in-time boundary errors."""

from __future__ import annotations


class PITError(ValueError):
    """Base class for all point-in-time boundary errors."""


class PITBoundaryError(PITError):
    """Raised when a fold boundary extends beyond the as_of timestamp."""


class PITLeakageError(PITError):
    """Raised when test data is present in a training window."""


__all__ = ["PITError", "PITBoundaryError", "PITLeakageError"]
