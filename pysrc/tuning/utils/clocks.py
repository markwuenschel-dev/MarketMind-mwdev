"""UTC-aware time utilities; never returns naive datetimes.

Centralises time access so tests can monkey-patch a single call site.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

__all__ = ["utc_now", "monotonic_ns"]


def utc_now() -> datetime:
    """Return the current UTC-aware datetime."""
    return datetime.now(tz=UTC)


def monotonic_ns() -> int:
    """Return the monotonic clock in nanoseconds."""
    return time.monotonic_ns()
