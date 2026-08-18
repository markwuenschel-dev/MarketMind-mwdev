"""Tracing: lightweight span-based instrumentation for the tuning pipeline."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """A single named tracing span with start/end times and metadata."""

    name: str
    start_ns: int = field(default_factory=time.monotonic_ns)
    end_ns: int | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        """Mark the span as complete."""
        self.end_ns = time.monotonic_ns()

    @property
    def duration_ms(self) -> float | None:
        """Return span duration in milliseconds, or None if not yet finished."""
        if self.end_ns is None:
            return None
        return (self.end_ns - self.start_ns) / 1e6


@contextmanager
def trace(name: str, tags: dict[str, Any] | None = None) -> Generator[Span, None, None]:
    """Context manager that creates a Span, yields it, then finishes it."""
    span = Span(name=name, tags=tags or {})
    try:
        yield span
    finally:
        span.finish()


__all__ = ["Span", "trace"]
