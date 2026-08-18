"""Pure regime-label projection onto task fold boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = ["RegimeSegment", "project_regime"]


@dataclass(frozen=True)
class RegimeSegment:
    """A contiguous time segment labeled with a regime identifier."""

    label: str
    start: datetime
    end: datetime


def project_regime(
    fold_start: datetime,
    fold_end: datetime,
    segments: tuple[RegimeSegment, ...],
) -> str:
    """Return the dominant regime label that overlaps most with [fold_start, fold_end].

    Returns "unknown" when no segment overlaps the fold window.
    """
    overlap: dict[str, float] = {}
    for seg in segments:
        lo = max(seg.start, fold_start)
        hi = min(seg.end, fold_end)
        if hi > lo:
            overlap[seg.label] = overlap.get(seg.label, 0.0) + (hi - lo).total_seconds()
    if not overlap:
        return "unknown"
    return max(overlap, key=lambda k: overlap[k])
