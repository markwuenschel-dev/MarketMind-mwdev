"""Lineage: typed record linking an artifact to its parent inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LineageRecord:
    """Immutable lineage record: output artifact plus its input artifact hashes."""

    output_hash: str
    input_hashes: tuple[str, ...]
    operation: str
    job_id: str
    recorded_at: datetime


def record_lineage(
    output_hash: str,
    input_hashes: tuple[str, ...],
    operation: str,
    job_id: str,
    now: datetime,
) -> LineageRecord:
    """Construct and return a LineageRecord."""
    return LineageRecord(
        output_hash=output_hash,
        input_hashes=input_hashes,
        operation=operation,
        job_id=job_id,
        recorded_at=now,
    )


__all__ = ["LineageRecord", "record_lineage"]
