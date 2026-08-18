"""Scheduler: determines when and in what order jobs are dispatched."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScheduledJob:
    job_id: str
    priority: int = 0
    tags: dict[str, str] = field(default_factory=dict)


class Scheduler:
    """Priority-queue scheduler for pending tuning jobs."""

    def __init__(self) -> None:
        self._queue: list[ScheduledJob] = []

    def enqueue(self, job: ScheduledJob) -> None:
        """Add a job to the scheduling queue."""
        self._queue.append(job)
        self._queue.sort(key=lambda j: -j.priority)

    def next(self) -> ScheduledJob | None:
        """Pop and return the highest-priority pending job, or None if empty."""
        return self._queue.pop(0) if self._queue else None

    def pending(self) -> list[ScheduledJob]:
        """Return a snapshot of the pending queue (read-only)."""
        return list(self._queue)


__all__ = ["ScheduledJob", "Scheduler"]
