"""PartitionPlan: how a job is split across symbols, time windows, folds, and tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = ["TimePartition", "PartitionPlan"]


@dataclass(frozen=True)
class TimePartition:
    """A named time window used as a training or evaluation boundary."""

    name: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class PartitionPlan:
    """Declares how the job is divided across symbols, folds, and time windows."""

    job_id: str
    symbols: tuple[str, ...]
    time_partitions: tuple[TimePartition, ...]
    n_folds: int
    n_total_tasks: int
