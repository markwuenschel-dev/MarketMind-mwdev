"""TaskIR: canonical IR for a single tuning task (one fold + one candidate)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pysrc.tuning.core.ir.nodes import HParam, IRMetadata

__all__ = ["FoldBoundary", "TaskIR"]


@dataclass(frozen=True)
class FoldBoundary:
    """Index boundaries for a single train/test fold."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True)
class TaskIR:
    """Immutable specification for a single training + evaluation task."""

    task_id: str
    job_id: str
    candidate_id: str
    fold: FoldBoundary
    params: tuple[HParam, ...]
    feature_hash: str
    meta: IRMetadata
    symbol: str = ""
    regime_label: str = ""
