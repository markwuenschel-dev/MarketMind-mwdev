"""Append-only in-memory TaskRegistry (MLN-01)."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from pysrc.meta_learning.contracts.meta_task import MetaTask
from pysrc.meta_learning.contracts.task_registry import (
    TaskNotFoundError,
    TaskRegistryDuplicateError,
)


class AppendOnlyTaskRegistry:
    """Keyed by ``task_id`` and stable identity ``(regime_id, t0)``; no silent overwrites."""

    CONTRACT_VERSION: ClassVar[str] = "v1"

    def __init__(self) -> None:
        self._by_id: dict[str, MetaTask] = {}
        self._by_stable: dict[tuple[str, str], str] = {}

    def append(self, task: MetaTask) -> None:
        if task.task_id in self._by_id:
            raise TaskRegistryDuplicateError(
                "task_id already registered",
                details={"task_id": task.task_id},
            )
        stable = (task.regime_id, task.t0)
        if stable in self._by_stable:
            raise TaskRegistryDuplicateError(
                "stable identity (regime_id, t0) already registered",
                details={
                    "regime_id": task.regime_id,
                    "t0": task.t0,
                    "existing_task_id": self._by_stable[stable],
                },
            )
        self._by_id[task.task_id] = task
        self._by_stable[stable] = task.task_id

    def get(self, task_id: str) -> MetaTask:
        if task_id not in self._by_id:
            raise TaskNotFoundError("task not found", details={"task_id": task_id})
        return self._by_id[task_id]

    def get_by_stable(self, *, regime_id: str, t0: str) -> MetaTask:
        key = (regime_id, t0)
        if key not in self._by_stable:
            raise TaskNotFoundError(
                "task not found for stable key", details={"regime_id": regime_id, "t0": t0}
            )
        return self._by_id[self._by_stable[key]]

    def query(
        self,
        regime_id: str | None = None,
        since: str | None = None,
    ) -> list[MetaTask]:
        """``since`` is ISO 8601 UTC, compared to ``MetaTask.t0`` (``TaskRegistryProtocol``)."""
        out = list(self._by_id.values())
        if regime_id is not None:
            out = [t for t in out if t.regime_id == regime_id]
        if since is not None:
            since_ts = pd.Timestamp(since, tz="UTC")
            out = [t for t in out if pd.Timestamp(t.t0, tz="UTC") >= since_ts]
        return sorted(out, key=lambda x: (x.regime_id, x.t0))

    def to_records(self) -> list[dict[str, Any]]:
        from pysrc.meta_learning.task_generator import meta_task_to_record

        return [
            meta_task_to_record(t) for t in sorted(self._by_id.values(), key=lambda x: x.task_id)
        ]

    def __len__(self) -> int:
        return len(self._by_id)


__all__ = ["AppendOnlyTaskRegistry"]
