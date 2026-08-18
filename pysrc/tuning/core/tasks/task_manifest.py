"""TaskManifest: immutable ordered record of all TaskIRs in a job."""

from __future__ import annotations

from dataclasses import dataclass

from pysrc.tuning.core.ir.task_ir import TaskIR

__all__ = ["TaskManifest"]


@dataclass(frozen=True)
class TaskManifest:
    """Ordered, immutable list of TaskIRs for a single tuning job."""

    job_id: str
    tasks: tuple[TaskIR, ...]
    spec_hash: str

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def by_id(self, task_id: str) -> TaskIR:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"TaskIR not found: {task_id!r}")
