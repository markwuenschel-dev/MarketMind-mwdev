"""TaskProtocol: interface for tuning task construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pysrc.tuning.core.ir.task_ir import TaskIR


@runtime_checkable
class TaskProtocol(Protocol):
    """Constructs a TaskIR from a job spec and partition information."""

    def build(self, spec_hash: str, partition: dict[str, object]) -> TaskIR: ...

    def key(self, task_ir: TaskIR) -> str: ...


__all__ = ["TaskProtocol"]
