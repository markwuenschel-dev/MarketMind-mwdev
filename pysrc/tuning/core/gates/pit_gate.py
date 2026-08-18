"""PIT gate: verify no future data leaked into the candidate's training window."""

from __future__ import annotations

from datetime import datetime

from pysrc.tuning.core.ir.task_ir import TaskIR
from pysrc.tuning.core.validation_ir.pit_checks import PITViolationError, validate_task_pit


def passes_pit_gate(task_ir: TaskIR, as_of: datetime) -> tuple[bool, str]:
    """Return (passed, reason); never raises — converts PITViolationError to reason string."""
    try:
        validate_task_pit(task_ir, as_of)
        return True, "pit_ok"
    except PITViolationError as exc:
        return False, str(exc)


__all__ = ["passes_pit_gate"]
