"""run_training_plan: train a candidate model on a designated partition."""

from __future__ import annotations

from typing import Any

from pysrc.tuning.core.ir.task_ir import TaskIR


def run_training_plan(task_ir: TaskIR, context: dict[str, Any]) -> dict[str, Any]:
    """Train and evaluate a single candidate; return a result artifact dict."""
    raise NotImplementedError("run_training_plan must be wired to model registry and data layer")


__all__ = ["run_training_plan"]
