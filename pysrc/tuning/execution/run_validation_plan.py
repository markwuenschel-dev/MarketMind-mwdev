"""run_validation_plan: execute cross-validation for a set of candidates."""

from __future__ import annotations

from typing import Any

from pysrc.tuning.core.ir.validation_ir import ValidationIR


def run_validation_plan(
    ir: ValidationIR,
    candidates: list[dict[str, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run validation folds for each candidate; return per-fold score dicts."""
    raise NotImplementedError("run_validation_plan must be wired to executor and data layers")


__all__ = ["run_validation_plan"]
