"""ObjectiveProtocol: interface for objective/scoring functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from pysrc.tuning.core.ir.objective_ir import ObjectiveIR


@runtime_checkable
class ObjectiveProtocol(Protocol):
    """Pure scoring function: returns a scalar from returns + ir."""

    def score(self, ir: ObjectiveIR, returns: pd.Series) -> float: ...

    def is_feasible(self, ir: ObjectiveIR, returns: pd.Series) -> bool: ...


__all__ = ["ObjectiveProtocol"]
