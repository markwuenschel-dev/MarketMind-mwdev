"""ValidatorProtocol: interface for cross-validation strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from pysrc.tuning.core.ir.validation_ir import ValidationIR


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Pure, deterministic validator; returns fold split indices."""

    def splits(self, ir: ValidationIR, data: pd.Index) -> list[tuple[pd.Index, pd.Index]]: ...

    def n_splits(self, ir: ValidationIR) -> int: ...


__all__ = ["ValidatorProtocol"]
