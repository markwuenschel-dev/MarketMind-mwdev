"""ValidationIR: canonical IR for a cross-validation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pysrc.tuning.core.ir.nodes import IRMetadata

__all__ = ["EmbargoSpec", "ValidationIR"]


@dataclass(frozen=True)
class EmbargoSpec:
    """Embargo parameters to prevent leakage between train/test splits."""

    periods: int
    unit: Literal["bars", "days"] = "bars"


@dataclass(frozen=True)
class ValidationIR:
    """Immutable CV configuration derived from ValidationSpec."""

    job_id: str
    method: Literal["walkforward", "purged_cv", "cpcv", "crisis_holdout"]
    n_splits: int
    embargo: EmbargoSpec
    crisis_holdout: bool
    cost_stress_bps: float
    min_train_periods: int
    meta: IRMetadata
