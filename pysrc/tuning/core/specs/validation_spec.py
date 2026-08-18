"""ValidationSpec: validated, frozen spec for a cross-validation strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ValidationSpec:
    """Validated, immutable validation spec derived from ValidationConfig."""

    name: str
    version: str
    spec_hash: str
    method: Literal["walkforward", "purged_cv", "cpcv", "crisis_holdout"]
    n_splits: int
    embargo_periods: int
    crisis_holdout: bool
    cost_stress_bps: float
    min_train_periods: int


__all__ = ["ValidationSpec"]
