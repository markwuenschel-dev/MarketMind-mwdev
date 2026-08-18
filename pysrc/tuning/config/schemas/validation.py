"""ValidationConfig: declarative spec for cross-validation strategy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ValidationConfig"]


class ValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str = "1.0.0"
    method: Literal["walkforward", "purged_cv", "cpcv", "crisis_holdout"] = "purged_cv"
    n_splits: int = Field(default=5, ge=2)
    embargo_periods: int = Field(default=10, ge=0)
    crisis_holdout: bool = True
    cost_stress_bps: float = 0.0
    min_train_periods: int = Field(default=252, ge=1)
