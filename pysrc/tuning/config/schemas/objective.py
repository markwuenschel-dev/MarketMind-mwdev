"""ObjectiveConfig: declarative spec for a tuning objective."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ObjectiveConfig"]


class ObjectiveConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str = "1.0.0"
    direction: Literal["maximize", "minimize"] = "maximize"
    metrics: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    penalty_refs: list[str] = Field(default_factory=list)
    min_sharpe: float | None = None
    max_turnover: float | None = None
