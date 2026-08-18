"""SearchSpaceConfig: declarative spec for a parameter search space."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["DimensionConfig", "SearchSpaceConfig"]


class DimensionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["real", "int", "categorical", "log_real"]
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    prior: Literal["uniform", "log-uniform"] = "uniform"


class SearchSpaceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str = "1.0.0"
    model_type: str
    dimensions: list[DimensionConfig] = Field(default_factory=list)
    fixed: dict[str, Any] = Field(default_factory=dict)
