"""SearchSpaceSpec: validated, frozen spec for a parameter search space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class DimensionSpec:
    name: str
    kind: Literal["real", "int", "categorical", "log_real"]
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None
    prior: Literal["uniform", "log-uniform"] = "uniform"


@dataclass(frozen=True)
class SearchSpaceSpec:
    """Validated, immutable search space derived from SearchSpaceConfig."""

    name: str
    version: str
    model_type: str
    spec_hash: str
    dimensions: tuple[DimensionSpec, ...] = field(default_factory=tuple)
    fixed: dict[str, Any] = field(default_factory=dict)


__all__ = ["DimensionSpec", "SearchSpaceSpec"]
