"""ObjectiveSpec: validated, frozen spec for an objective function."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ObjectiveSpec:
    """Validated, immutable objective spec derived from ObjectiveConfig."""

    name: str
    version: str
    spec_hash: str
    direction: Literal["maximize", "minimize"]
    metrics: tuple[str, ...]
    weights: dict[str, float] = field(default_factory=dict)
    penalty_refs: tuple[str, ...] = field(default_factory=tuple)
    min_sharpe: float | None = None
    max_turnover: float | None = None


__all__ = ["ObjectiveSpec"]
