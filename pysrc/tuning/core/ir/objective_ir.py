"""ObjectiveIR: canonical IR for a composite objective function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pysrc.tuning.core.ir.nodes import IRMetadata, Scalar

__all__ = ["PenaltySpec", "ObjectiveIR"]


@dataclass(frozen=True)
class PenaltySpec:
    """A named penalty term attached to an objective."""

    name: str
    coefficient: float


@dataclass(frozen=True)
class ObjectiveIR:
    """Immutable objective specification with penalty terms."""

    job_id: str
    direction: Literal["maximize", "minimize"]
    metrics: tuple[str, ...]
    weights: dict[str, float]
    penalties: tuple[PenaltySpec, ...]
    constraints: tuple[Scalar, ...]
    meta: IRMetadata
