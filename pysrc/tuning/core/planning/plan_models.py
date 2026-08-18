"""Base plan types and execution budget model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["ExecutionBudget", "PlanMetadata"]


@dataclass(frozen=True)
class ExecutionBudget:
    """Declares the resource envelope for a single tuning job execution."""

    max_trials: int
    timeout_seconds: int | None
    max_parallel: int = 1
    priority: Literal["throughput", "latency", "cost"] = "throughput"


@dataclass(frozen=True)
class PlanMetadata:
    """Provenance fields common to all plan objects."""

    plan_hash: str
    spec_hash: str
    created_at_ns: int
    determinism_tier: str = "d1"
