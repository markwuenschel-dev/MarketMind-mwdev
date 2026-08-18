from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pysrc.backtesting.contracts.errors import DeterminismTierMissingError


class DeterminismTier(StrEnum):
    D0 = "d0"
    D1 = "d1"
    D2 = "d2"
    D3 = "d3"


@dataclass(frozen=True)
class EngineConfig:
    """Serialized engine-local configuration."""

    lane: str = "vectorized"
    bar_frequency: str = "1d"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestPlan:
    """Stable plan surface for backtesting substrate components."""

    engine_id: str
    execution_model_id: str
    cost_model_id: str
    ledger_id: str
    validator_ids: list[str]
    determinism: DeterminismTier
    seed: int
    pit_required: bool
    engine_config: EngineConfig | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.determinism is None:
            raise DeterminismTierMissingError("BacktestPlan requires a determinism tier (d0-d3).")
