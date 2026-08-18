from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pysrc.backtesting.contracts.registry import register_cost_model
from pysrc.backtesting.contracts.types import CostEstimate


@dataclass(frozen=True)
class FeeModelConfig:
    commission_bps: float = 0.0


class FlatFeeCostModel:
    def __init__(self, config: FeeModelConfig | None = None) -> None:
        self.config = config or FeeModelConfig()

    def estimate(self, fills: list[dict[str, Any]], ctx: dict[str, Any]) -> CostEstimate:
        total_notional = sum(
            abs(float(fill.get("quantity", 0.0))) * float(fill.get("price", 0.0)) for fill in fills
        )
        commission = total_notional * (self.config.commission_bps / 10000.0)
        return CostEstimate(total_cost=commission, components={"commission": commission})

    def to_execution_assumptions(self) -> dict[str, Any]:
        return {
            "cost_model_id": "fees.zero" if self.config.commission_bps == 0.0 else "fees.flat",
            "commission_bps": self.config.commission_bps,
        }


register_cost_model("fees.zero", lambda: FlatFeeCostModel(FeeModelConfig(commission_bps=0.0)))
register_cost_model("fees.flat", lambda: FlatFeeCostModel(FeeModelConfig(commission_bps=1.0)))
