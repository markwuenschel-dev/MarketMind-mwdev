from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pysrc.backtesting.contracts.registry import register_execution_model


@dataclass(frozen=True)
class FillModelConfig:
    fill_ratio: float = 1.0


class IdentityFillModel:
    def __init__(self, config: FillModelConfig | None = None) -> None:
        self.config = config or FillModelConfig()

    def simulate(self, orders: list[dict[str, Any]], ctx: dict[str, Any]) -> list[dict[str, Any]]:
        timestamp = str(ctx.get("timestamp", "1970-01-01T00:00:00+00:00"))
        fills: list[dict[str, Any]] = []
        for order in orders:
            fills.append(
                {
                    "symbol": order.get("symbol", "UNKNOWN"),
                    "quantity": float(order.get("quantity", 0.0)) * self.config.fill_ratio,
                    "price": float(order.get("price", 0.0)),
                    "side": str(order.get("side", "BUY")),
                    "timestamp": str(order.get("timestamp", timestamp)),
                }
            )
        return fills

    def to_execution_assumptions(self) -> dict[str, Any]:
        return {
            "fill_model_id": "fill.identity",
            "fill_ratio": self.config.fill_ratio,
        }


register_execution_model("fill.identity", lambda: IdentityFillModel())
