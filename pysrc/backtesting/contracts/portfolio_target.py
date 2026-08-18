"""Portfolio target plan contract for backtesting handoff.

Contract layers (orthogonal — do not merge):

- ``BacktestPlan`` (``backtesting/contracts/plan.py``): engine substrate —
  execution model, cost model, ledger, determinism tier.
- ``BacktestSuitePlan`` (``backtesting/orchestration/plan.py``): batch of
  ``BacktestPlan`` instances plus shared context.
- ``PortfolioTargetPlan`` (this module): **allocation decision** — instrument
  weights, cash, exposure, gate lineage from the local MetaRouter.
- ``PortfolioService.rebalance(target_weights)`` (``domain/portfolio/service.py``):
  domain rebalance API; accepts weight maps, not engine config.

``PortfolioTargetPlan`` is the canonical handoff for MetaRouter → backtesting.
The pilot wires it via ``to_backtest_context()`` into ``BacktestSuitePlan.context``
and via ``to_weight_path_frame()`` into the same weight-panel shape used by
``portfolio.labels.compute_weight_path_series``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class InstrumentTarget:
    symbol: str
    weight: float


@dataclass(frozen=True)
class PortfolioTargetPlan:
    """Validated portfolio allocation plan consumable by backtesting."""

    decision_timestamp: str
    interval: str
    instrument_weights: tuple[InstrumentTarget, ...]
    cash_weight: float
    gross_exposure: float
    net_exposure: float
    gate_id: str
    policy_lineage: str
    model_weights: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    constraints_applied: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cash_weight < 0.0:
            raise ValueError("cash_weight must be non-negative")
        weight_sum = sum(target.weight for target in self.instrument_weights) + self.cash_weight
        if abs(weight_sum - 1.0) > 1e-4:
            raise ValueError(f"instrument weights + cash must sum to 1.0 (got {weight_sum:.6f})")

    def target_weight_map(self) -> dict[str, float]:
        return {target.symbol: target.weight for target in self.instrument_weights}

    def rebalance_weight_map(self) -> dict[str, Decimal]:
        """Weight map compatible with ``PortfolioService.rebalance``."""

        return {symbol: Decimal(str(weight)) for symbol, weight in self.target_weight_map().items()}

    def to_weight_path_frame(self) -> pd.DataFrame:
        """Weight panel for portfolio simulation and backtest engine input."""

        if not self.instrument_weights:
            return pd.DataFrame(columns=["date", "ticker", "target_weight"])
        return pd.DataFrame(
            [
                {
                    "date": self.decision_timestamp,
                    "ticker": target.symbol,
                    "target_weight": target.weight,
                }
                for target in self.instrument_weights
            ]
        )

    def to_backtest_context(self) -> dict[str, object]:
        """Payload for ``BacktestSuitePlan.context['portfolio_target']``."""

        return {
            "schema_version": "portfolio_target_plan.v1",
            "decision_timestamp": self.decision_timestamp,
            "interval": self.interval,
            "target_weights": self.target_weight_map(),
            "cash_weight": self.cash_weight,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "gate_id": self.gate_id,
            "policy_lineage": self.policy_lineage,
            "model_weights": dict(self.model_weights),
            "confidence": self.confidence,
            "constraints_applied": list(self.constraints_applied),
            "metadata": dict(self.metadata),
        }


def portfolio_target_from_context(context: dict[str, object]) -> PortfolioTargetPlan:
    """Rehydrate a plan from ``to_backtest_context()`` output (round-trip check)."""

    weights_raw = context.get("target_weights", {})
    if not isinstance(weights_raw, dict):
        raise ValueError("target_weights must be a mapping")
    instruments = tuple(
        InstrumentTarget(symbol=str(symbol), weight=float(weight))
        for symbol, weight in weights_raw.items()
    )
    constraints = context.get("constraints_applied", ())
    if not isinstance(constraints, (list, tuple)):
        constraints = ()
    metadata = context.get("metadata", {})
    model_weights = context.get("model_weights", {})
    return PortfolioTargetPlan(
        decision_timestamp=str(context["decision_timestamp"]),
        interval=str(context.get("interval", "1d")),
        instrument_weights=instruments,
        cash_weight=float(context["cash_weight"]),
        gross_exposure=float(context["gross_exposure"]),
        net_exposure=float(context["net_exposure"]),
        gate_id=str(context["gate_id"]),
        policy_lineage=str(context["policy_lineage"]),
        model_weights={
            str(k): float(v)
            for k, v in model_weights.items()  # type: ignore[union-attr]
        }
        if isinstance(model_weights, dict)
        else {},
        confidence=float(context["confidence"]) if context.get("confidence") is not None else None,
        constraints_applied=tuple(str(c) for c in constraints),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )
