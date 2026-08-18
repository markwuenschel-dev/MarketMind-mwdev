from __future__ import annotations

from dataclasses import dataclass, field

from pysrc.backtesting.contracts.plan import BacktestPlan


@dataclass(frozen=True)
class BacktestSuitePlan:
    plans: list[BacktestPlan]
    bundle_path: str = ""
    store: object | None = None
    context: dict[str, object] = field(default_factory=dict)
