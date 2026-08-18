from __future__ import annotations

from dataclasses import dataclass

from pysrc.backtesting.contracts.registry import resolve_engine
from pysrc.backtesting.contracts.types import BacktestResult


@dataclass
class ChainedEngine:
    engine_ids: list[str]

    def run(self, plan, data, store) -> BacktestResult:
        subresults: dict[str, float] = {}
        warnings: list[str] = []
        for engine_id in self.engine_ids:
            engine = resolve_engine(engine_id)
            result = engine.run(plan, data, store)
            subresults[engine_id] = float(result.metrics.get("total_return", 0.0))
            warnings.extend(result.warnings)
        return BacktestResult(
            metrics={"engine_count": float(len(self.engine_ids)), **subresults}, warnings=warnings
        )
