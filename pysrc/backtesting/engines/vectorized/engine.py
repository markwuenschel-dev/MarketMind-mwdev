from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.backtesting.contracts.errors import PitUnsafeInputError
from pysrc.backtesting.contracts.plan import BacktestPlan
from pysrc.backtesting.contracts.registry import (
    register_engine,
    resolve_cost_model,
    resolve_execution_model,
)
from pysrc.backtesting.contracts.types import ArtifactRef, MarketSlice
from pysrc.backtesting.contracts.types import BacktestResult as CoreBacktestResult
from pysrc.backtesting.data.pit import PITSafeDataView, PitUnsafeFrame
from pysrc.ops.mm_logkit import get_logger

LOG = get_logger(__name__)


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "num_trades": self.num_trades,
        }


def _add_signals(df: pl.DataFrame, fast_col: str, slow_col: str) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(fast_col) > pl.col(slow_col)).then(1).otherwise(-1).alias("signal")
    )


def _add_strategy_returns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("signal").shift(1) * pl.col("returns")).alias("strategy_returns")
    )


def _compute_metrics(df: pl.DataFrame) -> BacktestResult:
    returns = df["strategy_returns"].drop_nulls()
    if returns.is_empty():
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0)

    total_return = float((1 + returns).product() - 1)
    std = returns.std()
    sharpe = float(returns.mean() / std * (252**0.5)) if std and std > 0 else 0.0
    cumulative = (1 + returns).cum_prod()
    rolling_max = cumulative.cum_max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())
    wins = (returns > 0).sum()
    total = returns.len()
    win_rate = float(wins / total) if total > 0 else 0.0
    signals = df["signal"].drop_nulls()
    num_trades = int((signals.diff().abs() > 0).sum()) if signals.len() > 1 else 0
    return BacktestResult(total_return, sharpe, max_drawdown, win_rate, num_trades)


def run_backtest(df: pl.DataFrame, fast_sma: int = 20, slow_sma: int = 50) -> BacktestResult:
    fast_col = f"sma_{fast_sma}"
    slow_col = f"sma_{slow_sma}"
    required = ["returns", fast_col, slow_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _add_signals(df, fast_col, slow_col)
    df = _add_strategy_returns(df)
    return _compute_metrics(df)


class VectorizedBacktestEngine:
    def run(
        self,
        plan: BacktestPlan,
        data: PITSafeDataView | PitUnsafeFrame,
        store: BundleBacktestArtifactStore | None,
    ) -> CoreBacktestResult:
        if plan.pit_required and not isinstance(data, PITSafeDataView):
            raise PitUnsafeInputError(
                "PIT-required backtest plans must receive PITSafeDataView inputs via the DataView.as_of(T) front door."
            )

        frame = self._materialize_frame(data)
        config = plan.engine_config.params if plan.engine_config is not None else {}
        fast_sma = int(config.get("fast_sma", 5))
        slow_sma = int(config.get("slow_sma", 10))
        metrics = run_backtest(frame, fast_sma=fast_sma, slow_sma=slow_sma)

        execution_model = resolve_execution_model(plan.execution_model_id)
        cost_model = resolve_cost_model(plan.cost_model_id)
        assumptions = {
            "schema_version": "1.0.0",
            "execution_model_id": plan.execution_model_id,
            "cost_model_id": plan.cost_model_id,
            "seed": plan.seed,
        }
        if hasattr(execution_model, "to_execution_assumptions"):
            assumptions.update(execution_model.to_execution_assumptions())
        if hasattr(cost_model, "to_execution_assumptions"):
            assumptions.update(cost_model.to_execution_assumptions())
        if store is not None:
            execution_ref = store.put_json("execution_assumptions.json", assumptions)
        else:
            execution_ref = ArtifactRef(
                role="execution_assumptions", path="execution_assumptions.json"
            )

        LOG.info(
            "vectorized_backtest_complete",
            engine_id=plan.engine_id,
            determinism=plan.determinism.value,
            run_id=plan.run_id,
        )
        return CoreBacktestResult(
            metrics={key: float(value) for key, value in metrics.to_dict().items()},
            artifacts={"execution_assumptions.json": execution_ref},
            warnings=[],
            fills=[],
        )

    def _materialize_frame(self, data: PITSafeDataView | PitUnsafeFrame) -> pl.DataFrame:
        if isinstance(data, PITSafeDataView):
            knowledge_dates: Iterable[datetime] | None = data.metadata.get("knowledge_dates")
            if knowledge_dates:
                all_rows: list[dict[str, Any]] = []
                for ts in knowledge_dates:
                    if isinstance(ts, datetime):
                        query_ts = ts
                    else:
                        continue
                    market_slice: MarketSlice = data.as_of(query_ts)
                    rows = market_slice.features or market_slice.prices
                    all_rows.extend(rows)
                if all_rows:
                    return pl.DataFrame(all_rows)

            market_slice = data.as_of(datetime.now(UTC))
            rows = market_slice.features or market_slice.prices
            return pl.DataFrame(rows)
        if "frame" in data.metadata:
            return data.metadata["frame"]
        raise PitUnsafeInputError(
            "PitUnsafeFrame metadata must include an in-memory 'frame' for scaffold execution."
        )


register_engine("vectorized.sma", lambda: VectorizedBacktestEngine())
