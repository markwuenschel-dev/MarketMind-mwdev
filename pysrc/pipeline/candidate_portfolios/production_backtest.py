"""Gate 5 backtest engine smoke: candidate positions → PortfolioTargetPlan → BacktestSuiteRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.backtesting.contracts.plan import BacktestPlan, DeterminismTier, EngineConfig
from pysrc.backtesting.contracts.portfolio_target import InstrumentTarget, PortfolioTargetPlan
from pysrc.backtesting.data.pit import PitUnsafeFrame
from pysrc.backtesting.orchestration.plan import BacktestSuitePlan
from pysrc.backtesting.orchestration.suite_runner import BacktestSuiteRunner
from pysrc.pipeline.candidate_portfolios.simulate import simulate_candidate_portfolios
from pysrc.pipeline.candidate_portfolios.viability import (
    _attach_fold_id_to_outputs,
    _capacity_limit,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec


def candidate_positions_to_portfolio_target_plans(
    positions: pd.DataFrame,
    *,
    strategy_id: str,
    fold_id: str | None = None,
) -> list[PortfolioTargetPlan]:
    """Convert one day of candidate positions into PortfolioTargetPlan instances."""

    frame = positions.copy()
    if fold_id is not None:
        frame = frame.loc[frame["fold_id"].astype(str) == str(fold_id)]
    if frame.empty:
        return []

    plans: list[PortfolioTargetPlan] = []
    for date, group in frame.groupby("date", sort=True):
        weights = group.groupby("ticker", sort=True)["target_weight"].sum().astype(float)
        gross = float(weights.abs().sum())
        net = float(weights.sum())
        invested = float(weights.clip(lower=0.0).sum())
        cash_weight = max(0.0, 1.0 - invested)
        total = invested + cash_weight
        if total <= 0:
            continue
        scale = 1.0 / total
        instruments = tuple(
            InstrumentTarget(symbol=str(ticker), weight=float(weight) * scale)
            for ticker, weight in weights.items()
            if abs(float(weight)) > 0.0
        )
        cash_weight = max(0.0, 1.0 - sum(target.weight for target in instruments))
        plans.append(
            PortfolioTargetPlan(
                decision_timestamp=str(date),
                interval="1d",
                instrument_weights=instruments,
                cash_weight=cash_weight,
                gross_exposure=gross * scale,
                net_exposure=net * scale,
                gate_id="gate5_production_smoke",
                policy_lineage=strategy_id,
                metadata={"fold_id": str(group["fold_id"].iloc[0])},
            )
        )
    return plans


def _panel_to_polars_smoke_frame(
    panel: pd.DataFrame,
    *,
    fast_sma: int = 2,
    slow_sma: int = 3,
) -> pl.DataFrame:
    """Aggregate panel to daily returns plus SMA columns for ``vectorized.sma`` smoke."""

    if "forward_return_1d" not in panel.columns:
        raise ValueError("panel slice must include forward_return_1d")
    frame = panel.loc[:, ["date", "forward_return_1d"]].copy()
    frame["forward_return_1d"] = pd.to_numeric(frame["forward_return_1d"], errors="coerce").fillna(
        0.0
    )
    daily = (
        frame.groupby("date", sort=True)["forward_return_1d"]
        .mean()
        .reset_index()
        .rename(columns={"forward_return_1d": "returns"})
    )
    pl_frame = pl.from_pandas(daily)
    fast_col = f"sma_{fast_sma}"
    slow_col = f"sma_{slow_sma}"
    return pl_frame.with_columns(
        pl.col("returns").rolling_mean(fast_sma, min_periods=1).alias(fast_col),
        pl.col("returns").rolling_mean(slow_sma, min_periods=1).alias(slow_col),
    )


def run_production_backtest_smoke(
    positions: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    strategy_id: str,
    fold_id: str = "fold_0",
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Smoke-test BacktestSuiteRunner with portfolio targets derived from positions."""

    target_plans = candidate_positions_to_portfolio_target_plans(
        positions,
        strategy_id=strategy_id,
        fold_id=fold_id,
    )
    if not target_plans:
        return {
            "smoke_pass": False,
            "error": "no_portfolio_target_plans",
            "fold_id": fold_id,
        }

    sample_plan = target_plans[0]
    fast_sma = 2
    slow_sma = 3
    returns_frame = _panel_to_polars_smoke_frame(panel, fast_sma=fast_sma, slow_sma=slow_sma)
    bundle_root = Path(work_dir or Path.cwd() / "artifacts" / "gate5_backtest_smoke")
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_root / f"{strategy_id}_{fold_id}"

    plan = BacktestPlan(
        engine_id="vectorized.sma",
        execution_model_id="fill.identity",
        cost_model_id="fees.zero",
        ledger_id="ledger.simple",
        validator_ids=["mechanical.v1"],
        determinism=DeterminismTier.D1,
        seed=42,
        pit_required=False,
        engine_config=EngineConfig(params={"fast_sma": fast_sma, "slow_sma": slow_sma}),
        run_id=f"gate5-smoke-{fold_id}",
    )
    store = BundleBacktestArtifactStore(BundleWriter(bundle_path))
    suite_plan = BacktestSuitePlan(
        plans=[plan],
        bundle_path=str(bundle_path),
        store=store,
        context={
            "data": PitUnsafeFrame(payload_ref="panel_slice", metadata={"frame": returns_frame}),
            "portfolio_target": sample_plan.to_backtest_context(),
            "portfolio_target_count": len(target_plans),
        },
    )

    try:
        bundle_ref = BacktestSuiteRunner().execute(suite_plan)
        smoke_pass = bundle_ref.run_id == plan.run_id
        return {
            "smoke_pass": smoke_pass,
            "fold_id": fold_id,
            "bundle_path": str(bundle_path),
            "run_id": bundle_ref.run_id,
            "portfolio_target_count": len(target_plans),
            "sample_decision_timestamp": sample_plan.decision_timestamp,
        }
    except Exception as exc:  # noqa: BLE001 — smoke report must capture engine failures
        return {
            "smoke_pass": False,
            "fold_id": fold_id,
            "error": str(exc),
            "portfolio_target_count": len(target_plans),
        }


def _strategy_net_returns(
    positions: pd.DataFrame,
    panel: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
    *,
    strategy_id: str,
    cost_bps: float,
) -> list[float]:
    """Weight-path replay returns (same simulate path as Gate 5/6 promotion)."""

    pos = positions.loc[positions["candidate_id"].astype(str) == str(strategy_id)].copy()
    if pos.empty:
        return []
    simulated = simulate_candidate_portfolios(
        pos,
        panel,
        cost_bps=cost_bps,
        capacity_limit=_capacity_limit(portfolio_spec),
    )
    simulated = _attach_fold_id_to_outputs(simulated, pos)
    ordered = simulated.sort_values("date", kind="mergesort")
    return [float(value) for value in ordered["net_return"].astype(float).tolist()]


def _returns_to_polars_smoke_frame(
    returns: list[float], *, fast_sma: int = 2, slow_sma: int = 3
) -> pl.DataFrame:
    pl_frame = pl.DataFrame({"returns": returns})
    fast_col = f"sma_{fast_sma}"
    slow_col = f"sma_{slow_sma}"
    return pl_frame.with_columns(
        pl.col("returns").rolling_mean(fast_sma, min_periods=1).alias(fast_col),
        pl.col("returns").rolling_mean(slow_sma, min_periods=1).alias(slow_col),
    )


def run_production_backtest_stat_integration(
    positions: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    portfolio_spec: PortfolioSpec,
    strategy_id: str,
    n_trials: int,
    pbo_path_pairs: list[dict[str, Any]] | None = None,
    cost_bps: float = 10.0,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Run BacktestSuiteRunner with statistical.v1 on promotion-path returns."""

    returns = _strategy_net_returns(
        positions,
        panel,
        portfolio_spec,
        strategy_id=strategy_id,
        cost_bps=cost_bps,
    )
    if not returns:
        return {
            "integration_pass": False,
            "error": "no_strategy_returns",
        }

    target_plans = candidate_positions_to_portfolio_target_plans(
        positions,
        strategy_id=strategy_id,
    )
    sample_plan = target_plans[0] if target_plans else None
    fast_sma = 2
    slow_sma = 3
    returns_frame = _returns_to_polars_smoke_frame(returns, fast_sma=fast_sma, slow_sma=slow_sma)
    bundle_root = Path(work_dir or Path.cwd() / "artifacts" / "gate6_backtest_stat")
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_root / strategy_id

    plan = BacktestPlan(
        engine_id="vectorized.sma",
        execution_model_id="fill.identity",
        cost_model_id="fees.zero",
        ledger_id="ledger.simple",
        validator_ids=["statistical.v1"],
        determinism=DeterminismTier.D1,
        seed=42,
        pit_required=False,
        engine_config=EngineConfig(params={"fast_sma": fast_sma, "slow_sma": slow_sma}),
        run_id=f"gate6-stat-{strategy_id}",
    )
    store = BundleBacktestArtifactStore(BundleWriter(bundle_path))
    context: dict[str, Any] = {
        "data": PitUnsafeFrame(payload_ref="promotion_returns", metadata={"frame": returns_frame}),
        "returns": returns,
        "n_trials": n_trials,
    }
    if pbo_path_pairs:
        context["pbo_path_pairs"] = pbo_path_pairs
    if sample_plan is not None:
        context["portfolio_target"] = sample_plan.to_backtest_context()
        context["portfolio_target_count"] = len(target_plans)

    suite_plan = BacktestSuitePlan(
        plans=[plan],
        bundle_path=str(bundle_path),
        store=store,
        context=context,
    )

    try:
        bundle_ref = BacktestSuiteRunner().execute(suite_plan)
        stat_path = bundle_path / "stat_validity_report.json"
        integration_pass = bundle_ref.run_id == plan.run_id and stat_path.is_file()
        return {
            "integration_pass": integration_pass,
            "bundle_path": str(bundle_path),
            "run_id": bundle_ref.run_id,
            "n_returns": len(returns),
            "stat_validity_report_path": str(stat_path) if stat_path.is_file() else None,
        }
    except Exception as exc:  # noqa: BLE001 — integration report must capture failures
        return {
            "integration_pass": False,
            "error": str(exc),
            "n_returns": len(returns),
        }


__all__ = [
    "candidate_positions_to_portfolio_target_plans",
    "run_production_backtest_smoke",
    "run_production_backtest_stat_integration",
]
