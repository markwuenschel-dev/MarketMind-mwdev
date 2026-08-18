"""Gate 5 production bridge: panel predictions → strategy intents → candidate portfolios."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from pysrc.pipeline.candidate_portfolios.build_positions import predictions_to_candidate_positions
from pysrc.pipeline.candidate_portfolios.simulate import simulate_candidate_portfolios
from pysrc.pipeline.candidate_portfolios.strategy_products import (
    trade_intent_envelope_to_candidate_positions,
)
from pysrc.pipeline.candidate_portfolios.viability import (
    _attach_fold_id_to_outputs,
    _candidate_economics_by_fold,
    _capacity_limit,
    _panel_slice_for_simulation,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec
from pysrc.strategies import build_threshold_intents

_DEFAULT_PARITY_SHARPE_TOL = 0.05
_DEFAULT_PARITY_CUM_LOG_TOL = 0.5


def load_model_predictions(run_dir: Path, model_id: str) -> pd.DataFrame:
    """Load one model slice from a model-matrix prediction panel."""

    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions" / "model_prediction_panel.parquet"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing predictions: {pred_path}")
    frame = pd.read_parquet(pred_path)
    if "model_id" not in frame.columns:
        raise ValueError("model_prediction_panel missing model_id column")
    filtered = frame.loc[frame["model_id"].astype(str) == str(model_id)].copy()
    if filtered.empty:
        raise ValueError(f"No prediction rows for model_id={model_id!r}")
    return filtered


def _top_k_prediction_slice(predictions: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    required = {"date", "instrument", "prediction", "fold_id", "split"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    parts: list[pd.DataFrame] = []
    for (_date, _fold_id, _split), group in predictions.groupby(
        ["date", "fold_id", "split"], sort=True
    ):
        ranked = group.sort_values("prediction", ascending=False).head(top_k)
        if not ranked.empty:
            parts.append(ranked)
    if not parts:
        return predictions.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def predictions_to_trade_intent_envelope(
    predictions: pd.DataFrame,
    *,
    strategy_id: str,
    source_product_id: str,
    lineage: Mapping[str, str],
    top_k: int,
    threshold: float = 0.0,
    source_model_id: str | None = None,
) -> pd.DataFrame:
    """Top-k slice + threshold intents, preserving fold_id and split envelope fields."""

    sliced = _top_k_prediction_slice(predictions, top_k=top_k)
    if sliced.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for (fold_id, split), fold_group in sliced.groupby(["fold_id", "split"], sort=True):
        intent_frame = fold_group.loc[:, ["date", "instrument", "prediction"]].copy()
        intent_frame["prediction"] = (
            pd.to_numeric(intent_frame["prediction"], errors="coerce").astype(float).abs()
        )
        intent_frame = intent_frame.loc[intent_frame["prediction"] > 0.0]
        if intent_frame.empty:
            continue
        intents = build_threshold_intents(
            intent_frame,
            strategy_id=strategy_id,
            source_product_id=source_product_id,
            lineage=dict(lineage),
            threshold=threshold,
            source_model_id=source_model_id,
        )
        intents["fold_id"] = str(fold_id)
        intents["split"] = str(split)
        parts.append(intents)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_direct_candidate_products(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
    *,
    model_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gate 3 shortcut: predictions → positions → simulate (single model only)."""

    positions = predictions_to_candidate_positions(
        predictions,
        top_k=portfolio_spec.top_k,
        single_name_cap=portfolio_spec.single_name_cap,
    )
    positions = positions.loc[positions["candidate_id"].astype(str) == str(model_id)]
    outputs = simulate_candidate_portfolios(
        positions,
        panel,
        cost_bps=portfolio_spec.cost_bps,
        capacity_limit=_capacity_limit(portfolio_spec),
    )
    outputs = _attach_fold_id_to_outputs(outputs, positions)
    return positions, outputs


def build_production_candidate_products(
    intents: pd.DataFrame,
    panel: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strategy path: trade intent envelope → positions → simulate."""

    positions = trade_intent_envelope_to_candidate_positions(intents, portfolio_spec)
    outputs = simulate_candidate_portfolios(
        positions,
        panel,
        cost_bps=portfolio_spec.cost_bps,
        capacity_limit=_capacity_limit(portfolio_spec),
    )
    outputs = _attach_fold_id_to_outputs(outputs, positions)
    return positions, outputs


def build_production_parity_report(
    *,
    direct_by_fold: dict[str, dict[str, dict[str, float]]],
    strategy_by_fold: dict[str, dict[str, dict[str, float]]],
    model_id: str,
    strategy_id: str,
    sharpe_tol: float = _DEFAULT_PARITY_SHARPE_TOL,
    cum_log_tol: float = _DEFAULT_PARITY_CUM_LOG_TOL,
) -> dict[str, Any]:
    """Compare per-fold economics between direct and strategy production paths."""

    folds = sorted(set(direct_by_fold) | set(strategy_by_fold))
    per_fold: dict[str, dict[str, object]] = {}
    max_sharpe_delta = 0.0
    max_cum_log_delta = 0.0
    parity_pass = True

    for fold_id in folds:
        direct_metrics = direct_by_fold.get(fold_id, {}).get(model_id, {})
        strategy_metrics = strategy_by_fold.get(fold_id, {}).get(strategy_id, {})
        delta_sharpe = float(
            strategy_metrics.get("net_sharpe", 0.0) - direct_metrics.get("net_sharpe", 0.0)
        )
        delta_cum = float(
            strategy_metrics.get("cumulative_log_return", 0.0)
            - direct_metrics.get("cumulative_log_return", 0.0)
        )
        max_sharpe_delta = max(max_sharpe_delta, abs(delta_sharpe))
        max_cum_log_delta = max(max_cum_log_delta, abs(delta_cum))
        fold_pass = abs(delta_sharpe) <= sharpe_tol and abs(delta_cum) <= cum_log_tol
        if not fold_pass:
            parity_pass = False
        per_fold[fold_id] = {
            "direct": direct_metrics,
            "strategy": strategy_metrics,
            "delta_sharpe": delta_sharpe,
            "delta_cum_log_return": delta_cum,
            "parity_pass": fold_pass,
        }

    return {
        "schema_version": "gate5_production_parity.v1",
        "model_id": model_id,
        "strategy_id": strategy_id,
        "parity_pass": parity_pass,
        "max_abs_delta_sharpe": max_sharpe_delta,
        "max_abs_delta_cum_log_return": max_cum_log_delta,
        "sharpe_tolerance": sharpe_tol,
        "cum_log_tolerance": cum_log_tol,
        "by_fold": per_fold,
    }


def run_production_smoke_for_model_matrix_run(
    run_dir: Path,
    *,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    model_id: str = "xgboost",
    target_column: str = "forward_return_horizon",
    strategy_id: str | None = None,
) -> dict[str, Any]:
    """Run Gate 5 production smoke and write gate5_production_smoke.json."""

    run_dir = Path(run_dir)
    strategy_id = strategy_id or f"prediction_threshold_{model_id}"
    predictions = load_model_predictions(run_dir, model_id)
    panel = _panel_slice_for_simulation(panel_path, target_column=target_column)
    lineage = {
        "run_id": run_dir.name,
        "model_id": model_id,
        "source_product_id": "model_prediction_panel",
    }

    intents = predictions_to_trade_intent_envelope(
        predictions,
        strategy_id=strategy_id,
        source_product_id="model_prediction_panel",
        lineage=lineage,
        top_k=portfolio_spec.top_k,
        source_model_id=model_id,
    )
    strategy_positions, strategy_outputs = build_production_candidate_products(
        intents,
        panel,
        portfolio_spec,
    )
    direct_positions, direct_outputs = build_direct_candidate_products(
        predictions,
        panel,
        portfolio_spec,
        model_id=model_id,
    )

    strategy_by_fold = _candidate_economics_by_fold(strategy_positions, panel, portfolio_spec)
    direct_by_fold = _candidate_economics_by_fold(direct_positions, panel, portfolio_spec)
    parity = build_production_parity_report(
        direct_by_fold=direct_by_fold,
        strategy_by_fold=strategy_by_fold,
        model_id=model_id,
        strategy_id=strategy_id,
    )

    from pysrc.pipeline.candidate_portfolios.production_backtest import (
        run_production_backtest_smoke,
    )

    backtest_smoke = run_production_backtest_smoke(
        strategy_positions,
        panel,
        strategy_id=strategy_id,
    )

    report: dict[str, Any] = {
        "schema_version": "gate5_production_smoke.v1",
        "run_id": run_dir.name,
        "model_id": model_id,
        "strategy_id": strategy_id,
        "production_chain": "predictions → threshold intents → positions → simulate",
        "direct_path_economics": direct_by_fold,
        "strategy_path_economics": strategy_by_fold,
        "parity": parity,
        "backtest_smoke": backtest_smoke,
        "gate_pass": bool(parity["parity_pass"] and backtest_smoke.get("smoke_pass", False)),
    }

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    from pysrc.artifact_registry._atomic import atomic_write_json
    from pysrc.pipeline.panel.model_matrix_diagnostics import _json_ready

    out_path = reports_dir / "gate5_production_smoke.json"
    atomic_write_json(out_path, _json_ready(report))
    report["report_path"] = str(out_path)
    return report


__all__ = [
    "build_direct_candidate_products",
    "build_production_candidate_products",
    "build_production_parity_report",
    "load_model_predictions",
    "predictions_to_trade_intent_envelope",
    "run_production_smoke_for_model_matrix_run",
]
