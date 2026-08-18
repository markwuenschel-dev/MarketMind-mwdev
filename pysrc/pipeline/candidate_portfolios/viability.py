"""Gate 3 candidate portfolio viability and Gate 4 economic summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pysrc.contracts.meta_router import DEFAULT_CANDIDATE_ID
from pysrc.pipeline.candidate_portfolios import (
    build_candidate_portfolio_products,
    write_candidate_portfolio_products,
)
from pysrc.pipeline.candidate_portfolios.simulate import simulate_candidate_portfolios
from pysrc.pipeline.p2_config_loader import PortfolioSpec
from pysrc.portfolio.labels import FORWARD_RETURN_COLUMN

_DEFAULT_FOCUS_CANDIDATES: tuple[str, ...] = ("xgboost", DEFAULT_CANDIDATE_ID, "random_forest")
_DEFAULT_COST_BPS_LEVELS: tuple[float, ...] = (5.0, 10.0, 20.0)


def _candidate_economics(
    outputs: pd.DataFrame, *, split: str | None = None
) -> dict[str, dict[str, float]]:
    frame = outputs.copy()
    if split is not None and "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str) == split]
    if frame.empty:
        return {}
    out: dict[str, dict[str, float]] = {}
    for candidate_id, group in frame.groupby("model_id", sort=True):
        net = group["net_return"].astype(float).to_numpy()
        net = net[np.isfinite(net)]
        if net.size == 0:
            continue
        cum_log_return = float(np.sum(np.log1p(net)))
        sharpe = float(np.mean(net) / np.std(net) * np.sqrt(252)) if np.std(net) > 0 else 0.0
        equity = np.cumprod(1.0 + net)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.maximum(peak, 1e-12)
        max_dd = float(drawdown.max() * 100.0) if drawdown.size else 0.0
        turnover = (
            float(group["turnover"].astype(float).mean()) if "turnover" in group.columns else 0.0
        )
        out[str(candidate_id)] = {
            "cumulative_log_return": cum_log_return,
            "net_sharpe": sharpe,
            "max_drawdown_pct": max_dd,
            "mean_turnover": turnover,
            "n_days": int(len(group)),
        }
    return out


def _capacity_limit(portfolio_spec: PortfolioSpec) -> float:
    return 1.0 if portfolio_spec.capacity_constraints else 10.0


def _beats_baseline(metrics: dict[str, float], baseline: dict[str, float]) -> bool:
    return bool(
        metrics.get("net_sharpe", 0.0) > baseline.get("net_sharpe", 0.0)
        and metrics.get("cumulative_log_return", 0.0) > baseline.get("cumulative_log_return", 0.0)
    )


def _economics_for_positions_slice(
    positions: pd.DataFrame,
    panel: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
    *,
    cost_bps: float | None = None,
) -> dict[str, dict[str, float]]:
    if positions.empty:
        return {}
    cost = float(portfolio_spec.cost_bps if cost_bps is None else cost_bps)
    capacity = _capacity_limit(portfolio_spec)
    out: dict[str, dict[str, float]] = {}
    for candidate_id, pos in positions.groupby("candidate_id", sort=True):
        simulated = simulate_candidate_portfolios(
            pos,
            panel,
            cost_bps=cost,
            capacity_limit=capacity,
        )
        econ = _candidate_economics(simulated)
        key = str(candidate_id)
        if key in econ:
            out[key] = econ[key]
    return out


def _candidate_economics_by_fold(
    positions: pd.DataFrame,
    panel: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
    *,
    cost_bps: float | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    if positions.empty or "fold_id" not in positions.columns:
        return {}
    by_fold: dict[str, dict[str, dict[str, float]]] = {}
    for fold_id in sorted(positions["fold_id"].astype(str).unique().tolist()):
        fold_pos = positions.loc[positions["fold_id"].astype(str) == fold_id]
        by_fold[str(fold_id)] = _economics_for_positions_slice(
            fold_pos,
            panel,
            portfolio_spec,
            cost_bps=cost_bps,
        )
    return by_fold


def _attach_fold_id_to_outputs(outputs: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    if outputs.empty or positions.empty or "fold_id" not in positions.columns:
        return outputs
    fold_keys = (
        positions.groupby(["candidate_id", "date"], sort=True)["fold_id"].first().reset_index()
    )
    merged = outputs.merge(
        fold_keys,
        left_on=["model_id", "date"],
        right_on=["candidate_id", "date"],
        how="left",
    )
    return merged.drop(columns=["candidate_id"], errors="ignore")


def _attach_split_to_outputs(outputs: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    if outputs.empty or positions.empty or "split" not in positions.columns:
        return outputs
    keys = positions.groupby(["candidate_id", "date"], sort=True)["split"].first().reset_index()
    merged = outputs.merge(
        keys,
        left_on=["model_id", "date"],
        right_on=["candidate_id", "date"],
        how="left",
    )
    return merged.drop(columns=["candidate_id"], errors="ignore")


def build_gate3_viability_report(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build positions, simulate paths, and emit Gate 3 viability metadata."""

    expression_failures: list[dict[str, str]] = []
    try:
        positions, outputs = build_candidate_portfolio_products(predictions, panel, portfolio_spec)
        outputs = _attach_split_to_outputs(outputs, positions)
        outputs = _attach_fold_id_to_outputs(outputs, positions)
    except Exception as exc:  # noqa: BLE001 — viability report must capture structural failures
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "schema_version": "gate3_viability_report.v1",
                "expression_failures": [{"stage": "build", "error": str(exc)}],
                "gate_pass": False,
                "routing_ready_count": 0,
                "candidate_count": 0,
                "candidates": {},
            },
        )

    candidates = sorted(positions["candidate_id"].astype(str).unique().tolist())
    economics = _candidate_economics(outputs)
    routing_ready: list[str] = []
    candidate_docs: dict[str, dict[str, object]] = {}
    for candidate_id in candidates:
        pos = positions.loc[positions["candidate_id"].astype(str) == candidate_id]
        out = outputs.loc[outputs["model_id"].astype(str) == candidate_id]
        status = "routing-ready"
        if pos.empty or out.empty:
            status = "expression-failed"
            expression_failures.append({"candidate_id": candidate_id, "stage": "empty_panel"})
        elif not np.isfinite(out["net_return"].astype(float)).any():
            status = "expression-failed"
            expression_failures.append({"candidate_id": candidate_id, "stage": "nonfinite_returns"})
        else:
            routing_ready.append(candidate_id)
        candidate_docs[candidate_id] = {
            "status": status,
            **economics.get(candidate_id, {}),
        }

    report: dict[str, Any] = {
        "schema_version": "gate3_viability_report.v1",
        "expression_failures": expression_failures,
        "expression_failure_count": len(expression_failures),
        "gate_pass": len(expression_failures) == 0 and len(routing_ready) == len(candidates),
        "routing_ready_count": len(routing_ready),
        "candidate_count": len(candidates),
        "candidates": candidate_docs,
        "baseline_candidate_id": DEFAULT_CANDIDATE_ID,
    }
    return positions, outputs, report


def build_gate4_panel_validation_summary(
    outputs: pd.DataFrame,
    *,
    baseline_candidate_id: str = DEFAULT_CANDIDATE_ID,
    test_split: str = "test",
) -> dict[str, Any]:
    """Compare candidates vs equal_blend on test folds (Gate 4 minimum)."""

    if "split" not in outputs.columns:
        test_outputs = outputs
        split_used = None
    else:
        test_outputs = outputs.loc[outputs["split"].astype(str) == test_split]
        split_used = test_split

    economics = _candidate_economics(test_outputs, split=None)
    baseline = economics.get(baseline_candidate_id)
    comparisons: dict[str, dict[str, object]] = {}
    wins_vs_baseline = 0
    for candidate_id, metrics in economics.items():
        if candidate_id == baseline_candidate_id:
            continue
        delta_sharpe = None
        delta_cum = None
        if baseline is not None:
            delta_sharpe = float(metrics["net_sharpe"] - baseline["net_sharpe"])
            delta_cum = float(metrics["cumulative_log_return"] - baseline["cumulative_log_return"])
            if delta_sharpe > 0 and delta_cum > 0:
                wins_vs_baseline += 1
        comparisons[candidate_id] = {
            **metrics,
            "delta_sharpe_vs_baseline": delta_sharpe,
            "delta_cum_log_return_vs_baseline": delta_cum,
        }

    model_candidates = [cid for cid in economics if cid != baseline_candidate_id]
    if wins_vs_baseline == 0:
        outcome = "A"
        rationale = (
            "No candidate beats equal_blend on both Sharpe and cumulative log return on test folds."
        )
    elif wins_vs_baseline == 1:
        outcome = "C"
        rationale = "Single narrow winner vs equal_blend — robustness battery recommended."
    elif wins_vs_baseline >= max(2, len(model_candidates) // 2):
        outcome = "D"
        rationale = "Multiple candidates beat equal_blend on test folds."
    else:
        outcome = "B"
        rationale = "Near-miss — some metric improvement but not a clear product win."

    return {
        "schema_version": "gate4_panel_validation.v1",
        "split": split_used,
        "baseline_candidate_id": baseline_candidate_id,
        "outcome": outcome,
        "rationale": rationale,
        "wins_vs_baseline_count": wins_vs_baseline,
        "baseline": baseline,
        "candidates": comparisons,
        "all_splits_economics": economics,
    }


def _outcome_from_fold_wins(
    fold_win_count: int,
    *,
    focus_candidate: str,
) -> tuple[str, str]:
    if fold_win_count >= 2:
        return (
            "D",
            f"{focus_candidate} beats equal_blend on Sharpe and cumulative log return in "
            f"{fold_win_count} walk-forward folds at 10 bps.",
        )
    if fold_win_count == 1:
        return (
            "C",
            f"Single-fold win for {focus_candidate} vs equal_blend at 10 bps — narrow product signal.",
        )
    return (
        "A",
        f"No fold-level win for {focus_candidate} vs equal_blend at 10 bps.",
    )


def build_gate4_robustness_report(
    positions: pd.DataFrame,
    panel: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
    *,
    baseline_candidate_id: str = DEFAULT_CANDIDATE_ID,
    focus_candidate: str = "xgboost",
    focus_candidates: tuple[str, ...] = _DEFAULT_FOCUS_CANDIDATES,
    cost_bps_levels: tuple[float, ...] = _DEFAULT_COST_BPS_LEVELS,
    evaluation_cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Per-fold and cost-sensitivity robustness battery for Gate 4 outcome C."""

    by_fold = _candidate_economics_by_fold(positions, panel, portfolio_spec)
    by_fold_at_eval_cost = _candidate_economics_by_fold(
        positions,
        panel,
        portfolio_spec,
        cost_bps=evaluation_cost_bps,
    )

    vs_baseline_by_fold: dict[str, dict[str, dict[str, float | None]]] = {}
    fold_win_count = 0
    for fold_id, econ in by_fold_at_eval_cost.items():
        baseline = econ.get(baseline_candidate_id, {})
        fold_compare: dict[str, dict[str, float | None]] = {}
        for candidate_id, metrics in econ.items():
            if candidate_id == baseline_candidate_id:
                continue
            delta_sharpe: float | None = None
            delta_cum: float | None = None
            if baseline:
                delta_sharpe = float(metrics["net_sharpe"] - baseline["net_sharpe"])
                delta_cum = float(
                    metrics["cumulative_log_return"] - baseline["cumulative_log_return"]
                )
            fold_compare[candidate_id] = {
                "delta_sharpe_vs_baseline": delta_sharpe,
                "delta_cum_log_return_vs_baseline": delta_cum,
                **metrics,
            }
            if candidate_id == focus_candidate and baseline and _beats_baseline(metrics, baseline):
                fold_win_count += 1
        vs_baseline_by_fold[fold_id] = fold_compare

    cost_sensitivity: dict[str, dict[str, dict[str, float]]] = {}
    for cost_bps in cost_bps_levels:
        focus_pos = positions.loc[
            positions["candidate_id"].astype(str).isin([str(c) for c in focus_candidates])
        ]
        cost_sensitivity[str(cost_bps)] = _economics_for_positions_slice(
            focus_pos,
            panel,
            portfolio_spec,
            cost_bps=float(cost_bps),
        )

    upgraded_outcome, rationale = _outcome_from_fold_wins(
        fold_win_count,
        focus_candidate=focus_candidate,
    )

    return {
        "schema_version": "gate4_robustness_report.v1",
        "evaluation_unit": "fold_id",
        "split_note": "model_matrix emits OOS rows only; split=test is expected",
        "baseline_candidate_id": baseline_candidate_id,
        "focus_candidate": focus_candidate,
        "evaluation_cost_bps": evaluation_cost_bps,
        "by_fold": by_fold,
        "vs_baseline_by_fold": vs_baseline_by_fold,
        "cost_sensitivity": cost_sensitivity,
        "fold_win_count": fold_win_count,
        "upgraded_outcome": upgraded_outcome,
        "rationale": rationale,
    }


def run_gate4_robustness_for_model_matrix_run(
    run_dir: Path,
    *,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    target_column: str = "forward_return_horizon",
    rebuild_gate3: bool = False,
) -> dict[str, Path]:
    """Load or build Gate 3 positions, run robustness battery, write report."""

    run_dir = Path(run_dir)
    positions_path = run_dir / "predictions" / "candidate_position_panel.parquet"
    if rebuild_gate3 or not positions_path.is_file():
        run_gate3_viability_for_model_matrix_run(
            run_dir,
            panel_path=panel_path,
            portfolio_spec=portfolio_spec,
            target_column=target_column,
        )
    if not positions_path.is_file():
        raise FileNotFoundError(f"Missing candidate positions: {positions_path}")

    positions = pd.read_parquet(positions_path)
    panel = _panel_slice_for_simulation(panel_path, target_column=target_column)
    report = build_gate4_robustness_report(positions, panel, portfolio_spec)

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    from pysrc.artifact_registry._atomic import atomic_write_json
    from pysrc.pipeline.panel.model_matrix_diagnostics import _json_ready

    robustness_path = reports_dir / "gate4_robustness_report.json"
    atomic_write_json(robustness_path, _json_ready(report))

    gate4_path = reports_dir / "gate4_panel_validation.json"
    if (run_dir / "diagnostics" / "candidate_portfolio_output_panel.parquet").is_file():
        outputs = pd.read_parquet(
            run_dir / "diagnostics" / "candidate_portfolio_output_panel.parquet"
        )
        outputs = _attach_fold_id_to_outputs(
            _attach_split_to_outputs(outputs, positions),
            positions,
        )
        summary = build_gate4_panel_validation_summary(outputs)
        summary["robustness_outcome"] = report["upgraded_outcome"]
        summary["fold_win_count"] = report["fold_win_count"]
        atomic_write_json(gate4_path, _json_ready(summary))

    return {
        "gate4_robustness_report": robustness_path,
        "gate4_panel_validation": gate4_path,
    }


def _panel_slice_for_simulation(panel_path: Path, *, target_column: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    schema = pq.read_schema(panel_path)
    available = {field.name for field in schema}
    read_cols = ["date", "instrument"]
    return_col = target_column
    if return_col not in available:
        for fallback in (
            "forward_return_horizon",
            "forward_return_1d",
            "adjusted_return_1d",
            "forward_return",
        ):
            if fallback in available:
                return_col = fallback
                break
        else:
            raise ValueError(f"Panel missing forward-return column; have {sorted(available)[:20]}")
    read_cols.append(return_col)
    panel = pd.read_parquet(panel_path, columns=read_cols)
    if return_col != FORWARD_RETURN_COLUMN:
        panel[FORWARD_RETURN_COLUMN] = panel[return_col]
    return panel


def run_gate3_viability_for_model_matrix_run(
    run_dir: Path,
    *,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    target_column: str = "forward_return_horizon",
) -> dict[str, Path]:
    """Load model-matrix predictions, build Gate 3 products, write reports."""

    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions" / "model_prediction_panel.parquet"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing predictions: {pred_path}")
    if not panel_path.is_file():
        raise FileNotFoundError(f"Missing panel: {panel_path}")

    predictions = pd.read_parquet(pred_path)
    panel = _panel_slice_for_simulation(panel_path, target_column=target_column)
    positions, outputs, gate3_report = build_gate3_viability_report(
        predictions, panel, portfolio_spec
    )
    if gate3_report.get("expression_failure_count", 0) > 0:
        paths: dict[str, Path] = {}
    else:
        paths = write_candidate_portfolio_products(run_dir, positions, outputs)

    gate4_summary = build_gate4_panel_validation_summary(outputs) if not outputs.empty else {}

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    from pysrc.artifact_registry._atomic import atomic_write_json
    from pysrc.pipeline.panel.model_matrix_diagnostics import _json_ready

    gate3_path = reports_dir / "gate3_viability_report.json"
    gate4_path = reports_dir / "gate4_panel_validation.json"
    atomic_write_json(gate3_path, _json_ready(gate3_report))
    atomic_write_json(gate4_path, _json_ready(gate4_summary))
    paths["gate3_viability_report"] = gate3_path
    paths["gate4_panel_validation"] = gate4_path
    return paths


__all__ = [
    "build_gate3_viability_report",
    "build_gate4_panel_validation_summary",
    "build_gate4_robustness_report",
    "run_gate3_viability_for_model_matrix_run",
    "run_gate4_robustness_for_model_matrix_run",
]
