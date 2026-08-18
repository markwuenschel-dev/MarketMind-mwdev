"""Gate 6 promotion statistics: strategy returns → DSR/PBO/Harvey battery."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pysrc.backtesting.validation.statistical.pbo import compute_pbo
from pysrc.backtesting.validation.statistical.pbo_bridge import (
    CANONICAL_PBO_MODE,
    build_pbo_path_pairs,
)
from pysrc.backtesting.validation.statistical.report import run_validity_report
from pysrc.contracts.meta_router import DEFAULT_CANDIDATE_ID
from pysrc.pipeline.candidate_portfolios.production_bridge import (
    build_production_candidate_products,
    load_model_predictions,
    predictions_to_trade_intent_envelope,
)
from pysrc.pipeline.candidate_portfolios.simulate import simulate_candidate_portfolios
from pysrc.pipeline.candidate_portfolios.viability import (
    _attach_fold_id_to_outputs,
    _candidate_economics_by_fold,
    _capacity_limit,
    _economics_for_positions_slice,
    _panel_slice_for_simulation,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec
from pysrc.tuning.core.gates.promotion_gate import evaluate_promotion_gate

_DEFAULT_EVALUATION_COST_BPS = 10.0
_DEFAULT_N_RESAMPLES = 10_000
_DEFAULT_RANDOM_STATE = 42


def _sharpe_from_returns(returns: np.ndarray, *, periods_per_year: int = 252) -> float:
    clean = returns[np.isfinite(returns)]
    if clean.size == 0 or float(np.std(clean)) <= 0.0:
        return 0.0
    return float(np.mean(clean) / np.std(clean) * math.sqrt(periods_per_year))


def _harvey_t_stat(sharpe_annual: float, n_obs: int, *, periods_per_year: int = 252) -> float:
    if n_obs <= 0:
        return 0.0
    return float(sharpe_annual * math.sqrt(n_obs / periods_per_year))


def resolve_model_matrix_n_trials(run_dir: Path) -> int:
    """Count trained model families in a model-matrix run (excludes equal_blend comparator)."""

    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions" / "model_prediction_panel.parquet"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing predictions: {pred_path}")
    frame = pd.read_parquet(pred_path, columns=["model_id"])
    models = {str(value) for value in frame["model_id"].astype(str).unique()}
    models.discard(DEFAULT_CANDIDATE_ID)
    if not models:
        raise ValueError("No model_id values found for n_trials resolution")
    return len(models)


def extract_strategy_daily_returns(
    positions: pd.DataFrame,
    panel: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
    *,
    strategy_id: str,
    cost_bps: float | None = None,
) -> dict[str, Any]:
    """Simulate strategy-path positions and return pooled + per-fold net_return series."""

    cost = float(cost_bps if cost_bps is not None else portfolio_spec.cost_bps)
    capacity = _capacity_limit(portfolio_spec)
    pos = positions.loc[positions["candidate_id"].astype(str) == str(strategy_id)].copy()
    if pos.empty:
        raise ValueError(f"No strategy positions for candidate_id={strategy_id!r}")

    simulated = simulate_candidate_portfolios(
        pos,
        panel,
        cost_bps=cost,
        capacity_limit=capacity,
    )
    simulated = _attach_fold_id_to_outputs(simulated, pos)
    if simulated.empty:
        raise ValueError("Strategy simulation produced no return rows")

    ordered = simulated.sort_values("date", kind="mergesort")
    finite_mask = np.isfinite(ordered["net_return"].astype(float))
    pooled_frame = ordered.loc[finite_mask]
    pooled = pooled_frame["net_return"].astype(float).to_numpy()

    by_fold: dict[str, dict[str, float | int]] = {}
    if "fold_id" in simulated.columns:
        for fold_id in sorted(simulated["fold_id"].astype(str).dropna().unique().tolist()):
            fold_rows = simulated.loc[simulated["fold_id"].astype(str) == fold_id].sort_values(
                "date", kind="mergesort"
            )
            fold_returns = fold_rows["net_return"].astype(float).to_numpy()
            fold_returns = fold_returns[np.isfinite(fold_returns)]
            by_fold[str(fold_id)] = {
                "n_days": int(fold_returns.size),
                "net_sharpe": _sharpe_from_returns(fold_returns),
                "return_std": float(np.std(fold_returns)) if fold_returns.size else 0.0,
                "mean_return": float(np.mean(fold_returns)) if fold_returns.size else 0.0,
            }

    return {
        "pooled_returns": pooled,
        "pooled_dates": pooled_frame["date"].astype(str).tolist(),
        "by_fold": by_fold,
        "n_obs": int(pooled.size),
        "cost_bps": cost,
        "strategy_id": strategy_id,
    }


def build_walk_forward_pbo_surface(
    positions: pd.DataFrame,
    panel: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
    *,
    cost_bps: float | None = None,
    trial_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build walk-forward CPCV records: fold_id=path, candidate_id=trial."""

    cost = float(cost_bps if cost_bps is not None else portfolio_spec.cost_bps)
    if positions.empty or "fold_id" not in positions.columns:
        return {
            "records": [],
            "path_pairs": [],
            "pbo_result": _unavailable_pbo(),
            "rationale": "missing_fold_id_positions",
        }

    by_fold = _candidate_economics_by_fold(positions, panel, portfolio_spec, cost_bps=cost)
    fold_ids = sorted(by_fold.keys())
    if len(fold_ids) < 2:
        return {
            "records": [],
            "path_pairs": [],
            "pbo_result": _unavailable_pbo(),
            "rationale": "insufficient_fold_count_for_pbo",
        }

    candidates = sorted(
        {str(candidate_id) for fold_econ in by_fold.values() for candidate_id in fold_econ}
    )
    if trial_ids is not None:
        allowed = {str(value) for value in trial_ids}
        candidates = [candidate for candidate in candidates if candidate in allowed]
    candidates = [candidate for candidate in candidates if candidate != DEFAULT_CANDIDATE_ID]
    if len(candidates) < 2:
        return {
            "records": [],
            "path_pairs": [],
            "pbo_result": _unavailable_pbo(),
            "rationale": "insufficient_trial_count_for_pbo",
        }

    records: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        other_folds = [value for value in fold_ids if value != fold_id]
        is_econ = _economics_for_positions_slice(
            positions.loc[positions["fold_id"].astype(str).isin(other_folds)],
            panel,
            portfolio_spec,
            cost_bps=cost,
        )
        oos_econ = by_fold.get(fold_id, {})
        for candidate_id in candidates:
            is_score = float(is_econ.get(candidate_id, {}).get("net_sharpe", 0.0))
            oos_score = float(oos_econ.get(candidate_id, {}).get("net_sharpe", 0.0))
            records.append(
                {
                    "path_id": fold_id,
                    "trial_id": candidate_id,
                    "in_sample_score": is_score,
                    "out_of_sample_score": oos_score,
                }
            )

    try:
        path_pairs = build_pbo_path_pairs(records)
        pbo_result = compute_pbo(path_pairs, mode=CANONICAL_PBO_MODE)
    except Exception as exc:  # noqa: BLE001 — report must capture PBO assembly failures
        return {
            "records": records,
            "path_pairs": [],
            "pbo_result": _unavailable_pbo(error=str(exc)),
            "rationale": "pbo_assembly_failed",
        }

    return {
        "records": records,
        "path_pairs": path_pairs,
        "pbo_result": pbo_result,
        "n_paths": len(path_pairs),
        "n_trials": len(candidates),
        "rationale": "walk_forward_fold_holdout_is_vs_oos_sharpe",
    }


def _unavailable_pbo(*, error: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "value": 0.50,
        "threshold": 0.50,
        "warn_threshold": 0.40,
        "gate_result": "WARN",
        "method": "unavailable",
        "score_basis": "net_sharpe",
        "n_trials": 0,
        "n_paths": 0,
    }
    if error is not None:
        payload["error"] = error
    return payload


def run_promotion_stat_battery(
    returns: np.ndarray | Sequence[float],
    *,
    n_trials: int,
    pbo_result: dict[str, Any] | None = None,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    random_state: int = _DEFAULT_RANDOM_STATE,
    pit_ok: bool = True,
    determinism_ok: bool = True,
) -> dict[str, Any]:
    """Run Appendix H validity report and promotion gate checks."""

    stat_validity_report = run_validity_report(
        np.asarray(returns, dtype=float),
        n_trials=n_trials,
        n_resamples=n_resamples,
        random_state=random_state,
        pbo_result=pbo_result,
    )
    n_obs = int(np.asarray(returns).size)
    sharpe = float(stat_validity_report.get("sharpe_ratio", 0.0))
    dsr_value = float(stat_validity_report.get("dsr", {}).get("value", 0.0))
    t_stat = _harvey_t_stat(sharpe, n_obs)
    promotion_pass, promotion_checks = evaluate_promotion_gate(
        dsr=dsr_value,
        t_stat=t_stat,
        pit_ok=pit_ok,
        determinism_ok=determinism_ok,
    )
    return {
        "stat_validity_report": stat_validity_report,
        "promotion_gate": {
            "overall_pass": promotion_pass,
            "checks": promotion_checks,
            "harvey_t_stat": t_stat,
            "dsr_value": dsr_value,
        },
        "gate_result": stat_validity_report.get("gate_result", "WARN"),
        "gate_pass": promotion_pass,
    }


def run_gate6_promotion_for_model_matrix_run(
    run_dir: Path,
    *,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    model_id: str = "xgboost",
    target_column: str = "forward_return_horizon",
    strategy_id: str | None = None,
    evaluation_cost_bps: float = _DEFAULT_EVALUATION_COST_BPS,
) -> dict[str, Any]:
    """Run Gate 6 promotion battery and write gate6_promotion_report.json."""

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
    strategy_positions, _strategy_outputs = build_production_candidate_products(
        intents,
        panel,
        portfolio_spec,
    )

    returns_payload = extract_strategy_daily_returns(
        strategy_positions,
        panel,
        portfolio_spec,
        strategy_id=strategy_id,
        cost_bps=evaluation_cost_bps,
    )
    n_trials = resolve_model_matrix_n_trials(run_dir)

    gate3_positions_path = run_dir / "predictions" / "candidate_position_panel.parquet"
    if gate3_positions_path.is_file():
        gate3_positions = pd.read_parquet(gate3_positions_path)
    else:
        gate3_positions = strategy_positions

    pbo_all = build_walk_forward_pbo_surface(
        gate3_positions,
        panel,
        portfolio_spec,
        cost_bps=evaluation_cost_bps,
    )
    focus_trials = ("xgboost", "extra_trees", "random_forest")
    pbo_focus = build_walk_forward_pbo_surface(
        gate3_positions,
        panel,
        portfolio_spec,
        cost_bps=evaluation_cost_bps,
        trial_ids=focus_trials,
    )

    battery = run_promotion_stat_battery(
        returns_payload["pooled_returns"],
        n_trials=n_trials,
        pbo_result=pbo_all["pbo_result"],
    )

    from pysrc.pipeline.candidate_portfolios.production_backtest import (
        run_production_backtest_stat_integration,
    )

    backtest_stat = run_production_backtest_stat_integration(
        strategy_positions,
        panel,
        portfolio_spec=portfolio_spec,
        strategy_id=strategy_id,
        n_trials=n_trials,
        pbo_path_pairs=pbo_all.get("path_pairs"),
        cost_bps=evaluation_cost_bps,
    )

    report: dict[str, Any] = {
        "schema_version": "gate6_promotion_report.v1",
        "run_id": run_dir.name,
        "model_id": model_id,
        "strategy_id": strategy_id,
        "cost_bps": evaluation_cost_bps,
        "n_trials": n_trials,
        "n_obs": returns_payload["n_obs"],
        "by_fold_returns_summary": returns_payload["by_fold"],
        "stat_validity_report": battery["stat_validity_report"],
        "promotion_gate": battery["promotion_gate"],
        "pbo_surface": {
            "all_models": {
                "n_paths": pbo_all.get("n_paths", 0),
                "n_trials": pbo_all.get("n_trials", 0),
                "pbo": pbo_all["pbo_result"],
                "rationale": pbo_all.get("rationale"),
            },
            "focus_subset": {
                "trial_ids": list(focus_trials),
                "n_paths": pbo_focus.get("n_paths", 0),
                "n_trials": pbo_focus.get("n_trials", 0),
                "pbo": pbo_focus["pbo_result"],
                "rationale": pbo_focus.get("rationale"),
            },
        },
        "backtest_stat_integration": backtest_stat,
        "gate_result": battery["gate_result"],
        "gate_pass": bool(battery["gate_pass"]),
    }

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    from pysrc.artifact_registry._atomic import atomic_write_json
    from pysrc.pipeline.panel.model_matrix_diagnostics import _json_ready

    out_path = reports_dir / "gate6_promotion_report.json"
    atomic_write_json(out_path, _json_ready(report))
    report["report_path"] = str(out_path)
    return report


_RESERVED_CRISIS_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("GFC_2008", "2008-09-01", "2009-06-30"),
    ("COVID_2020", "2020-02-01", "2020-06-30"),
)


def _promotion_status(
    model_id: str,
    *,
    selected_model: str,
    pooled_rank: int,
) -> str:
    if model_id == DEFAULT_CANDIDATE_ID:
        return "comparator"
    if model_id == selected_model:
        return "selected"
    if pooled_rank <= 3:
        return "competitive"
    return "evaluated"


def build_promotion_model_ledger(
    run_dir: Path,
    *,
    selected_model: str = "xgboost",
    run_stat_battery_for: Sequence[str] | None = None,
    panel_path: Path | None = None,
    portfolio_spec: PortfolioSpec | None = None,
    evaluation_cost_bps: float = _DEFAULT_EVALUATION_COST_BPS,
    target_column: str = "forward_return_horizon",
) -> dict[str, Any]:
    """Summarize all model-matrix candidates for promotion audit."""

    run_dir = Path(run_dir)
    robustness_path = run_dir / "reports" / "gate4_robustness_report.json"
    if not robustness_path.is_file():
        raise FileNotFoundError(f"Missing Gate 4 robustness report: {robustness_path}")
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    by_fold: dict[str, dict[str, dict[str, float]]] = robustness.get("by_fold", {})

    pooled: dict[str, dict[str, float]] = {}
    for _fold_id, fold_econ in by_fold.items():
        for model_id, metrics in fold_econ.items():
            entry = pooled.setdefault(
                str(model_id),
                {
                    "cumulative_log_return": 0.0,
                    "net_sharpe_sum": 0.0,
                    "fold_count": 0,
                    "n_days": 0,
                },
            )
            entry["cumulative_log_return"] += float(metrics.get("cumulative_log_return", 0.0))
            entry["net_sharpe_sum"] += float(metrics.get("net_sharpe", 0.0))
            entry["fold_count"] += 1
            entry["n_days"] += int(metrics.get("n_days", 0))

    ranked = sorted(
        pooled.items(),
        key=lambda item: float(item[1]["net_sharpe_sum"]) / max(int(item[1]["fold_count"]), 1),
        reverse=True,
    )
    rank_lookup = {model_id: idx + 1 for idx, (model_id, _) in enumerate(ranked)}

    models: dict[str, Any] = {}
    for model_id, agg in pooled.items():
        fold_count = max(int(agg["fold_count"]), 1)
        models[model_id] = {
            "promotion_status": _promotion_status(
                model_id,
                selected_model=selected_model,
                pooled_rank=rank_lookup.get(model_id, 999),
            ),
            "pooled_rank": rank_lookup.get(model_id),
            "pooled_net_sharpe_mean": float(agg["net_sharpe_sum"]) / fold_count,
            "pooled_cumulative_log_return": float(agg["cumulative_log_return"]),
            "n_days": int(agg["n_days"]),
            "by_fold": {
                fold_id: fold_econ.get(model_id, {})
                for fold_id, fold_econ in by_fold.items()
                if model_id in fold_econ
            },
        }

    stat_battery: dict[str, Any] = {}
    if run_stat_battery_for and panel_path is not None and portfolio_spec is not None:
        from pysrc.pipeline.candidate_portfolios.production_bridge import (
            build_production_candidate_products,
            load_model_predictions,
            predictions_to_trade_intent_envelope,
        )

        n_trials = resolve_model_matrix_n_trials(run_dir)
        panel = _panel_slice_for_simulation(panel_path, target_column=target_column)
        for model_id in run_stat_battery_for:
            strategy_id = f"prediction_threshold_{model_id}"
            predictions = load_model_predictions(run_dir, model_id)
            lineage = {"run_id": run_dir.name, "model_id": model_id}
            intents = predictions_to_trade_intent_envelope(
                predictions,
                strategy_id=strategy_id,
                source_product_id="model_prediction_panel",
                lineage=lineage,
                top_k=portfolio_spec.top_k,
                source_model_id=model_id,
            )
            positions, _ = build_production_candidate_products(intents, panel, portfolio_spec)
            returns_payload = extract_strategy_daily_returns(
                positions,
                panel,
                portfolio_spec,
                strategy_id=strategy_id,
                cost_bps=evaluation_cost_bps,
            )
            pbo_all = build_walk_forward_pbo_surface(
                pd.read_parquet(run_dir / "predictions" / "candidate_position_panel.parquet"),
                panel,
                portfolio_spec,
                cost_bps=evaluation_cost_bps,
            )
            battery = run_promotion_stat_battery(
                returns_payload["pooled_returns"],
                n_trials=n_trials,
                pbo_result=pbo_all["pbo_result"],
                n_resamples=200,
            )
            stat_battery[model_id] = {
                "gate_pass": battery["gate_pass"],
                "harvey_t_stat": battery["promotion_gate"]["harvey_t_stat"],
                "dsr_value": battery["promotion_gate"]["dsr_value"],
                "sharpe_ratio": battery["stat_validity_report"].get("sharpe_ratio"),
            }

    return {
        "schema_version": "promotion_model_ledger.v1",
        "run_id": run_dir.name,
        "selected_model": selected_model,
        "evaluation_cost_bps": evaluation_cost_bps,
        "selection_rationale": robustness.get("rationale"),
        "gate4_outcome": robustness.get("upgraded_outcome"),
        "models": models,
        "ranking_by_pooled_sharpe": [
            {
                "model_id": model_id,
                "pooled_net_sharpe_mean": models[model_id]["pooled_net_sharpe_mean"],
            }
            for model_id, _ in ranked
        ],
        "optional_stat_battery": stat_battery,
    }


def build_crisis_holdout_report(
    returns: np.ndarray | Sequence[float],
    dates: Sequence[str],
) -> dict[str, Any]:
    """Exclude reserved crisis windows from promotion return series."""

    frame = pd.DataFrame(
        {"date": pd.to_datetime(dates), "net_return": np.asarray(returns, dtype=float)}
    )
    frame = frame.sort_values("date")
    excluded_mask = pd.Series(False, index=frame.index)
    windows: list[dict[str, Any]] = []
    for label, start, end in _RESERVED_CRISIS_WINDOWS:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask = (frame["date"] >= start_ts) & (frame["date"] <= end_ts)
        excluded_mask = excluded_mask | mask
        crisis_returns = frame.loc[mask, "net_return"].to_numpy()
        windows.append(
            {
                "window": label,
                "start": start,
                "end": end,
                "n_days": int(mask.sum()),
                "net_sharpe": _sharpe_from_returns(crisis_returns) if crisis_returns.size else None,
                "cumulative_log_return": float(np.sum(np.log1p(crisis_returns)))
                if crisis_returns.size
                else None,
            }
        )

    holdout = frame.loc[~excluded_mask, "net_return"].to_numpy()
    holdout = holdout[np.isfinite(holdout)]
    return {
        "schema_version": "crisis_holdout_report.v1",
        "reserved_windows": windows,
        "excluded_day_count": int(excluded_mask.sum()),
        "holdout_day_count": int(holdout.size),
        "holdout_net_sharpe": _sharpe_from_returns(holdout),
        "holdout_cumulative_log_return": float(np.sum(np.log1p(holdout))) if holdout.size else 0.0,
    }


def run_pdr001_finish_for_model_matrix_run(
    run_dir: Path,
    *,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    model_id: str = "xgboost",
    target_column: str = "forward_return_horizon",
    evaluation_cost_bps: float = _DEFAULT_EVALUATION_COST_BPS,
    run_stat_battery: bool = False,
) -> dict[str, Any]:
    """Gate 7 finish line: ledger, bundle, mm-gate validate, crisis holdout, finish report."""

    from pysrc.artifact_registry._atomic import atomic_write_json
    from pysrc.cli.gate import ExitCode, validate_bundle, write_gate_report
    from pysrc.pipeline.candidate_portfolios.production_bridge import (
        build_production_candidate_products,
        load_model_predictions,
        predictions_to_trade_intent_envelope,
    )
    from pysrc.pipeline.candidate_portfolios.promotion_bundle import (
        assemble_promotion_bundle,
    )
    from pysrc.pipeline.panel.model_matrix_diagnostics import _json_ready

    run_dir = Path(run_dir)
    reports_dir = run_dir / "reports"
    for required in (
        "gate3_viability_report.json",
        "gate5_production_smoke.json",
        "gate6_promotion_report.json",
    ):
        if not (reports_dir / required).is_file():
            raise FileNotFoundError(f"Missing prerequisite report: {reports_dir / required}")

    strategy_id = f"prediction_threshold_{model_id}"
    stat_targets = ("xgboost", "random_forest", "extra_trees") if run_stat_battery else None
    ledger = build_promotion_model_ledger(
        run_dir,
        selected_model=model_id,
        run_stat_battery_for=stat_targets,
        panel_path=panel_path if run_stat_battery else None,
        portfolio_spec=portfolio_spec if run_stat_battery else None,
        evaluation_cost_bps=evaluation_cost_bps,
        target_column=target_column,
    )
    ledger_path = reports_dir / "promotion_model_ledger.json"
    atomic_write_json(ledger_path, _json_ready(ledger))

    bundle_dir = assemble_promotion_bundle(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=portfolio_spec,
        model_id=model_id,
        strategy_id=strategy_id,
        evaluation_cost_bps=evaluation_cost_bps,
    )

    gate_report, exit_code = validate_bundle(bundle_dir)
    gate_result_path = bundle_dir / "gate_result.json"
    write_gate_report(gate_report, gate_result_path, bundle_dir)

    panel = _panel_slice_for_simulation(panel_path, target_column=target_column)
    predictions = load_model_predictions(run_dir, model_id)
    intents = predictions_to_trade_intent_envelope(
        predictions,
        strategy_id=strategy_id,
        source_product_id="model_prediction_panel",
        lineage={"run_id": run_dir.name, "model_id": model_id},
        top_k=portfolio_spec.top_k,
        source_model_id=model_id,
    )
    positions, _outputs = build_production_candidate_products(intents, panel, portfolio_spec)
    returns_payload = extract_strategy_daily_returns(
        positions,
        panel,
        portfolio_spec,
        strategy_id=strategy_id,
        cost_bps=evaluation_cost_bps,
    )
    crisis = build_crisis_holdout_report(
        returns_payload["pooled_returns"],
        returns_payload["pooled_dates"],
    )

    finish_pass = bool(
        exit_code == ExitCode.PASS
        and json.loads(
            (reports_dir / "gate6_promotion_report.json").read_text(encoding="utf-8")
        ).get("gate_pass")
    )
    finish_report: dict[str, Any] = {
        "schema_version": "pdr001_finish_report.v1",
        "run_id": run_dir.name,
        "promotion_model": model_id,
        "strategy_id": strategy_id,
        "bundle_path": str(bundle_dir),
        "gate_validate_exit_code": int(exit_code.value),
        "gate_validate_pass": exit_code == ExitCode.PASS,
        "gate_result_path": str(gate_result_path),
        "ledger_path": str(ledger_path),
        "crisis_holdout": crisis,
        "finish_pass": finish_pass,
        "pdr001_research_lane_complete": finish_pass,
    }

    manifest = {
        "schema_version": "panel_promotion_manifest.v1",
        "run_id": run_dir.name,
        "promotion_model": model_id,
        "bundle_path": str(bundle_dir),
        "gate_result_path": str(gate_result_path),
        "ledger_path": str(ledger_path),
        "finish_report_path": str(reports_dir / "pdr001_finish_report.json"),
        "pinned_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    from pysrc.pipeline.candidate_portfolios.provenance import compute_promotion_provenance_hashes

    manifest.update(compute_promotion_provenance_hashes(bundle_dir, gate_result_path))
    atomic_write_json(reports_dir / "panel_promotion_manifest.json", _json_ready(manifest))
    atomic_write_json(bundle_dir / "panel_promotion_manifest.json", _json_ready(manifest))

    finish_path = reports_dir / "pdr001_finish_report.json"
    atomic_write_json(finish_path, _json_ready(finish_report))
    finish_report["report_path"] = str(finish_path)
    finish_report["manifest_path"] = str(reports_dir / "panel_promotion_manifest.json")
    return finish_report


__all__ = [
    "build_crisis_holdout_report",
    "build_promotion_model_ledger",
    "build_walk_forward_pbo_surface",
    "extract_strategy_daily_returns",
    "resolve_model_matrix_n_trials",
    "run_gate6_promotion_for_model_matrix_run",
    "run_pdr001_finish_for_model_matrix_run",
    "run_promotion_stat_battery",
]
