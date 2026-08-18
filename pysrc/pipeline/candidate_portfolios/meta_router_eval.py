"""PDR-002 meta-router evaluation battery on an existing model-matrix run."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.contracts.meta_router import (
    CASH_CANDIDATE_ID,
    TRAINING_TARGET_COLUMN,
    MetaRouterConfig,
    select_feature_columns,
    validate_training_frame,
)
from pysrc.meta.gating_network import neural_gate_decisions
from pysrc.meta.local_policy_selector import (
    predict_candidate_deltas,
    route_decisions,
    train_local_policy_selector,
)
from pysrc.meta.mixture_of_experts import mixture_of_experts_decisions
from pysrc.meta.policy_selector import (
    boosted_tree_gate_decisions,
    equal_weight_decisions,
    oracle_diagnostic_decisions,
    recent_winner_selector_decisions,
    regime_lookup_gate,
    tree_gate_decisions,
    validation_weighted_blend,
)
from pysrc.meta.reptile import reptile_neural_gate_decisions
from pysrc.pipeline.candidate_portfolios.policy_bridge import (
    _capacity_limit,
    _economics_for_mask,
    _fold1_head_to_head,
    _load_candidate_positions,
    _panel_slice_for_simulation,
    build_policy_training_frame,
    decisions_use_weighted_blend,
    load_prediction_panel,
    routed_positions_from_decisions,
    routed_positions_from_weighted_decisions,
)
from pysrc.pipeline.candidate_portfolios.promotion_stats import _sharpe_from_returns
from pysrc.pipeline.candidate_portfolios.simulate import simulate_candidate_portfolios
from pysrc.pipeline.candidate_portfolios.viability import _attach_fold_id_to_outputs
from pysrc.pipeline.p2_config_loader import (
    PortfolioSpec,
    resolve_meta_router_battery_gate_ids,
    resolve_meta_router_evaluation_criteria,
)

_SCHEMA_VERSION = "meta_router_evaluation_report.v2"
_DEFAULT_REPORT_NAME = "meta_router_evaluation_report.json"

_GATE_ALIASES: dict[str, str] = {
    "equal_weight": "equal_weight_blend",
    "static_blends": "validation_weighted_blend",
    "linear_gate": "local_policy_selector",
    "recent_winner": "recent_winner_selector",
    "moe_gate": "mixture_of_experts",
}

_LEAKAGE_GATES: frozenset[str] = frozenset({"oracle_diagnostic"})

_NON_ROUTING_GATES: frozenset[str] = frozenset(
    {
        "cash",
        "oracle_diagnostic",
        "equal_weight_blend",
        "equal_weight",
        "validation_weighted_blend",
        "static_blends",
        "best_base_model",
    }
)


def _normalize_gate_id(gate_id: str) -> str:
    return _GATE_ALIASES.get(gate_id, gate_id)


def _regime_panel_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "regime_id" not in frame.columns:
        return pd.DataFrame(columns=["date", "regime"])
    panel = frame.loc[:, ["date", "regime_id"]].drop_duplicates(subset=["date"])
    return panel.rename(columns={"regime_id": "regime"})


def _canonical_test_context(
    training_frame: pd.DataFrame,
    baseline_id: str,
) -> tuple[set[str], dict[str, str], list[str]]:
    base = training_frame.loc[training_frame["candidate_id"].astype(str) == baseline_id]
    test_rows = base.loc[base["split"].astype(str) == "test", ["date", "fold_id"]].drop_duplicates(
        subset=["date"]
    )
    test_dates = set(test_rows["date"].astype(str).tolist())
    date_to_fold = test_rows.set_index("date")["fold_id"].astype(str).to_dict()
    fold_ids = sorted({f for f in date_to_fold.values() if f and f != "nan"})
    return test_dates, date_to_fold, fold_ids


def _daily_return_series(outputs: pd.DataFrame) -> np.ndarray:
    if outputs.empty:
        return np.array([], dtype=np.float64)
    daily = outputs.groupby("date", sort=True)["net_return"].first()
    net = daily.astype(float).to_numpy(dtype=np.float64)
    return net[np.isfinite(net)]


def _economics_from_returns(net: np.ndarray) -> dict[str, float | int]:
    return {
        "net_sharpe": _sharpe_from_returns(net),
        "cumulative_log_return": float(np.sum(np.log1p(net))) if net.size else 0.0,
        "n_days": int(net.size),
    }


def _label_economics(
    frame: pd.DataFrame,
    candidate_id: str,
    *,
    test_dates: set[str],
    value_column: str = "net_return",
) -> dict[str, float | int | str]:
    test = frame.loc[
        (frame["candidate_id"].astype(str) == candidate_id)
        & (frame["split"].astype(str) == "test")
        & (frame["date"].astype(str).isin(test_dates))
    ].drop_duplicates(subset=["date"])
    if value_column not in test.columns:
        value_column = TRAINING_TARGET_COLUMN
    net = test[value_column].astype(float).to_numpy()
    net = net[np.isfinite(net)]
    econ = _economics_from_returns(net)
    return {
        **econ,
        "economics_path": "label_panel",
        "n_days_requested": len(test_dates),
        "n_days_simulated": econ["n_days"],
    }


def _label_economics_by_fold(
    frame: pd.DataFrame,
    candidate_id: str,
    *,
    fold_ids: list[str],
) -> dict[str, dict[str, float | int]]:
    by_fold: dict[str, dict[str, float | int]] = {}
    for fold_id in fold_ids:
        mask = (frame["fold_id"].astype(str) == fold_id) & (frame["split"].astype(str) == "test")
        by_fold[fold_id] = _economics_for_mask(frame, candidate_id=candidate_id, mask=mask)
    return by_fold


def _fold_sharpe_from_row(row: dict[str, Any], fold_id: str) -> float:
    by_fold = row.get("by_fold")
    if not isinstance(by_fold, dict):
        return 0.0
    fold = by_fold.get(fold_id)
    if not isinstance(fold, dict):
        return 0.0
    return float(fold.get("net_sharpe", 0.0))


_DEGENERATE_ROUTING_CONCENTRATION: float = 0.95


def _apply_evaluate_fold_splits(frame: pd.DataFrame, evaluate_fold: str) -> pd.DataFrame:
    """Reassign train/test splits so the named fold is the OOS evaluation window."""

    out = frame.copy()
    folds = sorted(
        {
            str(f)
            for f in out["fold_id"].astype(str).unique()
            if str(f) not in {"", "nan", "None"}
        }
    )
    if evaluate_fold not in folds:
        return out
    idx = folds.index(evaluate_fold)
    train_folds = set(folds[:idx])
    out["split"] = np.where(
        out["fold_id"].astype(str).isin(train_folds),
        "train",
        "test",
    )
    return out


def _max_routing_concentration_pct(row: dict[str, Any], fold_id: str) -> float:
    summary = row.get("routing_summary")
    if not isinstance(summary, dict):
        return 1.0
    by_fold = summary.get("chosen_candidate_counts_by_fold")
    if not isinstance(by_fold, dict):
        return 1.0
    counts = by_fold.get(fold_id)
    if not isinstance(counts, dict) or not counts:
        return 1.0
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return 1.0
    return max(int(v) for v in counts.values()) / total


def _routing_summary_from_decisions(
    decisions: pd.DataFrame,
    *,
    date_to_fold: dict[str, str],
) -> dict[str, Any]:
    if decisions.empty:
        return {"chosen_candidate_counts_by_fold": {}}
    if "chosen_candidate_id" in decisions.columns:
        candidate_col = "chosen_candidate_id"
    elif "selected_candidate_id" in decisions.columns:
        candidate_col = "selected_candidate_id"
    else:
        return {"chosen_candidate_counts_by_fold": {}}
    by_fold: dict[str, dict[str, int]] = {}
    for fold_id in sorted({f for f in date_to_fold.values() if f and f != "nan"}):
        fold_dates = {d for d, f in date_to_fold.items() if f == fold_id}
        sub = decisions.loc[decisions["date"].astype(str).isin(fold_dates)]
        counts = sub[candidate_col].astype(str).value_counts()
        by_fold[fold_id] = {str(k): int(v) for k, v in counts.items()}
    return {"chosen_candidate_counts_by_fold": by_fold}


def _simulate_gate_economics(
    decisions: pd.DataFrame,
    *,
    gate_id: str,
    positions_by_candidate: dict[str, pd.DataFrame],
    panel_slice: pd.DataFrame,
    portfolio_spec: PortfolioSpec,
    capacity: float,
    test_dates: set[str],
    date_to_fold: dict[str, str],
    fold_ids: list[str],
    training_frame: pd.DataFrame,
) -> dict[str, Any]:
    routed = decisions.copy()
    if "chosen_candidate_id" not in routed.columns:
        routed["chosen_candidate_id"] = routed["selected_candidate_id"].astype(str)
    route_dates = set(routed["date"].astype(str).tolist())
    positions = {
        cid: pos.loc[pos["date"].astype(str).isin(route_dates)].copy()
        for cid, pos in positions_by_candidate.items()
    }
    routed_positions = (
        routed_positions_from_weighted_decisions(routed, positions)
        if decisions_use_weighted_blend(routed)
        else routed_positions_from_decisions(routed, positions)
    )
    if routed_positions.empty:
        return {
            "net_sharpe": 0.0,
            "cumulative_log_return": 0.0,
            "n_days": 0,
            "economics_path": "routed_simulation",
            "n_days_requested": len(test_dates),
            "n_days_simulated": 0,
            "by_fold": {},
            "routing_summary": _routing_summary_from_decisions(routed, date_to_fold=date_to_fold),
        }
    routed_positions["gate_id"] = gate_id
    outputs = simulate_candidate_portfolios(
        routed_positions,
        panel_slice,
        cost_bps=portfolio_spec.cost_bps,
        capacity_limit=capacity,
    )
    outputs = _attach_fold_id_to_outputs(outputs, routed_positions)
    if "fold_id" not in outputs.columns or outputs["fold_id"].isna().all():
        outputs = outputs.copy()
        outputs["fold_id"] = outputs["date"].astype(str).map(date_to_fold)
    routed_test = outputs.loc[outputs["date"].astype(str).isin(test_dates)]
    net = _daily_return_series(routed_test)
    econ = _economics_from_returns(net)

    by_fold: dict[str, dict[str, float | int]] = {}
    for fold_id in fold_ids:
        fold_dates = {d for d, f in date_to_fold.items() if f == fold_id}
        fold_out = routed_test.loc[routed_test["date"].astype(str).isin(fold_dates)]
        by_fold[fold_id] = _economics_from_returns(_daily_return_series(fold_out))

    result: dict[str, Any] = {
        **econ,
        "economics_path": "routed_simulation",
        "n_days_requested": len(test_dates),
        "n_days_simulated": econ["n_days"],
        "by_fold": by_fold,
        "routing_summary": _routing_summary_from_decisions(routed, date_to_fold=date_to_fold),
    }
    if "fold_1" in fold_ids and not routed_test.empty:
        result["fold_1_head_to_head"] = _fold1_head_to_head(
            training_frame, routed_test, fold_id="fold_1"
        )
    return result


def _local_policy_selector_decisions(
    frame: pd.DataFrame,
    config: MetaRouterConfig,
) -> pd.DataFrame:
    validate_training_frame(frame, config=config)
    selector = train_local_policy_selector(frame, config)
    predictions = predict_candidate_deltas(selector, frame)
    routed = route_decisions(predictions, config)
    out = routed.rename(columns={"chosen_candidate_id": "selected_candidate_id"})
    out["gate_id"] = "local_policy_selector"
    out["fold_id"] = "all"
    out["exposure_scale"] = 1.0
    out["abstain_probability"] = 0.0
    out["action"] = out["action"].astype(str)
    out["model_weights_json"] = "{}"
    return out


def _gate_runner(
    gate_id: str,
    frame: pd.DataFrame,
    config: MetaRouterConfig,
    regime_panel: pd.DataFrame,
) -> Callable[[], pd.DataFrame] | None:
    normalized = _normalize_gate_id(gate_id)
    if normalized == "cash":
        return None
    if normalized == "equal_weight_blend":
        return lambda: equal_weight_decisions(
            frame, gate_id=normalized, default_candidate_id=config.default_candidate_id
        )
    if normalized == "validation_weighted_blend":
        return lambda: validation_weighted_blend(
            frame, gate_id=normalized, default_candidate_id=config.default_candidate_id
        )
    if normalized == "regime_lookup":
        return lambda: regime_lookup_gate(frame, regime_panel, gate_id=normalized)
    if normalized == "recent_winner_selector":
        return lambda: recent_winner_selector_decisions(
            frame, gate_id=normalized, default_candidate_id=config.default_candidate_id
        )
    if normalized == "tree_gate":
        return lambda: tree_gate_decisions(
            frame, gate_id=normalized, random_seed=config.random_seed, config=config
        )
    if normalized == "boosted_tree_gate":
        return lambda: boosted_tree_gate_decisions(
            frame, gate_id=normalized, random_seed=config.random_seed, config=config
        )
    if normalized == "neural_gate":
        return lambda: neural_gate_decisions(
            frame, gate_id=normalized, random_seed=config.random_seed, config=config
        )
    if normalized == "reptile_neural_gate":
        return lambda: reptile_neural_gate_decisions(
            frame, regime_panel, gate_id=normalized, router_config=config
        )
    if normalized == "mixture_of_experts":
        return lambda: mixture_of_experts_decisions(
            frame, gate_id=normalized, random_seed=config.random_seed
        )
    if normalized == "local_policy_selector":
        return lambda: _local_policy_selector_decisions(frame, config)
    if normalized == "oracle_diagnostic":
        return lambda: oracle_diagnostic_decisions(frame)
    return None


def _is_routing_gate(gate_id: str) -> bool:
    return _normalize_gate_id(gate_id) not in _NON_ROUTING_GATES


def _resolve_pass_baseline_sharpe(
    results: list[dict[str, Any]],
    training_frame: pd.DataFrame,
    *,
    pass_baseline: str,
    pass_fold: str,
    test_dates: set[str],
    fold_ids: list[str],
    baseline_id: str,
) -> tuple[float, dict[str, Any]]:
    normalized = _normalize_gate_id(pass_baseline)
    if normalized == "best_base_model":
        row = next((r for r in results if r.get("gate_id") == pass_baseline), None)
        if row is not None:
            sharpe = _fold_sharpe_from_row(row, pass_fold)
            return sharpe, {
                "gate_id": pass_baseline,
                "selected_candidate_id": row.get("selected_candidate_id"),
                "pass_fold": pass_fold,
                "net_sharpe": sharpe,
                "by_fold": row.get("by_fold", {}),
            }
        best_id = ""
        best_sharpe = float("-inf")
        for candidate_id in sorted(training_frame["candidate_id"].astype(str).unique()):
            if candidate_id in {baseline_id, CASH_CANDIDATE_ID}:
                continue
            by_fold = _label_economics_by_fold(training_frame, candidate_id, fold_ids=fold_ids)
            sharpe = float(by_fold.get(pass_fold, {}).get("net_sharpe", 0.0))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_id = candidate_id
        return best_sharpe, {
            "gate_id": pass_baseline,
            "selected_candidate_id": best_id,
            "pass_fold": pass_fold,
            "net_sharpe": best_sharpe,
            "by_fold": _label_economics_by_fold(training_frame, best_id, fold_ids=fold_ids)
            if best_id
            else {},
        }
    candidate_id = normalized if normalized != pass_baseline else pass_baseline
    by_fold = _label_economics_by_fold(training_frame, candidate_id, fold_ids=fold_ids)
    sharpe = float(by_fold.get(pass_fold, {}).get("net_sharpe", 0.0))
    return sharpe, {
        "gate_id": pass_baseline,
        "selected_candidate_id": candidate_id,
        "pass_fold": pass_fold,
        "net_sharpe": sharpe,
        "by_fold": by_fold,
    }


def _evaluate_routed_gate(
    gate_id: str,
    training_frame: pd.DataFrame,
    config: MetaRouterConfig,
    regime_panel: pd.DataFrame,
    sim_context: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, dict[str, str] | None]:
    normalized = _normalize_gate_id(gate_id)
    runner = _gate_runner(gate_id, training_frame, config, regime_panel)
    if runner is None:
        return gate_id, None, {"gate_id": gate_id, "error": f"unsupported gate: {gate_id}"}
    try:
        decisions = runner()
        leakage_flagged = normalized in _LEAKAGE_GATES
        econ = _simulate_gate_economics(decisions, gate_id=normalized, **sim_context)
        row: dict[str, Any] = {
            "gate_id": gate_id,
            "path": "routed_simulation",
            **econ,
        }
        if leakage_flagged:
            row["leakage_flagged"] = True
        return gate_id, row, None
    except Exception as exc:  # noqa: BLE001 — collect per-gate failures for battery report
        return gate_id, None, {"gate_id": gate_id, "error": str(exc)}


def run_meta_router_evaluation(
    run_dir: Path,
    *,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    config: MetaRouterConfig,
    target_column: str = "forward_return_horizon",
    gate_ids: tuple[str, ...] | None = None,
    yaml_dict: dict[str, Any] | None = None,
    report_name: str | None = None,
    parallel_gates: bool = True,
) -> dict[str, Any]:
    """Evaluate gating baselines vs equal_blend on an existing model-matrix run."""

    run_dir = Path(run_dir)
    eval_criteria = resolve_meta_router_evaluation_criteria(yaml_dict)
    pass_fold = str(eval_criteria["pass_fold"])
    pass_baseline = str(eval_criteria["pass_baseline"])
    gate_list = tuple(
        gate_ids
        or (resolve_meta_router_battery_gate_ids(config, yaml_dict) if yaml_dict else None)
        or config.gating_baselines
    )
    training_frame = build_policy_training_frame(
        run_dir,
        panel_path,
        portfolio_spec,
        target_column=target_column,
        config=config,
    )
    evaluate_fold = str(eval_criteria.get("evaluate_fold", pass_fold))
    training_frame = _apply_evaluate_fold_splits(training_frame, evaluate_fold)
    feature_columns_used = select_feature_columns(
        training_frame,
        whitelist=config.state_features,
    )
    baseline_id = config.default_candidate_id
    test_dates, date_to_fold, fold_ids = _canonical_test_context(training_frame, baseline_id)
    regime_panel = _regime_panel_from_frame(training_frame)
    predictions_panel = load_prediction_panel(run_dir)
    panel_slice = _panel_slice_for_simulation(Path(panel_path), target_column=target_column)
    capacity = _capacity_limit(portfolio_spec)
    all_positions = _load_candidate_positions(run_dir, predictions_panel, portfolio_spec)
    positions_by_candidate: dict[str, pd.DataFrame] = {}
    for candidate_id in sorted(all_positions["candidate_id"].astype(str).unique().tolist()):
        positions_by_candidate[candidate_id] = all_positions.loc[
            all_positions["candidate_id"].astype(str) == candidate_id
        ].copy()

    baseline_econ = _label_economics(training_frame, baseline_id, test_dates=test_dates)
    baseline_by_fold = _label_economics_by_fold(training_frame, baseline_id, fold_ids=fold_ids)
    sim_context = {
        "positions_by_candidate": positions_by_candidate,
        "panel_slice": panel_slice,
        "portfolio_spec": portfolio_spec,
        "capacity": capacity,
        "test_dates": test_dates,
        "date_to_fold": date_to_fold,
        "fold_ids": fold_ids,
        "training_frame": training_frame,
    }

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    routed_gate_ids: list[str] = []

    for gate_id in gate_list:
        normalized = _normalize_gate_id(gate_id)
        if normalized == "cash":
            if CASH_CANDIDATE_ID in training_frame["candidate_id"].astype(str).unique():
                econ = _label_economics(training_frame, CASH_CANDIDATE_ID, test_dates=test_dates)
                econ["by_fold"] = _label_economics_by_fold(
                    training_frame, CASH_CANDIDATE_ID, fold_ids=fold_ids
                )
                results.append({"gate_id": gate_id, "path": "label_panel", **econ})
            else:
                errors.append({"gate_id": gate_id, "error": "cash candidate not in training frame"})
            continue
        if normalized == "best_base_model":
            best_id = ""
            best_sharpe = float("-inf")
            for candidate_id in sorted(training_frame["candidate_id"].astype(str).unique()):
                if candidate_id in {baseline_id, CASH_CANDIDATE_ID}:
                    continue
                econ = _label_economics(training_frame, candidate_id, test_dates=test_dates)
                if float(econ["net_sharpe"]) > best_sharpe:
                    best_sharpe = float(econ["net_sharpe"])
                    best_id = candidate_id
            if best_id:
                econ = _label_economics(training_frame, best_id, test_dates=test_dates)
                econ["by_fold"] = _label_economics_by_fold(
                    training_frame, best_id, fold_ids=fold_ids
                )
                results.append(
                    {
                        "gate_id": gate_id,
                        "path": "label_panel",
                        "selected_candidate_id": best_id,
                        **econ,
                    }
                )
            continue
        routed_gate_ids.append(gate_id)

    if routed_gate_ids:
        max_workers = min(len(routed_gate_ids), os.cpu_count() or 4)
        if parallel_gates and max_workers > 1:
            ordered: dict[str, tuple[dict[str, Any] | None, dict[str, str] | None]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _evaluate_routed_gate,
                        gate_id,
                        training_frame,
                        config,
                        regime_panel,
                        sim_context,
                    ): gate_id
                    for gate_id in routed_gate_ids
                }
                for future in as_completed(futures):
                    gate_id, row, err = future.result()
                    ordered[gate_id] = (row, err)
            for gate_id in routed_gate_ids:
                row, err = ordered[gate_id]
                if row is not None:
                    results.append(row)
                if err is not None:
                    errors.append(err)
        else:
            for gate_id in routed_gate_ids:
                _, row, err = _evaluate_routed_gate(
                    gate_id,
                    training_frame,
                    config,
                    regime_panel,
                    sim_context,
                )
                if row is not None:
                    results.append(row)
                if err is not None:
                    errors.append(err)

    eligible = [
        row
        for row in results
        if not row.get("leakage_flagged")
        and int(row.get("n_days_simulated", row.get("n_days", 0))) > 0
    ]
    ranked = sorted(eligible, key=lambda row: float(row.get("net_sharpe", 0.0)), reverse=True)
    ranked_fold_1 = sorted(
        eligible,
        key=lambda row: float(row.get("by_fold", {}).get("fold_1", {}).get("net_sharpe", 0.0)),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    baseline_sharpe = float(baseline_econ["net_sharpe"])
    beats_baseline = bool(best is not None and float(best["net_sharpe"]) >= baseline_sharpe)

    fold_1_baseline = float(baseline_by_fold.get("fold_1", {}).get("net_sharpe", 0.0))
    fold_1_best = ranked_fold_1[0] if ranked_fold_1 else None
    fold_1_best_sharpe = (
        float(fold_1_best.get("by_fold", {}).get("fold_1", {}).get("net_sharpe", 0.0))
        if fold_1_best
        else 0.0
    )
    fold_1_gap_vs_baseline = fold_1_best_sharpe - fold_1_baseline
    fold_1_beats_baseline = fold_1_best_sharpe >= fold_1_baseline

    fold_1_head_to_head = best.get("fold_1_head_to_head") if best is not None else None

    pass_baseline_sharpe, pass_baseline_econ = _resolve_pass_baseline_sharpe(
        results,
        training_frame,
        pass_baseline=pass_baseline,
        pass_fold=pass_fold,
        test_dates=test_dates,
        fold_ids=fold_ids,
        baseline_id=baseline_id,
    )
    routing_eligible = [
        row
        for row in eligible
        if _is_routing_gate(str(row.get("gate_id", "")))
        and row.get("path") == "routed_simulation"
    ]
    ranked_routing = sorted(
        routing_eligible,
        key=lambda row: _fold_sharpe_from_row(row, pass_fold),
        reverse=True,
    )
    best_routing = ranked_routing[0] if ranked_routing else None
    best_routing_sharpe = (
        _fold_sharpe_from_row(best_routing, pass_fold) if best_routing is not None else 0.0
    )
    routing_concentration = (
        _max_routing_concentration_pct(best_routing, pass_fold) if best_routing is not None else 1.0
    )
    not_degenerate = routing_concentration < _DEGENERATE_ROUTING_CONCENTRATION
    full_contract_pass = bool(
        eval_criteria["full_contract"]
        and best_routing is not None
        and best_routing_sharpe > pass_baseline_sharpe
        and not_degenerate
    )

    report: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_dir.name,
        "default_candidate_id": baseline_id,
        "evaluation_cost_bps": portfolio_spec.cost_bps,
        "feature_columns_used": feature_columns_used,
        "evaluation_criteria": eval_criteria,
        "baseline_test_economics": baseline_econ,
        "baseline_by_fold": baseline_by_fold,
        "pass_baseline_test_economics": pass_baseline_econ,
        "gate_ids_requested": list(gate_list),
        "results": results,
        "ranked_by_test_sharpe": [row["gate_id"] for row in ranked],
        "ranked_by_fold_1_sharpe": [row["gate_id"] for row in ranked_fold_1],
        "best_gate_id": best["gate_id"] if best else None,
        "best_beats_baseline_test_sharpe": beats_baseline,
        "fold_1_best_gate_id": fold_1_best["gate_id"] if fold_1_best else None,
        "fold_1_gap_vs_baseline": fold_1_gap_vs_baseline,
        "fold_1_beats_baseline": fold_1_beats_baseline,
        "fold_1_head_to_head": fold_1_head_to_head,
        "pdr002_meta_router_pass": beats_baseline,
        "best_routing_gate_id": best_routing["gate_id"] if best_routing else None,
        "best_routing_gate_sharpe": best_routing_sharpe,
        "best_routing_gate_sharpe_fold": pass_fold,
        "best_routing_gate_concentration": routing_concentration,
        "best_routing_gate_degenerate": not not_degenerate,
        "pdr002_full_contract_pass": full_contract_pass,
        "errors": errors,
    }
    out_name = report_name or _DEFAULT_REPORT_NAME
    out_path = run_dir / "reports" / out_name
    atomic_write_json(out_path, report)
    report["report_path"] = str(out_path)
    return report


__all__ = ["run_meta_router_evaluation"]
