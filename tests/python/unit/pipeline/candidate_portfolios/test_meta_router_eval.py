"""Tests for PDR-002 meta-router evaluation battery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysrc.contracts.meta_router import DEFAULT_CANDIDATE_ID, MetaRouterConfig
from pysrc.pipeline.candidate_portfolios.meta_router_eval import run_meta_router_evaluation
from pysrc.pipeline.p2_config_loader import PortfolioSpec, resolve_meta_router_battery_gate_ids

_YAML_STATE_FEATURES: tuple[str, ...] = (
    "state_market_ret_1d",
    "state_dispersion",
    "state_market_trend_20",
    "state_market_vol_20",
    "cand_recent_utility_20",
)


def _write_predictions(run_dir: Path, *, n_days: int = 4) -> None:
    if n_days <= 4:
        fold_plan: list[tuple[str, tuple[str, ...]]] = [
            ("fold_0", ("2024-01-02", "2024-01-03")),
            ("fold_1", ("2024-02-02", "2024-02-03")),
        ]
    else:
        dates = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
        by_fold: dict[str, list[str]] = {"fold_0": [], "fold_1": [], "fold_2": []}
        for idx, date in enumerate(dates):
            if idx < n_days // 3:
                by_fold["fold_0"].append(date)
            elif idx < 2 * n_days // 3:
                by_fold["fold_1"].append(date)
            else:
                by_fold["fold_2"].append(date)
        fold_plan = [(fid, tuple(ds)) for fid, ds in by_fold.items() if ds]

    rows: list[dict[str, object]] = []
    for fold_id, dates in fold_plan:
        for date in dates:
            for model_id, pred in (("xgboost", 0.2), ("equal_blend", 0.15)):
                rows.append(
                    {
                        "date": date,
                        "instrument": "AAA",
                        "model_id": model_id,
                        "prediction": pred,
                        "fold_id": fold_id,
                        "split": "test",
                    }
                )
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(pred_dir / "model_prediction_panel.parquet", index=False)


def _panel_for_predictions(n_days: int) -> pd.DataFrame:
    if n_days <= 4:
        return pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-02-02", "2024-02-03"],
                "instrument": ["AAA"] * 4,
                "forward_return_1d": [0.01, 0.02, -0.01, 0.015],
            }
        )
    dates = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame(
        {
            "date": dates,
            "instrument": ["AAA"] * n_days,
            "forward_return_1d": np.linspace(-0.01, 0.02, n_days),
        }
    )


@pytest.mark.determinism("d1")
def test_meta_router_eval_writes_report(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _write_predictions(run_dir)
    panel_path = tmp_path / "panel.parquet"
    _panel_for_predictions(4).to_parquet(panel_path, index=False)
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    config = MetaRouterConfig(
        default_candidate_id=DEFAULT_CANDIDATE_ID,
        gating_baselines=("equal_weight_blend", "validation_weighted_blend"),
        random_seed=42,
    )
    result = run_meta_router_evaluation(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=spec,
        config=config,
        target_column="forward_return_1d",
    )
    assert result["schema_version"] == "meta_router_evaluation_report.v2"
    assert len(result["results"]) >= 2
    report = json.loads(
        (run_dir / "reports" / "meta_router_evaluation_report.json").read_text(encoding="utf-8")
    )
    assert report["default_candidate_id"] == DEFAULT_CANDIDATE_ID
    assert "ranked_by_test_sharpe" in report
    assert "baseline_by_fold" in report


@pytest.mark.determinism("d1")
def test_meta_router_eval_fold_scoping(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _write_predictions(run_dir, n_days=60)
    panel_path = tmp_path / "panel.parquet"
    _panel_for_predictions(60).to_parquet(panel_path, index=False)
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    config = MetaRouterConfig(
        default_candidate_id=DEFAULT_CANDIDATE_ID,
        state_features=_YAML_STATE_FEATURES,
        gating_baselines=("equal_weight_blend", "tree_gate"),
        random_seed=42,
    )
    result = run_meta_router_evaluation(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=spec,
        config=config,
        target_column="forward_return_1d",
    )
    baseline_days = int(result["baseline_test_economics"]["n_days_simulated"])
    for row in result["results"]:
        if row.get("economics_path") == "routed_simulation":
            assert int(row["n_days_simulated"]) == baseline_days


@pytest.mark.determinism("d1")
def test_meta_router_eval_linear_gate_runs(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _write_predictions(run_dir, n_days=60)
    panel_path = tmp_path / "panel.parquet"
    _panel_for_predictions(60).to_parquet(panel_path, index=False)
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    config = MetaRouterConfig(
        default_candidate_id=DEFAULT_CANDIDATE_ID,
        state_features=_YAML_STATE_FEATURES,
        gating_baselines=("linear_gate",),
        random_seed=42,
    )
    result = run_meta_router_evaluation(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=spec,
        config=config,
        target_column="forward_return_1d",
    )
    gate_ids = {row["gate_id"] for row in result["results"]}
    assert "linear_gate" in gate_ids
    assert not any(err["gate_id"] == "linear_gate" for err in result["errors"])


@pytest.mark.determinism("d1")
def test_meta_router_eval_oracle_excluded_from_pass(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _write_predictions(run_dir, n_days=60)
    panel_path = tmp_path / "panel.parquet"
    _panel_for_predictions(60).to_parquet(panel_path, index=False)
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    config = MetaRouterConfig(
        default_candidate_id=DEFAULT_CANDIDATE_ID,
        state_features=_YAML_STATE_FEATURES,
        gating_baselines=("oracle_diagnostic", "equal_weight_blend"),
        random_seed=42,
    )
    result = run_meta_router_evaluation(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=spec,
        config=config,
        target_column="forward_return_1d",
    )
    assert result["best_gate_id"] != "oracle_diagnostic"
    oracle_rows = [r for r in result["results"] if r["gate_id"] == "oracle_diagnostic"]
    assert oracle_rows
    assert oracle_rows[0].get("leakage_flagged") is True


@pytest.mark.determinism("d1")
def test_mixture_of_experts_gate_runs_in_battery(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _write_predictions(run_dir, n_days=60)
    panel_path = tmp_path / "panel.parquet"
    _panel_for_predictions(60).to_parquet(panel_path, index=False)
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    config = MetaRouterConfig(
        default_candidate_id=DEFAULT_CANDIDATE_ID,
        state_features=_YAML_STATE_FEATURES,
        gating_baselines=("mixture_of_experts",),
        random_seed=42,
    )
    result = run_meta_router_evaluation(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=spec,
        config=config,
        target_column="forward_return_1d",
    )
    gate_ids = {row["gate_id"] for row in result["results"]}
    assert "mixture_of_experts" in gate_ids
    assert not any(err["gate_id"] == "mixture_of_experts" for err in result["errors"])


@pytest.mark.determinism("d1")
def test_resolve_meta_router_battery_gate_ids_merges_comparators(deterministic_seed: int) -> None:
    _ = deterministic_seed
    config = MetaRouterConfig(gating_baselines=("equal_weight_blend",))
    yaml_dict = {"evaluation": {"comparators": ["oracle_diagnostic", "best_base_model"]}}
    gates = resolve_meta_router_battery_gate_ids(config, yaml_dict)
    assert gates == ("equal_weight_blend", "oracle_diagnostic", "best_base_model")


@pytest.mark.determinism("d1")
def test_meta_router_eval_full_contract_feature_manifest(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _write_predictions(run_dir, n_days=60)
    panel_path = tmp_path / "panel.parquet"
    _panel_for_predictions(60).to_parquet(panel_path, index=False)
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    config = MetaRouterConfig(
        default_candidate_id=DEFAULT_CANDIDATE_ID,
        gating_baselines=("linear_gate", "best_base_model"),
        random_seed=42,
    )
    yaml_dict = {
        "evaluation": {
            "full_contract": True,
            "pass_baseline": "best_base_model",
            "pass_fold": "fold_2",
        }
    }
    result = run_meta_router_evaluation(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=spec,
        config=config,
        target_column="forward_return_1d",
        yaml_dict=yaml_dict,
    )
    features = result["feature_columns_used"]
    assert len(features) > 5
    assert any(col.startswith("state_change") or col == "state_change_probability" for col in features)
    assert "pass_baseline_test_economics" in result
    assert "best_routing_gate_id" in result
    assert "pdr002_full_contract_pass" in result


@pytest.mark.determinism("d1")
def test_max_routing_concentration_flags_degenerate_clone(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.candidate_portfolios.meta_router_eval import (
        _DEGENERATE_ROUTING_CONCENTRATION,
        _max_routing_concentration_pct,
    )

    row = {
        "routing_summary": {
            "chosen_candidate_counts_by_fold": {"fold_2": {"xgboost": 291, "extra_trees": 1}}
        }
    }
    concentration = _max_routing_concentration_pct(row, "fold_2")
    assert concentration >= _DEGENERATE_ROUTING_CONCENTRATION

    diverse = {
        "routing_summary": {
            "chosen_candidate_counts_by_fold": {"fold_2": {"xgboost": 150, "extra_trees": 141}}
        }
    }
    assert _max_routing_concentration_pct(diverse, "fold_2") < _DEGENERATE_ROUTING_CONCENTRATION
