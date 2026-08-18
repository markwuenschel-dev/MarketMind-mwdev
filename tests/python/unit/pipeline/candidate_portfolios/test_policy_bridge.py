"""Tests for PDR-002 policy bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysrc.pipeline.candidate_portfolios.policy_bridge import (
    _evaluate_pdr002_gate1,
    _evaluate_pdr002_gate2,
    _fold1_head_to_head,
    _policy_sweep_grid,
    build_fold_attribution_report,
    build_policy_training_frame,
    run_policy_smoke_for_model_matrix_run,
    run_policy_sweep,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec


def _write_minimal_predictions(run_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for fold_id, dates in (
        ("fold_0", ("2024-01-02", "2024-01-03")),
        ("fold_1", ("2024-02-02", "2024-02-03")),
    ):
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


def _minimal_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-02-02", "2024-02-03"],
            "instrument": ["AAA"] * 4,
            "forward_return_1d": [0.01, 0.02, -0.01, 0.015],
        }
    )


def _long_panel(n_days: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame(
        {
            "date": dates,
            "instrument": ["AAA"] * n_days,
            "forward_return_1d": np.linspace(-0.01, 0.02, n_days),
        }
    )


def _long_predictions(run_dir: Path, n_days: int = 60) -> None:
    dates = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    rows: list[dict[str, object]] = []
    for idx, date in enumerate(dates):
        if idx < n_days // 3:
            fold_id = "fold_0"
        elif idx < 2 * n_days // 3:
            fold_id = "fold_1"
        else:
            fold_id = "fold_2"
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


@pytest.mark.determinism("d1")
def test_fold_attribution_report_shape(deterministic_seed: int, tmp_path: Path) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _write_minimal_predictions(run_dir)
    panel = _minimal_panel()
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    report = build_fold_attribution_report(run_dir, panel_path, spec, fold_id="fold_1")
    assert report["fold_id"] == "fold_1"
    assert "xgboost" in report["by_candidate"]


@pytest.mark.determinism("d1")
def test_build_policy_state_features_adds_bocpd_regime_columns(
    deterministic_seed: int,
    tmp_path: Path,
) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _long_predictions(run_dir, n_days=60)
    panel_path = tmp_path / "panel.parquet"
    _long_panel(60).to_parquet(panel_path, index=False)
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    frame = build_policy_training_frame(
        run_dir, panel_path, spec, target_column="forward_return_1d"
    )
    assert "regime_id" in frame.columns
    assert "state_change_probability" in frame.columns
    assert "state_bocpd_boundary_code" in frame.columns
    assert frame["regime_id"].notna().all()


@pytest.mark.determinism("d1")
def test_build_policy_state_features_yaml_columns(
    deterministic_seed: int,
    tmp_path: Path,
) -> None:
    _ = deterministic_seed
    from pysrc.contracts.meta_router import MetaRouterConfig

    run_dir = tmp_path / "run"
    _long_predictions(run_dir, n_days=60)
    panel_path = tmp_path / "panel.parquet"
    _long_panel(60).to_parquet(panel_path, index=False)
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    yaml_features = (
        "state_market_ret_1d",
        "state_dispersion",
        "state_market_trend_20",
        "state_market_vol_20",
        "cand_recent_utility_20",
    )
    config = MetaRouterConfig(
        default_candidate_id="equal_blend",
        state_features=yaml_features,
    )
    frame = build_policy_training_frame(
        run_dir, panel_path, spec, target_column="forward_return_1d", config=config
    )
    for column in yaml_features:
        assert column in frame.columns
        assert frame[column].notna().all()


@pytest.mark.determinism("d1")
def test_build_policy_state_features_cross_sectional_columns(
    deterministic_seed: int,
    tmp_path: Path,
) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _long_predictions(run_dir, n_days=60)
    panel_path = tmp_path / "panel.parquet"
    _long_panel(60).to_parquet(panel_path, index=False)
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    frame = build_policy_training_frame(
        run_dir, panel_path, spec, target_column="forward_return_1d"
    )
    for column in (
        "state_cs_median_lagged_ret",
        "state_cs_breadth_positive",
        "state_cs_return_dispersion",
        "state_cs_universe_count",
    ):
        assert column in frame.columns
        assert np.isfinite(frame[column].astype(float)).all()


@pytest.mark.determinism("d1")
def test_pdr002_gate1_beats_equal_blend_or_fold1_gap_close(deterministic_seed: int) -> None:
    _ = deterministic_seed
    pass_via_test, gate = _evaluate_pdr002_gate1(
        routed_test_sharpe=1.0,
        baseline_test_econ={"equal_blend": {"net_sharpe": 0.8}},
        fold1_head_to_head={"gap_close_pass": False, "gap_closed_vs_equal_blend": 0.1},
    )
    assert pass_via_test is True
    assert gate["beats_equal_blend_test"] is True

    pass_via_fold1, gate2 = _evaluate_pdr002_gate1(
        routed_test_sharpe=0.5,
        baseline_test_econ={"equal_blend": {"net_sharpe": 0.8}},
        fold1_head_to_head={"gap_close_pass": True, "gap_closed_vs_equal_blend": 0.35},
    )
    assert pass_via_fold1 is True
    assert gate2["fold1_gap_close_pass"] is True


@pytest.mark.determinism("d1")
def test_fold1_head_to_head_reports_gap_closed(deterministic_seed: int) -> None:
    _ = deterministic_seed
    dates = ["2024-02-02", "2024-02-03", "2024-02-05"]
    training_rows: list[dict[str, object]] = []
    for date, xgb_ret, eq_ret in zip(dates, [0.02, 0.01, 0.03], [0.01, 0.005, 0.02], strict=True):
        training_rows.append(
            {
                "date": date,
                "candidate_id": "xgboost",
                "fold_id": "fold_1",
                "net_return": xgb_ret,
            }
        )
        training_rows.append(
            {
                "date": date,
                "candidate_id": "equal_blend",
                "fold_id": "fold_1",
                "net_return": eq_ret,
            }
        )
    training_frame = pd.DataFrame(training_rows)
    routed_outputs = pd.DataFrame(
        {
            "date": dates,
            "fold_id": ["fold_1"] * len(dates),
            "net_return": [0.015, 0.008, 0.025],
        }
    )
    report = _fold1_head_to_head(training_frame, routed_outputs, fold_id="fold_1")
    assert report["fold_id"] == "fold_1"
    assert report["gap_closed_vs_equal_blend"] == pytest.approx(
        report["routed"]["net_sharpe"] - report["equal_blend"]["net_sharpe"]
    )


@pytest.mark.determinism("d1")
def test_run_policy_smoke_writes_v2_report_and_target_plans(
    deterministic_seed: int,
    tmp_path: Path,
) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _long_predictions(run_dir, n_days=80)
    panel_path = tmp_path / "panel.parquet"
    _long_panel(80).to_parquet(panel_path, index=False)
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    result = run_policy_smoke_for_model_matrix_run(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=spec,
        target_column="forward_return_1d",
        emit_target_plans=True,
    )
    report_path = Path(result["report_path"])
    assert report_path.name == "policy_allocation_report.v2.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "policy_allocation_report.v2"
    assert "fold1_head_to_head" in payload
    assert "pdr002_gate1" in payload
    assert "portfolio_target_plans_path" in result
    plans_path = Path(str(result["portfolio_target_plans_path"]))
    assert plans_path.is_file()
    plans_payload = json.loads(plans_path.read_text(encoding="utf-8"))
    assert plans_payload["plan_count"] >= 0


@pytest.mark.determinism("d1")
def test_policy_sweep_grid_has_54_combos(deterministic_seed: int) -> None:
    _ = deterministic_seed
    grid = _policy_sweep_grid()
    assert len(grid) == 54
    alphas = {row["selector_ridge_alpha"] for row in grid}
    margins = {row["switch_margin"] for row in grid}
    uncertainty_ks = {row["selector_uncertainty_k"] for row in grid}
    cost_buffers = {row["selector_cost_buffer"] for row in grid}
    assert alphas == {0.1, 1.0, 10.0}
    assert margins == {0.0, 0.05, 0.1}
    assert uncertainty_ks == {0.5, 1.0, 2.0}
    assert cost_buffers == {0.0, 0.01}


@pytest.mark.determinism("d1")
def test_pdr002_gate2_matches_gate1_criteria(deterministic_seed: int) -> None:
    _ = deterministic_seed
    pass_via_test, gate = _evaluate_pdr002_gate2(
        routed_test_sharpe=1.2,
        baseline_test_econ={"equal_blend": {"net_sharpe": 1.0}},
        fold1_head_to_head={"gap_close_pass": False, "gap_closed_vs_equal_blend": 0.1},
    )
    assert pass_via_test is True
    assert gate["policy_sweep_pass"] is True
    assert gate["beats_equal_blend_test"] is True


@pytest.mark.determinism("d1")
def test_run_policy_sweep_writes_results_on_run_dir(
    deterministic_seed: int,
    tmp_path: Path,
) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _long_predictions(run_dir, n_days=80)
    panel_path = tmp_path / "panel.parquet"
    _long_panel(80).to_parquet(panel_path, index=False)
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    result = run_policy_sweep(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=spec,
        target_column="forward_return_1d",
    )
    report_path = Path(result["report_path"])
    assert report_path == run_dir / "reports" / "policy_sweep_results.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "policy_sweep_results.v1"
    assert payload["grid_size"] == 54
    assert "best_config" in payload
    assert "pdr002_gate2" in payload
    assert "policy_sweep_pass" in payload
    assert len(payload["results"]) == 54
    assert "selector_cost_buffer" in payload["grid"]


@pytest.mark.determinism("d1")
def test_weighted_blend_positions_sum_to_one(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.candidate_portfolios.policy_bridge import (
        routed_positions_from_weighted_decisions,
    )

    decisions = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "fold_id": "fold_0",
                "split": "test",
                "model_weights_json": json.dumps({"xgboost": 0.6, "equal_blend": 0.4}),
                "gate_id": "validation_weighted_blend",
            }
        ]
    )
    positions = {
        "xgboost": pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "candidate_id": "xgboost",
                    "ticker": "AAA",
                    "target_weight": 1.0,
                    "fold_id": "fold_0",
                    "split": "test",
                }
            ]
        ),
        "equal_blend": pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "candidate_id": "equal_blend",
                    "ticker": "AAA",
                    "target_weight": 1.0,
                    "fold_id": "fold_0",
                    "split": "test",
                }
            ]
        ),
    }
    blended = routed_positions_from_weighted_decisions(decisions, positions)
    assert not blended.empty
    weight_sum = float(blended.groupby("date")["target_weight"].sum().iloc[0])
    assert weight_sum == pytest.approx(1.0)
