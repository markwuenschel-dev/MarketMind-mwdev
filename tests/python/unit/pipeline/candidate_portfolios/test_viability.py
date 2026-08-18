"""Gate 3/4 candidate portfolio viability and robustness tests."""

from __future__ import annotations

import pandas as pd
import pytest

from pysrc.contracts.meta_router import DEFAULT_CANDIDATE_ID
from pysrc.pipeline.candidate_portfolios.viability import (
    _outcome_from_fold_wins,
    build_gate4_robustness_report,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec


def _positions_and_panel(
    *,
    fold_returns: dict[str, list[tuple[str, str, float]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build positions where each entry is (date, ticker, forward_return)."""

    pos_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    for fold_id, entries in fold_returns.items():
        for date, ticker, ret in entries:
            candidate_id = "model_a" if ticker == "WIN" else DEFAULT_CANDIDATE_ID
            pos_rows.append(
                {
                    "date": date,
                    "candidate_id": candidate_id,
                    "ticker": ticker,
                    "target_weight": 1.0,
                    "fold_id": fold_id,
                    "split": "test",
                }
            )
            panel_rows.append(
                {
                    "date": date,
                    "instrument": ticker,
                    "forward_return_1d": float(ret),
                }
            )
    return pd.DataFrame(pos_rows), pd.DataFrame(panel_rows)


@pytest.mark.determinism("d1")
def test_per_fold_economics_differ_by_fold(deterministic_seed: int) -> None:
    _ = deterministic_seed
    positions, panel = _positions_and_panel(
        fold_returns={
            "fold_0": [
                ("2024-01-02", "WIN", 0.02),
                ("2024-01-03", "WIN", 0.02),
                ("2024-01-02", "LOSE", -0.01),
                ("2024-01-03", "LOSE", -0.01),
            ],
            "fold_1": [
                ("2024-02-02", "WIN", 0.03),
                ("2024-02-03", "WIN", 0.03),
                ("2024-02-02", "LOSE", -0.02),
                ("2024-02-03", "LOSE", -0.02),
            ],
        },
    )
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=0.0, capacity_constraints=False)
    report = build_gate4_robustness_report(
        positions,
        panel,
        spec,
        focus_candidate="model_a",
        focus_candidates=("model_a", DEFAULT_CANDIDATE_ID),
        cost_bps_levels=(0.0,),
        evaluation_cost_bps=0.0,
    )

    assert report["evaluation_unit"] == "fold_id"
    assert set(report["by_fold"]) == {"fold_0", "fold_1"}
    fold0_a = report["by_fold"]["fold_0"]["model_a"]["cumulative_log_return"]
    fold1_a = report["by_fold"]["fold_1"]["model_a"]["cumulative_log_return"]
    assert fold0_a != fold1_a


@pytest.mark.determinism("d1")
def test_fold_win_count_outcome_mapping(deterministic_seed: int) -> None:
    _ = deterministic_seed
    assert _outcome_from_fold_wins(0, focus_candidate="xgboost")[0] == "A"
    assert _outcome_from_fold_wins(1, focus_candidate="xgboost")[0] == "C"
    assert _outcome_from_fold_wins(2, focus_candidate="xgboost")[0] == "D"


@pytest.mark.determinism("d1")
def test_robustness_counts_fold_wins_for_focus_candidate(deterministic_seed: int) -> None:
    _ = deterministic_seed
    positions, panel = _positions_and_panel(
        fold_returns={
            "fold_0": [
                ("2024-01-02", "WIN", 0.05),
                ("2024-01-03", "WIN", 0.02),
                ("2024-01-02", "LOSE", 0.01),
                ("2024-01-03", "LOSE", -0.01),
            ],
            "fold_1": [
                ("2024-02-02", "WIN", 0.04),
                ("2024-02-03", "WIN", 0.01),
                ("2024-02-02", "LOSE", 0.01),
                ("2024-02-03", "LOSE", -0.005),
            ],
            "fold_2": [
                ("2024-03-02", "WIN", -0.02),
                ("2024-03-03", "WIN", -0.03),
                ("2024-03-02", "LOSE", 0.02),
                ("2024-03-03", "LOSE", 0.01),
            ],
        },
    )
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=0.0, capacity_constraints=False)
    report = build_gate4_robustness_report(
        positions,
        panel,
        spec,
        baseline_candidate_id=DEFAULT_CANDIDATE_ID,
        focus_candidate="model_a",
        focus_candidates=("model_a", DEFAULT_CANDIDATE_ID),
        cost_bps_levels=(0.0,),
        evaluation_cost_bps=0.0,
    )

    assert report["fold_win_count"] == 2
    assert report["upgraded_outcome"] == "D"
