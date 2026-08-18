"""Gate 6 promotion statistics tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysrc.pipeline.candidate_portfolios.promotion_stats import (
    build_walk_forward_pbo_surface,
    run_promotion_stat_battery,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec


def _synthetic_positions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_id in ("fold_0", "fold_1"):
        for date in ("2024-01-02", "2024-01-03", "2024-01-04"):
            for candidate_id, weight in [("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3)]:
                rows.append(
                    {
                        "date": date,
                        "ticker": "AAA",
                        "target_weight": weight,
                        "candidate_id": candidate_id,
                        "fold_id": fold_id,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in ("2024-01-02", "2024-01-03", "2024-01-04"):
        rows.append({"date": date, "instrument": "AAA", "forward_return_1d": 0.01})
    return pd.DataFrame(rows)


@pytest.mark.determinism("d1")
def test_promotion_stat_battery_emits_appendix_h_keys(deterministic_seed: int) -> None:
    _ = deterministic_seed
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, size=300)
    battery = run_promotion_stat_battery(returns, n_trials=5, n_resamples=200, random_state=42)

    report = battery["stat_validity_report"]
    assert report["schema_version"] == "v1"
    assert "dsr" in report
    assert "min_trl" in report
    assert "bootstrap_ci" in report
    assert "pbo" in report
    assert "promotion_gate" in battery
    assert "harvey_t_stat" in battery["promotion_gate"]


@pytest.mark.determinism("d1")
def test_pbo_surface_builds_path_pairs(deterministic_seed: int) -> None:
    _ = deterministic_seed
    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=0.0, capacity_constraints=False)
    surface = build_walk_forward_pbo_surface(
        _synthetic_positions(),
        _synthetic_panel(),
        spec,
        cost_bps=0.0,
        trial_ids=("model_a", "model_b", "model_c"),
    )

    assert surface["records"]
    assert surface["path_pairs"]
    assert len(surface["path_pairs"]) == 2
    first = surface["path_pairs"][0]
    assert "in_sample_scores" in first
    assert "out_of_sample_scores" in first
    assert len(first["in_sample_scores"]) == 3


@pytest.mark.determinism("d1")
def test_higher_n_trials_deflates_dsr(deterministic_seed: int) -> None:
    _ = deterministic_seed
    rng = np.random.default_rng(7)
    returns = rng.normal(0.002, 0.01, size=400)
    single = run_promotion_stat_battery(returns, n_trials=1, n_resamples=200, random_state=7)
    multi = run_promotion_stat_battery(returns, n_trials=20, n_resamples=200, random_state=7)

    dsr_single = float(single["stat_validity_report"]["dsr"]["value"])
    dsr_multi = float(multi["stat_validity_report"]["dsr"]["value"])
    assert dsr_multi < dsr_single
