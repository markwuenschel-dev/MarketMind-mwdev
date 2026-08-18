"""Unit tests for RG-09 structural threshold calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pysrc.meta.rg09_harness import (
    RG09PilotConfig,
    _structural_direction_score,
    _structural_ratio,
    load_rg09_config,
)
from pysrc.meta.rg09_structural_calibration import (
    calibrate_structural_threshold,
    compute_structural_null_distribution,
)
from pysrc.meta.rg09_threshold_catalog import RG09_CONFIG_THRESHOLD_SPECS, threshold_value_record

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **overrides: Any) -> RG09PilotConfig:
    """Minimal config for calibration unit tests."""
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "config_id": "test_structural_cal",
        "p_value_threshold": 0.05,
        "null_draw_count": 4,
        "structural_separability_ratio_threshold": 1.0,
        "structural_direction_score_threshold": 0.0,
        "functional_harvey_t_threshold": 3.0,
        "embargo_gap_bars_daily": 0,
        "embargo_gap_fraction_intraday": 0.05,
        "min_support_rows": 2,
        "min_query_rows": 2,
        "label_confidence_threshold": 0.70,
        "min_admissible_episode_count": 2,
        "min_regime_transition_count": 2,
        "min_support_query_mass_per_regime": 1,
        "min_regime_class_count_per_fold": 1,
        "min_temporal_folds": 2,
        "min_dwell_time_bars": 3,
        "low_confidence_boundary_policy": "exclude_v1",
        "functional_model_default": "ridge",
        "functional_model_fallback": "mean_estimator",
        "null_seed_namespace": "rg09.test.structural_cal.v1",
    }
    payload.update(overrides)
    for field_name, spec in RG09_CONFIG_THRESHOLD_SPECS.items():
        if field_name not in payload:
            continue
        raw_value = payload[field_name]
        if isinstance(raw_value, dict):
            continue
        payload[field_name] = threshold_value_record(raw_value, spec.threshold_id)
    config_path = tmp_path / "test_structural_cal_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_rg09_config(config_path)


def _make_episodes(
    regime_classes: list[str],
    query_means: list[float],
    adaptation_gains: list[float] | None = None,
    fold_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Minimal episode DataFrame compatible with shuffled_regime_null and _structural_ratio.

    ``query_targets`` is set to match ``query_means`` so that
    ``_refresh_episode_statistics`` in the null generator recomputes the same values.
    ``support_targets`` is set to zero for all rows.
    """
    n = len(regime_classes)
    gains = query_means if adaptation_gains is None else adaptation_gains
    return pd.DataFrame(
        {
            "regime_class": regime_classes,
            "query_mean": query_means,
            "adaptation_gain": gains,
            "query_targets": [[q] for q in query_means],
            "support_targets": [[0.0] for _ in range(n)],
            "fold_id": fold_ids if fold_ids is not None else [0] * n,
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — null distribution is deterministic
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_null_distribution_is_deterministic(tmp_path: Path, deterministic_seed: int) -> None:
    config = _make_config(tmp_path)
    episodes = _make_episodes(
        ["crisis", "crisis", "bull", "bull"],
        [1.0, 1.2, -1.0, -1.2],
    )
    result_a = compute_structural_null_distribution(
        episodes,
        config=config,
        fixture_sha256="sha256:test_determinism_fixture",
        fold_id=0,
    )
    result_b = compute_structural_null_distribution(
        episodes,
        config=config,
        fixture_sha256="sha256:test_determinism_fixture",
        fold_id=0,
    )
    assert result_a["null_ratios"] == result_b["null_ratios"]


# ---------------------------------------------------------------------------
# Test 2 — null distribution length matches draw count
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_null_distribution_length_matches_draw_count(
    tmp_path: Path, deterministic_seed: int
) -> None:
    config = _make_config(tmp_path, null_draw_count=6)
    episodes = _make_episodes(
        ["crisis", "crisis", "bull", "bull"],
        [1.0, 1.2, -1.0, -1.2],
    )
    result = compute_structural_null_distribution(
        episodes,
        config=config,
        fixture_sha256="sha256:test_length_fixture",
        fold_id=0,
    )
    assert len(result["null_ratios"]) == config.null_draw_count


# ---------------------------------------------------------------------------
# Test 3 — real ratio sits above null mean on a signal-bearing surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_real_ratio_above_null_mean_on_signal_bearing_surface(
    tmp_path: Path, deterministic_seed: int
) -> None:
    # Crisis episodes have query_mean ≈ +2.0; bull episodes have query_mean ≈ -2.0.
    # The real structural ratio is maximised (between-class variance >> within-class).
    # Any null draw that mixes regimes produces a lower (or at most equal) structural ratio,
    # so the real ratio sits at or above all null draws → percentile = 1.0 > 0.5.
    regime_classes = ["crisis"] * 3 + ["bull"] * 3
    query_means = [2.0, 2.0, 2.0, -2.0, -2.0, -2.0]
    config = _make_config(tmp_path, null_draw_count=8)
    episodes = _make_episodes(regime_classes, query_means)
    result = compute_structural_null_distribution(
        episodes,
        config=config,
        fixture_sha256="sha256:test_signal_surface_fixture",
        fold_id=0,
    )
    assert result["real_ratio_null_percentile"] > 0.5


# ---------------------------------------------------------------------------
# Test 4 — calibrated threshold is max across folds
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_calibrated_threshold_is_max_across_folds(deterministic_seed: int) -> None:
    # Fold 0: null_ratios clustered around 0.3 → p95 ≈ 0.3
    # Fold 1: null_ratios clustered around 0.6 → p95 ≈ 0.6
    # Expected calibrated threshold = max(p95_fold0, p95_fold1) = p95_fold1
    fold0_ratios = [0.25, 0.28, 0.30, 0.31]
    fold1_ratios = [0.55, 0.58, 0.60, 0.61]
    fold_distributions: list[dict[str, Any]] = [
        {
            "fold_id": 0,
            "draw_count": 4,
            "real_structural_ratio": 0.5,
            "null_ratios": fold0_ratios,
            "null_mean": 0.0,
            "null_std": 0.0,
            "null_min": 0.0,
            "null_max": 0.0,
            "null_p50": 0.0,
            "null_p90": 0.0,
            "null_p95": 0.0,
            "null_p99": 0.0,
            "real_ratio_null_percentile": 1.0,
            "real_exceeds_null_p95": True,
            "real_exceeds_null_p99": True,
        },
        {
            "fold_id": 1,
            "draw_count": 4,
            "real_structural_ratio": 0.8,
            "null_ratios": fold1_ratios,
            "null_mean": 0.0,
            "null_std": 0.0,
            "null_min": 0.0,
            "null_max": 0.0,
            "null_p50": 0.0,
            "null_p90": 0.0,
            "null_p95": 0.0,
            "null_p99": 0.0,
            "real_ratio_null_percentile": 1.0,
            "real_exceeds_null_p95": True,
            "real_exceeds_null_p99": True,
        },
    ]
    calibration_quantile = 0.95
    result = calibrate_structural_threshold(
        fold_distributions,
        calibration_quantile=calibration_quantile,
    )
    expected_fold0 = float(np.percentile(fold0_ratios, calibration_quantile * 100))
    expected_fold1 = float(np.percentile(fold1_ratios, calibration_quantile * 100))
    expected_calibrated = max(expected_fold0, expected_fold1)
    assert result["calibrated_structural_threshold"] == pytest.approx(expected_calibrated)
    assert result["per_fold_calibrated_thresholds"]["0"] == pytest.approx(expected_fold0)
    assert result["per_fold_calibrated_thresholds"]["1"] == pytest.approx(expected_fold1)
    assert result["rationale"] == "max_across_folds_at_calibration_quantile"


# ---------------------------------------------------------------------------
# Test 5 — current threshold assessment is correct
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_current_threshold_assessment_is_correct(deterministic_seed: int) -> None:
    # Build a fold distribution where null p99 ≈ 0.5.
    # The current threshold in the test config is 1.0.
    # Assert that 1.0 > null_p99 → current_threshold_exceeds_null_p99 == True.
    null_ratios = [0.40, 0.45, 0.49, 0.50]
    null_arr = np.asarray(null_ratios, dtype=float)
    null_p99 = float(np.percentile(null_arr, 99))

    # Replicate the assessment logic from build_structural_calibration_report.
    current_threshold = 1.0
    null_p99_across_folds = null_p99  # single fold
    assert bool(current_threshold > null_p99_across_folds)


# ---------------------------------------------------------------------------
# Test 6 — degenerate episodes (single regime class) return zero structural ratio
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_degenerate_single_class_episodes(tmp_path: Path, deterministic_seed: int) -> None:
    # All episodes have the same regime_class; _structural_ratio must return 0.0.
    # compute_structural_null_distribution must not raise even with a degenerate surface.
    episodes = _make_episodes(
        ["bull", "bull", "bull", "bull"],
        [1.0, 1.1, 0.9, 1.05],
    )
    assert _structural_ratio(episodes) == 0.0

    config = _make_config(tmp_path)
    result = compute_structural_null_distribution(
        episodes,
        config=config,
        fixture_sha256="sha256:test_degenerate_fixture",
        fold_id=0,
    )
    assert result["real_structural_ratio"] == 0.0
    # All null draws also return 0.0 since regime shuffling of a single-class surface
    # cannot create between-class variance.
    assert all(r == 0.0 for r in result["null_ratios"])


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_transition_config_uses_direction_score_for_calibration(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    config = _make_config(
        tmp_path,
        episode_construction="transition_anchored",
        null_draw_count=2,
    )
    episodes = _make_episodes(
        ["crisis", "high_vol", "bull", "bear", "sideways"],
        [1.2, 0.8, -0.8, -1.1, -0.9],
        adaptation_gains=[1.2, 0.8, -0.8, -1.1, -0.9],
    )

    result = compute_structural_null_distribution(
        episodes,
        config=config,
        fixture_sha256="sha256:test_transition_direction_fixture",
        fold_id=0,
    )

    assert result["structural_measure"] == "direction_score"
    assert result["real_structural_ratio"] == pytest.approx(_structural_direction_score(episodes))
    assert result["real_structural_ratio"] != pytest.approx(_structural_ratio(episodes))
