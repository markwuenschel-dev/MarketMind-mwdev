"""Unit tests for RG-09 boundary recovery primitives."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from tests.python.unit.meta.test_rg09_harness import _config_with_geometry, _write_fixture_bundle

from pysrc.meta.rg09_boundary_treatment import (
    RG09BoundaryRecoverySpec,
    causal_hysteresis_labels,
    merge_tiny_transition_bursts,
)
from pysrc.meta.rg09_harness import _derive_episodes, load_rg09_config


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_causal_hysteresis_requires_two_bars_to_switch(deterministic_seed: int) -> None:
    _ = deterministic_seed
    raw = ["A", "A", "B", "B", "A"]
    assert causal_hysteresis_labels(raw, confirm_bars=2) == ["A", "A", "A", "B", "B"]


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_merge_tiny_burst_sandwiched_run(deterministic_seed: int) -> None:
    _ = deterministic_seed
    merged, events, iters = merge_tiny_transition_bursts(["A", "B", "A"], max_burst_bars=1)
    assert merged == ["A", "A", "A"]
    assert events >= 1
    assert iters >= 1


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_entity_local_boundary_recovery_reduces_fragmentation(
    tmp_path: Any, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp("2024-01-02T00:00:00+00:00")
    n = 40
    for i in range(n):
        for entity, ns in [("ES", 0), ("NQ", 1)]:
            ts = base + pd.Timedelta(days=i * 2, nanoseconds=ns)
            rows.append(
                {
                    "entity_id": entity,
                    "decision_ts": ts,
                    "regime_id": "trend_single__stable",
                    "regime_label": "trend_single__stable",
                    "effective_at": ts,
                    "state_snapshot_id": f"sha256:state-{entity}-{i}",
                    "input_snapshot_id": "sha256:input",
                    "config_version": "rg09_v1.0.2",
                    "change_probability": 0.05,
                    "boundary_flag": "stable",
                    "regime_class": "bull",
                    "diag_regime_class_bocpd_gated": "bull",
                    "run_length_mode": float(i + 1),
                    "run_length_expectation": float(i + 1),
                    "transition_probability": 0.10,
                    "posterior_entropy": 0.02,
                    "trend_score_raw": 0.1,
                    "vol_score_raw": 0.2,
                    "rg09_trading_day_ord": int(i),
                }
            )
    frame = pd.DataFrame(rows)
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        summary_overrides={
            "entity_id": ["ES", "NQ"],
            "fixture_scope": "multi_instrument_governed_basket",
            "single_series_sufficient": False,
            "es_only_sufficient": False,
        },
        metadata_overrides={"fixture_scope": "multi_instrument_governed_basket"},
        config_overrides=_config_with_geometry(
            min_temporal_folds=1,
            min_support_rows=4,
            min_query_rows=4,
            label_horizon_bars=1,
            embargo_gap_bars_daily=0,
            min_dwell_time_bars=2,
            min_admissible_episode_count=1,
            min_regime_transition_count=0,
        ),
    )
    config = load_rg09_config(paths["config"])
    base_eps, base_ex = _derive_episodes(frame, config)
    v1_eps, v1_ex = _derive_episodes(
        frame,
        config,
        boundary_recovery=RG09BoundaryRecoverySpec(mode="boundary_recovery_v1_hysteresis"),
    )
    assert base_ex["HORIZON_OVERLAP"] >= int(len(frame)) - 2
    assert int(len(v1_eps)) > int(len(base_eps))


def _mixed_regime_class_constant_regime_id_frame() -> pd.DataFrame:
    """One segment under constant ``regime_id`` but raw ``regime_class`` majority differs from first bar."""
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp("2024-01-02T00:00:00+00:00")
    n = 10
    for i in range(n):
        rc = "bull" if i == 0 else "bear"
        ts = base + pd.Timedelta(days=i)
        rows.append(
            {
                "entity_id": "ES",
                "decision_ts": ts,
                "regime_id": "trend_stable__one",
                "regime_label": "trend_stable__one",
                "effective_at": ts,
                "state_snapshot_id": f"sha256:state-{i}",
                "input_snapshot_id": "sha256:input",
                "config_version": "rg09_v1.0.2",
                "change_probability": 0.05,
                "boundary_flag": "stable",
                "regime_class": rc,
                "diag_regime_class_bocpd_gated": rc,
                "run_length_mode": float(i + 1),
                "run_length_expectation": float(i + 1),
                "transition_probability": 0.10,
                "posterior_entropy": 0.02,
                "trend_score_raw": 0.1,
                "vol_score_raw": 0.2 + float(i) * 0.01,
                "rg09_trading_day_ord": i,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_boundary_recovery_episode_regime_class_is_majority_of_segment_bars(
    tmp_path: Any, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _mixed_regime_class_constant_regime_id_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_temporal_folds=1,
            min_support_rows=2,
            min_query_rows=2,
            label_horizon_bars=1,
            embargo_gap_bars_daily=0,
            min_dwell_time_bars=2,
            min_admissible_episode_count=1,
            min_regime_transition_count=0,
            min_support_query_mass_per_regime=1,
            min_regime_class_count_per_fold=1,
        ),
    )
    config = load_rg09_config(paths["config"])
    base_eps, _ = _derive_episodes(frame, config, boundary_recovery=None)
    base_spec_eps, _ = _derive_episodes(
        frame,
        config,
        boundary_recovery=RG09BoundaryRecoverySpec(mode="baseline"),
    )
    v1_eps, _ = _derive_episodes(
        frame,
        config,
        boundary_recovery=RG09BoundaryRecoverySpec(mode="boundary_recovery_v1_hysteresis"),
    )
    assert len(base_eps) >= 1
    assert len(base_spec_eps) >= 1
    assert len(v1_eps) >= 1
    assert str(base_eps.iloc[0]["regime_class"]) == "bull"
    assert str(base_spec_eps.iloc[0]["regime_class"]) == "bull"
    assert str(v1_eps.iloc[0]["regime_class"]) == "bear"


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_episode_regime_class_purity_gate_excludes_underactive_recovery(
    tmp_path: Any, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _mixed_regime_class_constant_regime_id_frame()
    base_over = _config_with_geometry(
        min_support_rows=2,
        min_query_rows=2,
        label_horizon_bars=1,
        embargo_gap_bars_daily=0,
        min_dwell_time_bars=2,
        min_admissible_episode_count=1,
        min_regime_transition_count=0,
        min_support_query_mass_per_regime=1,
        min_regime_class_count_per_fold=1,
    )
    spec = RG09BoundaryRecoverySpec(mode="boundary_recovery_v1_hysteresis")
    paths_lo = _write_fixture_bundle(
        tmp_path / "lo",
        frame=frame,
        config_overrides={**base_over, "min_episode_regime_class_purity": 0.70},
    )
    paths_hi = _write_fixture_bundle(
        tmp_path / "hi",
        frame=frame,
        config_overrides={**base_over, "min_episode_regime_class_purity": 0.95},
    )
    config_lo = load_rg09_config(paths_lo["config"])
    config_hi = load_rg09_config(paths_hi["config"])
    v1_lo, ex_lo = _derive_episodes(frame, config_lo, boundary_recovery=spec)
    v1_hi, ex_hi = _derive_episodes(frame, config_hi, boundary_recovery=spec)
    assert len(v1_lo) >= 1
    assert ex_lo.get("LOW_REGIME_CLASS_PURITY", 0) == 0
    assert len(v1_hi) == 0
    assert ex_hi.get("LOW_REGIME_CLASS_PURITY", 0) >= 1


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_episode_regime_class_purity_gate_inactive_on_baseline_paths(
    tmp_path: Any, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _mixed_regime_class_constant_regime_id_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=2,
            min_query_rows=2,
            label_horizon_bars=1,
            embargo_gap_bars_daily=0,
            min_dwell_time_bars=2,
            min_admissible_episode_count=1,
            min_regime_transition_count=0,
            min_support_query_mass_per_regime=1,
            min_regime_class_count_per_fold=1,
            min_episode_regime_class_purity=0.95,
        ),
    )
    config = load_rg09_config(paths["config"])
    legacy_eps, legacy_ex = _derive_episodes(frame, config, boundary_recovery=None)
    base_eps, base_ex = _derive_episodes(
        frame,
        config,
        boundary_recovery=RG09BoundaryRecoverySpec(mode="baseline"),
    )
    assert len(legacy_eps) >= 1
    assert len(base_eps) >= 1
    assert legacy_ex.get("LOW_REGIME_CLASS_PURITY", 0) == 0
    assert base_ex.get("LOW_REGIME_CLASS_PURITY", 0) == 0
