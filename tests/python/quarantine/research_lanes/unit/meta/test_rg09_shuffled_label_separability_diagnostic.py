"""Tests for shuffled_label vs real separability diagnostic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
from tests.python.unit.meta.test_rg09_geometry_sensitivity import _seven_bar_three_episode_frame
from tests.python.unit.meta.test_rg09_harness import _config_with_geometry, _write_fixture_bundle

from pysrc.meta.rg09_boundary_treatment import RG09BoundaryRecoverySpec
from pysrc.meta.rg09_harness import _regime_separability_statistic, load_rg09_config
from pysrc.meta.rg09_nulls import shuffled_label_null
from pysrc.meta.rg09_shuffled_label_separability_diagnostic import (
    DIAGNOSTIC_SCHEMA_VERSION,
    build_shuffled_label_separability_diagnostic,
    candidate_segment_regime_class_purity,
    regime_separability_decomposition,
)


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_regime_separability_decomposition_matches_harness(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episodes = pd.DataFrame(
        {
            "regime_class": ["a", "a", "b", "b"],
            "adaptation_gain": [1.0, 2.0, 5.0, 7.0],
        }
    )
    dec = regime_separability_decomposition(episodes)
    assert dec["ratio"] == pytest.approx(_regime_separability_statistic(episodes))
    assert dec["n_regime_classes"] == 2
    assert dec["class_counts"] == {"a": 2, "b": 2}


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_shuffled_label_null_draw0_is_deterministic(deterministic_seed: int) -> None:
    _ = deterministic_seed
    episodes = pd.DataFrame(
        {
            "regime_class": ["bull", "bear"],
            "adaptation_gain": [0.1, -0.1],
            "support_features": [[0.0], [0.0]],
            "query_features": [[0.0], [0.0]],
            "support_targets": [[0.0, 0.1], [0.0, 0.1]],
            "query_targets": [[0.2, 0.3], [-0.2, -0.3]],
        }
    )
    ns = "rg09.test.sep_diag"
    fx = "sha256:fixture_test"
    a = shuffled_label_null(episodes, namespace=ns, fixture_sha256=fx, draw_index=0)
    b = shuffled_label_null(episodes, namespace=ns, fixture_sha256=fx, draw_index=0)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


@pytest.mark.unit
@pytest.mark.determinism("d1")
def test_candidate_segment_regime_class_purity_flags_mismatch(deterministic_seed: int) -> None:
    _ = deterministic_seed
    g0 = pd.DataFrame(
        {
            "entity_id": ["ES", "ES"],
            "regime_class": ["bull", "bear"],
        }
    )
    g1 = pd.DataFrame({"entity_id": ["ES"], "regime_class": ["bull"]})
    out = candidate_segment_regime_class_purity([g0, g1], majority_label=False)
    assert out["reference_rule"] == "iloc0"
    assert out["segments_with_any_mismatch"] == 1
    assert out["max_fraction_mismatch"] == pytest.approx(0.5)

    out_maj = candidate_segment_regime_class_purity([g0, g1], majority_label=True)
    assert out_maj["reference_rule"] == "majority"
    assert out_maj["segments_with_any_mismatch"] == 1
    assert out_maj["max_fraction_mismatch"] == pytest.approx(0.5)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_build_shuffled_label_separability_diagnostic_smoke(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _seven_bar_three_episode_frame()
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_support_rows=2,
            min_query_rows=2,
            min_dwell_time_bars=3,
            min_admissible_episode_count=1,
            min_regime_transition_count=0,
            min_support_query_mass_per_regime=1,
            min_regime_class_count_per_fold=1,
            min_temporal_folds=1,
            label_horizon_bars=1,
        ),
    )
    config = load_rg09_config(paths["config"])
    bundle_paths = paths
    from pysrc.meta.rg09_harness import _load_and_validate_fixture

    bundle, _fail = _load_and_validate_fixture(
        fixture_path=bundle_paths["fixture"],
        fixture_summary_path=bundle_paths["summary"],
        fixture_metadata_path=bundle_paths["metadata"],
        expected_temporal_folds=config.min_temporal_folds,
    )
    spec = RG09BoundaryRecoverySpec(mode="boundary_recovery_v1_hysteresis")
    raw_fc = bundle.summary.get("fold_construction")
    fold_construction = raw_fc if isinstance(raw_fc, dict) else None
    out = build_shuffled_label_separability_diagnostic(
        frame=bundle.frame,
        config=config,
        fixture_sha256=bundle.fixture_sha256,
        fold_construction=fold_construction,
        require_strict_geometry=False,
        boundary_recovery=spec,
    )
    assert out["schema_version"] == DIAGNOSTIC_SCHEMA_VERSION
    assert out["boundary_recovery_mode"] == "boundary_recovery_v1_hysteresis"
    assert "candidate_segment_regime_class_purity" in out
    assert out["closeout"]["recommendation"] in (
        "accept_null_statistic_mismatch_on_recovery_surface",
        "investigate_episode_regime_class_attribution",
    )
    assert out["closeout"]["attribution_mean_mismatch_ceiling"] == pytest.approx(0.10)
    assert out["closeout"]["min_episode_regime_class_purity_config"] == pytest.approx(0.0)
    assert isinstance(out["folds"], list)
    for row in out["folds"]:
        assert "real_separability" in row
        assert "null_shuffled_label_draw0" in row
