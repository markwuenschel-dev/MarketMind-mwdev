"""Unit tests for RG-09 pre-gate episode-feasibility reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
from tests.python.unit.meta.test_rg09_harness import (
    _config_with_geometry,
    _corrected_v2_frame,
    _corrected_v2_summary_overrides,
    _read_json,
    _single_episode_frame,
    _write_fixture_bundle,
)

from pysrc.meta.rg09_feasibility import build_episode_feasibility_report


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_feasibility_report_counts_candidate_and_admissible_episodes(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _single_episode_frame(length=8)
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        summary_overrides={
            "entity_id": ["ES", "NQ"],
            "single_series_sufficient": False,
            "es_only_sufficient": False,
            "fixture_scope": "multi_instrument_governed_basket",
        },
        metadata_overrides={
            "fixture_scope": "multi_instrument_governed_basket",
            "instrument_ids": ["ES", "NQ"],
        },
        config_overrides=_config_with_geometry(),
    )
    report = build_episode_feasibility_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_path=tmp_path / "rg09_episode_feasibility.json",
    )
    assert report["schema_version"] == "rg09.episode_feasibility.v3"
    assert "boundary_recovery_spec" in report
    assert report["candidate_episode_count"] == 1
    assert report["candidate_episode_count_by_entity"] == {"ES": 1}
    assert report["admissible_episode_count"] == 1
    assert report["admissible_episode_count_by_entity"] == {"ES": 1}
    assert report["distinct_regime_transition_count"] == 0
    assert report["folds_meeting_min_regime_class_count"] == 1
    assert report["precondition_fail_codes"] == []
    written = _read_json(tmp_path / "rg09_episode_feasibility.json")
    assert written["fixture_sha256"] == report["fixture_sha256"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_feasibility_report_tracks_exclusions_by_entity(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    rows: list[dict[str, Any]] = []
    ts = pd.Timestamp("2024-01-02T00:00:00+00:00")
    for step in range(6):
        rows.append(
            {
                "entity_id": "ES",
                "decision_ts": ts + pd.Timedelta(days=step),
                "regime_id": "trend_hi__vol_hi__bocpd_stable",
                "regime_label": "trend_hi__vol_hi__bocpd_stable",
                "effective_at": ts + pd.Timedelta(days=step),
                "state_snapshot_id": f"sha256:state-{step}",
                "input_snapshot_id": "sha256:input",
                "config_version": "rg09_v1.0.2",
                "change_probability": 0.05,
                "boundary_flag": "stable",
                "regime_class": "bull",
                "diag_regime_class_bocpd_gated": "bull",
                "run_length_mode": float(step + 1),
                "run_length_expectation": float(step + 1),
                "transition_probability": 0.1,
                "posterior_entropy": 0.02,
                "trend_score_raw": 0.1,
                "vol_score_raw": 0.2,
                "rg09_trading_day_ord": 0 if step == 0 else (step + 1),
            }
        )
    frame = pd.DataFrame(rows)
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        summary_overrides={
            "single_series_sufficient": False,
            "es_only_sufficient": False,
        },
        config_overrides=_config_with_geometry(),
    )
    report = build_episode_feasibility_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_path=tmp_path / "rg09_episode_feasibility.json",
    )
    assert report["candidate_episode_count"] == 1
    assert report["exclusion_counts_by_code"]["NONCONTIGUOUS"] == 1
    assert report["exclusion_counts_by_entity"]["ES"]["NONCONTIGUOUS"] == 1
    assert report["admissible_episode_count"] == 0


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_feasibility_report_includes_regime_class_and_crisis_surface_detail(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _single_episode_frame(length=8)
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        summary_overrides={
            "row_counts_by_class": {"bull": 8, "crisis": 10},
            "row_counts_by_class_bocpd_gated": {"bull": 8, "crisis": 1},
            "fixture_scope": "multi_instrument_governed_basket",
            "entity_id": ["ES", "NQ"],
            "single_series_sufficient": False,
            "es_only_sufficient": False,
        },
        config_overrides=_config_with_geometry(),
    )
    report = build_episode_feasibility_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_path=tmp_path / "rg09_episode_feasibility.json",
    )
    assert report["candidate_episode_count_by_regime_class"] == {"bull": 1}
    assert report["admissible_episode_count_by_regime_class"] == {"bull": 1}
    assert report["admissible_row_count_by_regime_class"] == {"bull": 8}
    assert report["crisis_surface_summary"]["canonical_crisis_rows"] == 10
    assert report["crisis_surface_summary"]["bocpd_gated_crisis_rows"] == 1
    assert report["crisis_surface_summary"]["candidate_crisis_episode_count"] == 0
    assert report["crisis_surface_summary"]["admissible_crisis_episode_count"] == 0


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_feasibility_report_passes_expected_temporal_folds_into_fixture_intake(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_feasibility as feasibility

    frame = _single_episode_frame(length=8)
    paths = _write_fixture_bundle(
        tmp_path, frame=frame, config_overrides=_config_with_geometry(min_temporal_folds=2)
    )
    captured: dict[str, Any] = {}
    original = feasibility._load_and_validate_fixture

    def _wrapped_load_and_validate_fixture(**kwargs: Any) -> tuple[Any, list[str]]:
        captured["expected_temporal_folds"] = kwargs.get("expected_temporal_folds")
        return original(**kwargs)

    monkeypatch.setattr(
        feasibility, "_load_and_validate_fixture", _wrapped_load_and_validate_fixture
    )
    report = build_episode_feasibility_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_path=tmp_path / "rg09_episode_feasibility.json",
    )
    assert report["min_temporal_folds"] == 2
    assert captured["expected_temporal_folds"] == 2


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_feasibility_report_corrected_v2_surfaces_geometry_contract_failure(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _corrected_v2_frame().drop(columns=["rg09_trading_day_ord"])
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        summary_overrides=_corrected_v2_summary_overrides(frame),
        config_overrides=_config_with_geometry(min_temporal_folds=2),
    )
    report = build_episode_feasibility_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_path=tmp_path / "rg09_episode_feasibility.json",
    )
    assert report["fixture_validation_fail_codes"] == ["FAIL_FIXTURE_GEOMETRY_CONTRACT"]
    assert report["precondition_fail_codes"] == [
        "FAIL_FIXTURE_GEOMETRY_CONTRACT",
        "FAIL_INSUFFICIENT_EPISODES",
    ]
    assert report["preflight_gate_outcome"] == "fail_closed_preconditions"
    assert report["fixture_identity"]["recomputed_fixture_sha_matches_sidecars"] is True


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_feasibility_report_legacy_surface_retains_fallback_behavior(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _single_episode_frame(length=8)
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(
            min_temporal_folds=1, min_support_rows=1, min_query_rows=1, label_horizon_bars=1
        ),
    )
    report = build_episode_feasibility_report(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_path=tmp_path / "rg09_episode_feasibility.json",
    )
    assert report["fixture_validation_fail_codes"] == []
    assert report["admissible_episode_count"] == 1
    assert report["preflight_gate_outcome"] == "clear_to_attempt_gate"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_feasibility_report_fixture_identity_flags_expected_h2_sha_mismatch(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    frame = _single_episode_frame(length=8)
    paths = _write_fixture_bundle(
        tmp_path,
        frame=frame,
        config_overrides=_config_with_geometry(),
    )
    h2_fixture_dir = tmp_path / "fixtures" / "rg09" / "h2"
    h2_fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = h2_fixture_dir / "rg09_fixture_v2.parquet"
    fixture_path.write_bytes(paths["fixture"].read_bytes())
    output_path = tmp_path / "runs" / "rg09_h2_preflight" / "rg09_episode_feasibility.json"
    report = build_episode_feasibility_report(
        fixture_path=fixture_path,
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_path=output_path,
    )
    assert (
        report["fixture_identity"]["expected_fixture_sha256"]
        == "sha256:07b28854ab30099bbe548ea77ec677122290c9412b6f451bd88fdb8ed781bfa9"
    )
    assert report["fixture_identity"]["matches_expected_fixture_sha256"] is False
    assert report["preflight_gate_outcome"] == "fail_closed_preconditions"
