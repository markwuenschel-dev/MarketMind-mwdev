"""Integration tests for the W2-SU signal-usability diagnostic runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysrc.meta.signal_usability.config import SignalUsabilityConfig
from pysrc.meta.signal_usability.runner import (
    build_default_signal_usability_market_data,
    run_default_signal_usability_diagnostic,
    run_signal_usability_diagnostic,
)


@pytest.mark.integration
@pytest.mark.determinism("d1")
def test_w2_su_runner_emits_signal_usability_report(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    out_dir = tmp_path / "w2_su"
    result = run_default_signal_usability_diagnostic(
        output_dir=out_dir,
        seed=4242,
        timestamp_utc="2026-05-03T12:00:00Z",
    )

    report_path = out_dir / "signal_usability_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.report_path == report_path
    assert report["schema_version"] == "w2_su.signal_usability_report.v2"
    assert report["run_id"].startswith("run.sha256:")
    assert report["created_at"] == "2026-05-03T12:00:00Z"
    assert int(report["panel"]["row_count"]) > 0
    assert int(report["panel"]["eligible_row_count"]) > 0
    assert sorted(report["panel"]["signals"]) == report["panel"]["signals"]
    assert "static_equal_weight_signal_ensemble" in report["baselines"]
    assert "rolling_ic_weighted_ensemble" in report["baselines"]
    assert "best_historical_signal" in report["baselines"]
    assert "simple_regime_conditioned_gate" in report["baselines"]
    assert report["data"]["data_mode"] == "synthetic"
    assert report["classification"]["result"] == "PASS"


@pytest.mark.integration
@pytest.mark.determinism("d1")
def test_w2_su_runner_does_not_replace_phase2_triple_contract(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    root_dir = tmp_path / "phase2_bundle"
    root_dir.mkdir(parents=True, exist_ok=True)
    sentinels = {
        "task_manifest.json": {"schema_version": "sentinel.task_manifest.v1"},
        "meta_validity_report.json": {"schema_version": "sentinel.meta_validity.v1"},
        "execution_assumptions.json": {"schema_version": "sentinel.execution_assumptions.v1"},
    }
    for filename, payload in sentinels.items():
        (root_dir / filename).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    run_default_signal_usability_diagnostic(
        output_dir=root_dir / "w2_su",
        seed=5151,
        timestamp_utc="2026-05-03T12:00:00Z",
    )

    for filename, payload in sentinels.items():
        observed = json.loads((root_dir / filename).read_text(encoding="utf-8"))
        assert observed == payload
    assert (root_dir / "w2_su" / "signal_usability_report.json").is_file()


@pytest.mark.integration
@pytest.mark.determinism("d1")
def test_w2_su_v1_real_or_fixture_panel_emits_v2_report(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    result = run_signal_usability_diagnostic(
        market_data=build_default_signal_usability_market_data(),
        config=SignalUsabilityConfig(
            output_dir=tmp_path / "w2_su",
            data_mode="fixture",
            data_source_id="integration_fixture_panel",
            min_recent_window=5,
            recent_ic_window=10,
            recent_hit_rate_window=10,
            decay_window=10,
        ),
    )

    report = result.report
    assert report["schema_version"] == "w2_su.signal_usability_report.v2"
    assert report["data"]["data_mode"] == "fixture"
    assert report["split"]["method"] == "temporal_holdout"
    assert set(report["panel_v1"]["row_count_by_split"]) == {"train", "validation", "test"}
    assert set(report["panel_v1"]["eligible_row_count_by_split"]) == {"train", "validation", "test"}
    assert sum(report["panel_v1"]["eligible_row_count_by_split"].values()) == int(
        report["panel"]["eligible_row_count"]
    )
    assert report["learnability_interpretation"]["result"] in {
        "SUPPORTS_LEARNABILITY",
        "NO_CLEAR_SIGNAL",
        "INSUFFICIENT_REAL_DATA",
        "STRUCTURAL_FAILURE",
    }
    assert "Gate-only" in report["learnability_interpretation"]["reason"]
