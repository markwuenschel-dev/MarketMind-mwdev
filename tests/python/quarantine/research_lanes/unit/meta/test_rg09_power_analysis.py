"""Unit tests for RG-09 advisory power analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pysrc.meta.rg09_threshold_catalog import RG09_CONFIG_THRESHOLD_SPECS, threshold_value_record


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _baseline_run_dir() -> Path:
    return _repo_root() / "runs/rg09_v2/empirical_v2bocpd_directional_nme"


def _comparison_run_dir() -> Path:
    return _repo_root() / "runs/rg09_v2/h1_power_a"


def _required_docs_exist() -> bool:
    root = _repo_root()
    return all(
        path.exists()
        for path in (
            _baseline_run_dir(),
            _comparison_run_dir(),
            root / "docs/rg09/rg09_gate_spec.md",
            root / "docs/rg09/rg09_pilot_config_v1.json",
            root / "docs/rg09/rg09_pilot_config_v1_power_a.json",
            root / "docs/rg09/rg09_bocpd_fixture_config_v2.json",
            root / "docs/rg09/rg09_power_a_comparison.md",
            root / "docs/rg09/rg09_multi_fixture_manifest_v2.json",
        )
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _pilot_config_payload(config_id: str) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "config_id": config_id,
        "p_value_threshold": 0.05,
        "null_draw_count": 64,
        "structural_separability_ratio_threshold": 1.0,
        "structural_direction_score_threshold": 0.0,
        "functional_harvey_t_threshold": 3.0,
        "embargo_gap_bars_daily": 0,
        "embargo_gap_fraction_intraday": 0.05,
        "min_support_rows": 64,
        "min_query_rows": 32,
        "label_confidence_threshold": 0.7,
        "min_admissible_episode_count": 30,
        "min_regime_transition_count": 5,
        "min_support_query_mass_per_regime": 2,
        "min_regime_class_count_per_fold": 2,
        "min_temporal_folds": 2,
        "min_dwell_time_bars": 32,
        "label_horizon_bars": 32,
        "low_confidence_boundary_policy": "exclude_v1",
        "functional_model_default": "ridge",
        "functional_model_fallback": "mean_estimator",
        "null_seed_namespace": "rg09.v1",
        "episode_construction": "stable_span",
    }
    for field_name, spec in RG09_CONFIG_THRESHOLD_SPECS.items():
        if field_name not in payload:
            continue
        payload[field_name] = threshold_value_record(payload[field_name], spec.threshold_id)
    return payload


def _fold_payload(
    fold_id: int, draw_count: int, real_statistic: float, null_mean: float
) -> dict[str, Any]:
    family = {
        "draw_count": draw_count,
        "real_statistic": real_statistic,
        "null_mean": null_mean,
        "null_std": 0.2,
        "empirical_p_value": 0.04,
    }
    return {
        "fold_id": fold_id,
        "structural_separability_ratio": 1.5,
        "functional_evidence": {
            "harvey_t": 2.2,
            "mean_delta": 0.4,
            "positive_delta": True,
        },
        "null_families": {
            "shuffled_regime": dict(family),
            "shuffled_label": dict(family),
            "matched_exchangeable_window": dict(family),
        },
    }


def _write_run_dir(
    run_dir: Path,
    *,
    fixture_summary_path: Path,
    fixture_sha256: str,
    source_dataset_id: str,
    config_id: str,
    episode_count: int,
    episode_construction: str = "stable_span",
    structural_measure: str = "separability_ratio",
) -> Path:
    gate_result = {
        "decision": "NEEDS_MORE_EVIDENCE",
        "evidence": {
            "non_exchangeability": {
                "structural_pass": True,
                "structural_separability_ratio": 1.75,
                "structural_measure": structural_measure,
                "fold_evidence": [
                    {
                        **_fold_payload(0, draw_count=40, real_statistic=1.3, null_mean=0.5),
                        "structural_measure": structural_measure,
                    },
                    {
                        **_fold_payload(1, draw_count=38, real_statistic=1.2, null_mean=0.45),
                        "structural_measure": structural_measure,
                    },
                ],
            },
            "null_collapse": {
                "functional_evidence": {
                    "harvey_t": 2.1,
                    "mean_delta": 0.35,
                    "positive_delta": True,
                }
            },
        },
    }
    empirical_summary = {
        "supports_non_exchangeability": "inconclusive",
        "relative_rule_conclusion": "severity_rule_better",
    }
    task_manifest = {
        "episode_count": episode_count,
        "fixture_sha256": fixture_sha256,
        "source_dataset_id": source_dataset_id,
        "exclusion_counts_by_code": {
            "COLD_START": 8,
            "HORIZON_OVERLAP": 1,
            "INSUFFICIENT_QUERY": 0,
            "INSUFFICIENT_SUPPORT": 0,
            "LABEL_NOT_YET_EFFECTIVE": 0,
            "LOW_CONFIDENCE_BOUNDARY": 0,
            "NONCONTIGUOUS": 0,
        },
    }
    run_config = {
        "pilot_config": {
            **_pilot_config_payload(config_id),
            "episode_construction": episode_construction,
        },
        "fixture_summary_path": str(fixture_summary_path),
    }
    _write_json(run_dir / "rg09_gate_result.json", gate_result)
    _write_json(run_dir / "rg09_empirical_summary.json", empirical_summary)
    _write_json(run_dir / "task_manifest.json", task_manifest)
    _write_json(run_dir / "run_config_resolved.json", run_config)
    _write_json(run_dir / "rg09_empirical_detail.json", {"schema_version": "test"})
    _write_json(run_dir / "rg09_diagnostics.json", {"schema_version": "test"})
    return run_dir


def _write_advisory_workspace(tmp_path: Path, *, contaminated: bool) -> dict[str, Path]:
    fixture_dir = tmp_path / "fixtures" / "rg09" / "v2"
    summary_path = fixture_dir / "rg09_fixture_summary.json"
    metadata_path = fixture_dir / "rg09_fixture_metadata.json"
    fixture_sha256 = "sha256:test-fixture"
    if contaminated:
        source_dataset_id = "rg09_v2:test;entities=ES,NQ;apply_diversification=false"
        summary = {
            "fixture_sha256": fixture_sha256,
            "source_dataset_id": source_dataset_id,
            "entity_id": ["ES", "NQ"],
            "fixture_scope": "multi_instrument_governed_basket",
            "uniform_calendar_day_index": True,
            "calendar_overlap_policy": "independent_instruments",
        }
    else:
        source_dataset_id = "rg09_v2:test;entities=ES,NQ;apply_diversification=false"
        summary = {
            "fixture_sha256": fixture_sha256,
            "source_dataset_id": source_dataset_id,
            "entity_id": ["ES", "NQ"],
            "fixture_scope": "multi_instrument_governed_basket",
            "uniform_calendar_day_index": False,
            "calendar_overlap_policy": "independent_instruments",
            "fold_construction": {
                "method": "calendar_time",
                "uniform_calendar_day_index": False,
                "temporal_folds": 2,
                "time_ranges": {
                    "fold_0": ["2020-01-01", "2020-06-30"],
                    "fold_1": ["2020-07-01", "2020-12-31"],
                },
            },
        }
    _write_json(summary_path, summary)
    _write_json(metadata_path, {"fixture_sha256": fixture_sha256})

    baseline_config_path = _write_json(
        tmp_path / "docs" / "rg09" / "rg09_pilot_config_v1.json",
        _pilot_config_payload("rg09_pilot_config_v1"),
    )
    comparison_config_path = _write_json(
        tmp_path / "docs" / "rg09" / "rg09_pilot_config_v1_power_a.json",
        _pilot_config_payload("rg09_pilot_config_v1_power_a"),
    )
    fixture_config_path = _write_json(
        tmp_path / "docs" / "rg09" / "rg09_bocpd_fixture_config_v2.json",
        {"config_version": "rg09_v1.1.0"},
    )
    multi_manifest_path = _write_json(
        tmp_path / "docs" / "rg09" / "rg09_multi_fixture_manifest_v2.json",
        {
            "schema_version": "rg09_multi_fixture_manifest/1",
            "uniform_calendar_day_index": False,
            "calendar_overlap_policy": "independent_instruments",
            "temporal_folds": 2,
            "segments": [
                {"entity_id": "ES", "input_path": "data/rg09/ES_F.parquet", "close_scale": 1.0},
                {"entity_id": "NQ", "input_path": "data/rg09/NQ_F.parquet", "close_scale": 1.0},
            ],
        },
    )
    gate_spec_path = _write_json(
        tmp_path / "docs" / "rg09" / "rg09_gate_spec.json", {"schema_version": "test"}
    )
    power_comparison_doc_path = tmp_path / "docs" / "rg09" / "rg09_power_a_comparison.md"
    power_comparison_doc_path.write_text("comparison doc", encoding="utf-8")

    baseline_run_dir = _write_run_dir(
        tmp_path / "runs" / "baseline",
        fixture_summary_path=summary_path,
        fixture_sha256=fixture_sha256,
        source_dataset_id=source_dataset_id,
        config_id="rg09_pilot_config_v1",
        episode_count=40,
    )
    comparison_run_dir = _write_run_dir(
        tmp_path / "runs" / "power_a",
        fixture_summary_path=summary_path,
        fixture_sha256=fixture_sha256,
        source_dataset_id=source_dataset_id,
        config_id="rg09_pilot_config_v1_power_a",
        episode_count=80,
    )
    return {
        "baseline_run_dir": baseline_run_dir,
        "comparison_run_dir": comparison_run_dir,
        "baseline_config_path": baseline_config_path,
        "comparison_config_path": comparison_config_path,
        "fixture_config_path": fixture_config_path,
        "multi_manifest_path": multi_manifest_path,
        "gate_spec_path": gate_spec_path,
        "power_comparison_doc_path": power_comparison_doc_path,
    }


@pytest.mark.unit
@pytest.mark.determinism("d0")
def test_rg09_power_analysis_emits_required_outputs(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    if not _required_docs_exist():
        pytest.skip("checked-in RG-09 power-analysis inputs are not present in this workspace")

    from pysrc.meta.rg09_power_analysis import run_rg09_power_analysis

    output_dir = tmp_path / "rg09_power_analysis"
    result = run_rg09_power_analysis(
        baseline_run_dir=_baseline_run_dir(),
        comparison_run_dirs=[_comparison_run_dir()],
        output_dir=output_dir,
    )

    assert result.output_dir == output_dir
    payload = json.loads((output_dir / "rg09_power_analysis.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output_dir / "rg09_power_analysis_manifest.json").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == "rg09.power_analysis.v1"
    assert payload["artifact_class"] == "advisory_research_only"
    assert payload["non_promotable"] is True
    assert payload["active_lane_untouched"] is True
    assert payload["baseline_geometry_assumed"] is True
    assert payload["fixture_config_version"] == "rg09_v1.1.0"
    assert payload["baseline_fold_surface"]["admissible"] is True
    assert payload["baseline_fold_surface"]["status"] == "corrected_calendar_time_surface"
    assert payload["baseline_fold_surface"]["uniform_calendar_day_index"] is False
    assert payload["baseline_fold_surface"]["calendar_overlap_policy"] == "independent_instruments"
    assert payload["advisory_recommendation"] == "insufficient_artifacts_for_power_estimation"
    assert payload["power_curve_by_episode_count"] == []
    assert payload["power_curve_by_instrument_count"] == []
    assert any("corrected surface" in item.lower() for item in payload["limitations"])

    assert manifest["advisory_only"] is True
    assert (
        manifest["baseline_run_path"]
        .replace("\\", "/")
        .endswith("runs/rg09_v2/empirical_v2bocpd_directional_nme")
    )
    assert [path.replace("\\", "/") for path in manifest["comparison_run_paths"]] == [
        str(_comparison_run_dir().resolve()).replace("\\", "/")
    ]
    assert "fixture_sidecars" in manifest["inputs"]


@pytest.mark.unit
@pytest.mark.determinism("d0")
def test_rg09_power_analysis_records_corrected_fold_surface(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    ws = _write_advisory_workspace(tmp_path, contaminated=False)

    from pysrc.meta.rg09_power_analysis import run_rg09_power_analysis

    output_dir = tmp_path / "out"
    result = run_rg09_power_analysis(
        baseline_run_dir=ws["baseline_run_dir"],
        comparison_run_dirs=[ws["comparison_run_dir"]],
        output_dir=output_dir,
        gate_spec_path=ws["gate_spec_path"],
        baseline_config_path=ws["baseline_config_path"],
        comparison_config_paths=[ws["comparison_config_path"]],
        fixture_config_path=ws["fixture_config_path"],
        power_comparison_doc_path=ws["power_comparison_doc_path"],
        multi_fixture_manifest_path=ws["multi_manifest_path"],
    )

    assert result.output_dir == output_dir
    payload = json.loads((output_dir / "rg09_power_analysis.json").read_text(encoding="utf-8"))
    surface = payload["baseline_fold_surface"]
    assert surface["admissible"] is True
    assert surface["status"] == "corrected_calendar_time_surface"
    assert surface["uniform_calendar_day_index"] is False
    assert surface["fold_construction"]["method"] == "calendar_time"
    assert payload["observed_baseline"]["admissible_episode_count"] == 40


@pytest.mark.unit
@pytest.mark.determinism("d0")
def test_rg09_power_analysis_marks_contaminated_fold_surface_inadmissible(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    ws = _write_advisory_workspace(tmp_path, contaminated=True)

    from pysrc.meta.rg09_power_analysis import run_rg09_power_analysis

    output_dir = tmp_path / "out_contaminated"
    result = run_rg09_power_analysis(
        baseline_run_dir=ws["baseline_run_dir"],
        comparison_run_dirs=[ws["comparison_run_dir"]],
        output_dir=output_dir,
        gate_spec_path=ws["gate_spec_path"],
        baseline_config_path=ws["baseline_config_path"],
        comparison_config_paths=[ws["comparison_config_path"]],
        fixture_config_path=ws["fixture_config_path"],
        power_comparison_doc_path=ws["power_comparison_doc_path"],
        multi_fixture_manifest_path=ws["multi_manifest_path"],
    )

    assert result.output_dir == output_dir
    payload = json.loads((output_dir / "rg09_power_analysis.json").read_text(encoding="utf-8"))
    surface = payload["baseline_fold_surface"]
    assert payload["advisory_recommendation"] == "inadmissible_contaminated_fixture_surface"
    assert surface["admissible"] is False
    assert surface["status"] == "inadmissible_contaminated_fixture_surface"
    assert surface["uniform_calendar_day_index"] is True
    assert any("contaminated" in item.lower() for item in payload["limitations"])
    assert payload["power_curve_by_episode_count"] == []


@pytest.mark.unit
@pytest.mark.determinism("d0")
def test_rg09_power_analysis_markdown_uses_computed_baseline_instrument_count(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    ws = _write_advisory_workspace(tmp_path, contaminated=False)

    import pysrc.meta.rg09_power_analysis as power_analysis

    monkeypatch.setattr(power_analysis, "EPISODE_SCENARIOS", (50,))
    monkeypatch.setattr(power_analysis, "INSTRUMENT_SCENARIOS", (8,))
    monkeypatch.setattr(
        power_analysis,
        "_estimate_power",
        lambda **_kwargs: power_analysis._PowerEstimate(
            admissible_episode_count=50,
            statistical_pass_probability=0.10,
            statistical_pass_probability_lower=0.05,
            statistical_pass_probability_upper=0.15,
            functional_pass_probability=0.20,
            functional_pass_probability_lower=0.15,
            functional_pass_probability_upper=0.25,
            decisive_pass_probability=0.05,
            decisive_pass_probability_lower=0.02,
            decisive_pass_probability_upper=0.08,
            weakest_link_fold_id=1,
            weakest_link_family="shuffled_label",
            weakest_link_pass_probability=0.10,
        ),
    )
    monkeypatch.setattr(power_analysis, "_find_episode_target", lambda **_kwargs: None)

    output_dir = tmp_path / "out_markdown_count"
    result = power_analysis.run_rg09_power_analysis(
        baseline_run_dir=ws["baseline_run_dir"],
        comparison_run_dirs=[ws["comparison_run_dir"]],
        output_dir=output_dir,
        gate_spec_path=ws["gate_spec_path"],
        baseline_config_path=ws["baseline_config_path"],
        comparison_config_paths=[ws["comparison_config_path"]],
        fixture_config_path=ws["fixture_config_path"],
        power_comparison_doc_path=ws["power_comparison_doc_path"],
        multi_fixture_manifest_path=ws["multi_manifest_path"],
    )

    assert result.output_dir == output_dir
    report = (output_dir / "rg09_power_analysis.md").read_text(encoding="utf-8")
    assert "across 2 instruments" in report
    assert "across 8 instruments" not in report


@pytest.mark.unit
@pytest.mark.determinism("d0")
def test_run_rg09_power_analysis_cli_writes_markdown_with_required_recommendation(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    if not _required_docs_exist():
        pytest.skip("checked-in RG-09 power-analysis inputs are not present in this workspace")

    from scripts.run_rg09_power_analysis import run_rg09_power_analysis_cli

    output_dir = tmp_path / "rg09_power_analysis_cli"
    exit_code = run_rg09_power_analysis_cli(
        baseline_run_dir=_baseline_run_dir(),
        comparison_run_dirs=[_comparison_run_dir()],
        output_dir=output_dir,
    )

    assert exit_code == 0
    report = (output_dir / "rg09_power_analysis.md").read_text(encoding="utf-8").rstrip()
    assert "## A. Purpose" in report
    assert "## D. Power study" in report
    assert "Baseline fold-surface status: `corrected_calendar_time_surface`" in report
    assert report.endswith(
        "Do not use this advisory for scope planning; the corrected governed reruns did not yield estimable inputs"
    )


@pytest.mark.unit
@pytest.mark.determinism("d0")
def test_power_analysis_loads_transition_structural_measure_from_gate_result(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    ws = _write_advisory_workspace(tmp_path, contaminated=False)

    transition_run_dir = _write_run_dir(
        tmp_path / "runs" / "transition",
        fixture_summary_path=ws["baseline_run_dir"].parent.parent
        / "fixtures"
        / "rg09"
        / "v2"
        / "rg09_fixture_summary.json",
        fixture_sha256="sha256:test-fixture",
        source_dataset_id="rg09_v2:test;entities=ES,NQ;apply_diversification=false",
        config_id="rg09_pilot_config_v1_transition",
        episode_count=40,
        episode_construction="transition_anchored",
        structural_measure="direction_score",
    )

    from pysrc.meta.rg09_power_analysis import _load_observed_run

    observed = _load_observed_run(transition_run_dir)

    assert observed.structural_measure == "direction_score"
    assert observed.per_fold[0].structural_measure == "direction_score"
