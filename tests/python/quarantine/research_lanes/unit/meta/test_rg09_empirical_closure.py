"""Unit tests for RG-09 empirical closure lane."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
from tests.python.unit.meta.test_rg09_harness import (
    _pilot_config_payload,
    _read_json,
    _write_fixture_bundle,
    _write_json,
)

from pysrc.meta.rg09_parquet_io import read_rg09_fixture_parquet, write_rg09_fixture_parquet


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_empirical_closure_emits_required_artifacts(
    tmp_path: Path, deterministic_seed: int
) -> None:
    paths = _write_fixture_bundle(tmp_path)
    out = tmp_path / "emp_out"
    from pysrc.meta.rg09_empirical_closure import (
        META_VALIDITY_RESEARCH_FILENAME,
        TASK_MANIFEST_RESEARCH_FILENAME,
        run_rg09_empirical_closure,
    )

    result = run_rg09_empirical_closure(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=out,
    )
    assert result.base_decision in (
        "PASS",
        "FAIL_KILL",
        "NEEDS_MORE_EVIDENCE",
        "PRECONDITION_FAILED",
        None,
    )
    for name in (
        "rg09_gate_result.json",
        "rg09_empirical_summary.json",
        "rg09_empirical_detail.json",
        "rg09_empirical_report.md",
        META_VALIDITY_RESEARCH_FILENAME,
        "run_config_resolved.json",
        "machine_manifest.json",
        "rg09_surface_manifest.json",
        "task_manifest.json",
        "meta_validity_report.json",
        "execution_assumptions.json",
        TASK_MANIFEST_RESEARCH_FILENAME,
    ):
        assert (out / name).exists(), name
    summary = _read_json(out / "rg09_empirical_summary.json")
    assert summary["non_promotable"] is True
    assert summary["artifact_class"] == "deterministic_evidence"
    assert summary["crisis_definition"] == "severity_gated_p90"
    assert summary["reference_condition"] == "diag_regime_class_bocpd_gated"
    assert "projection_rule_version" in summary
    assert "projection_rule_text" not in summary
    meta = _read_json(out / META_VALIDITY_RESEARCH_FILENAME)
    assert meta["schema_version"] == "rg09.meta_validity_report.research.v1"
    assert meta["non_promotable"] is True
    assert meta["artifact_class"] == "deterministic_evidence"
    assert meta["overall_result"] == "RESEARCH_SCAFFOLD_INCOMPLETE"
    assert meta["overall_conclusion_research"] == summary["supports_non_exchangeability"]
    assert meta["inner_loop_gain_harvey_t"]["status"] == "unavailable"
    assert "ic_harvey_t" in meta
    for key in (
        "encoder_clustering",
        "proxy_ic_correlation",
        "forgetting_delta",
        "confidence_ece",
        "net_allocation_sharpe",
    ):
        assert meta[key]["status"] == "unavailable"
    assert meta["signal_set_version"]["status"] == "unavailable"
    assert "task_manifest.json" not in meta
    tm_ii0a = _read_json(out / "rg09_surface_manifest.json")
    tm_mln06 = _read_json(out / "task_manifest.json")
    assert tm_mln06.get("artifact_version") is not None
    assert "research_schema_version" not in tm_ii0a
    tm_res = _read_json(out / TASK_MANIFEST_RESEARCH_FILENAME)
    assert tm_res.get("research_schema_version") == "rg09.task_manifest.research.v1"
    assert tm_res.get("mln_01_status") == "open"
    assert "schema_version" not in tm_res
    assert "row_count_after_exclusions" not in tm_res
    assert "exclusion_counts_by_code" not in tm_res
    assert "fixture_identity" in tm_res
    assert tm_res.get("pilot_config_id") == "rg09_pilot_config_test"
    mm = _read_json(out / "machine_manifest.json")
    assert mm.get("wrapper_role") == "empirical_inventory"
    assert mm.get("wrapper_manifest_filename") == "machine_manifest.json"
    assert "machine_manifest.json" not in mm.get("empirical_research_output_filenames", [])
    assert "ii0a_contract_filenames" in mm
    rc = _read_json(out / "run_config_resolved.json")
    assert rc["artifact_class"] == "run_local_provenance"
    assert rc["successor_hypotheses"]["enabled"] is False
    prov = rc["segmentation_provenance"]
    assert prov["boundary_recovery_mode"] is None
    assert prov["rerun_contract_episode_grouping_effective"] == "stable_span"
    assert prov["pilot_episode_construction"] == "stable_span"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_segmentation_provenance_records_boundary_recovery_v2(
    tmp_path: Path, deterministic_seed: int
) -> None:
    paths = _write_fixture_bundle(tmp_path)
    out = tmp_path / "emp_boundary_v2"
    from pysrc.meta.rg09_boundary_treatment import RG09BoundaryRecoverySpec
    from pysrc.meta.rg09_empirical_closure import run_rg09_empirical_closure

    run_rg09_empirical_closure(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=out,
        boundary_recovery=RG09BoundaryRecoverySpec(mode="boundary_recovery_v2_enforced_dwell"),
    )
    rc = _read_json(out / "run_config_resolved.json")
    prov = rc["segmentation_provenance"]
    assert prov["pilot_episode_construction"] == "stable_span"
    assert prov["boundary_recovery_mode"] == "boundary_recovery_v2_enforced_dwell"
    assert prov["rerun_contract_episode_grouping_effective"] == "boundary_recovery_v2"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_sanctioned_internal_harness_seam_name(tmp_path: Path, deterministic_seed: int) -> None:
    import pysrc.meta.rg09_harness as rh

    assert hasattr(rh, "run_rg09_harness_internal")
    assert hasattr(rh, "run_rg09_harness")
    assert not hasattr(rh, "_run_rg09_harness_impl")


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_empirical_summary_deterministic(tmp_path: Path, deterministic_seed: int) -> None:
    paths = _write_fixture_bundle(tmp_path)
    from pysrc.meta.rg09_empirical_closure import run_rg09_empirical_closure

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run_rg09_empirical_closure(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=out_a,
    )
    run_rg09_empirical_closure(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=out_b,
    )
    assert _read_json(out_a / "rg09_empirical_summary.json") == _read_json(
        out_b / "rg09_empirical_summary.json"
    )
    assert _read_json(out_a / "rg09_empirical_detail.json") == _read_json(
        out_b / "rg09_empirical_detail.json"
    )
    mv_a = _read_json(out_a / "meta_validity_report_research.json")
    mv_b = _read_json(out_b / "meta_validity_report_research.json")
    assert mv_a == mv_b


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_null_invalid_base_run_is_reported_as_inconclusive(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    import pysrc.meta.rg09_empirical_closure as closure
    from pysrc.meta.rg09_harness import RG09HarnessRunData, load_rg09_config

    paths = _write_fixture_bundle(tmp_path)
    frame = read_rg09_fixture_parquet(paths["fixture"])
    summary = _read_json(paths["summary"])
    metadata = _read_json(paths["metadata"])
    config = load_rg09_config(paths["config"])
    gate_result = {
        "decision": None,
        "fail_codes": ["FAIL_NULL_DISTRIBUTION_INVALID"],
        "evidence": {
            "leakage_geometry": {"clean": True, "reproducibility_consistent": True},
            "non_exchangeability": {
                "statistical_pass": False,
                "structural_pass": True,
                "functional_pass": False,
                "structural_contamination": False,
                "structural_separability_ratio": 1.5,
            },
            "null_collapse": {
                "null_distribution_valid": False,
                "invalid_families": [{"fold_id": 0, "families": ["shuffled_label"]}],
                "functional_evidence": {"admissible": True},
            },
        },
    }
    run_data = RG09HarnessRunData(
        output_dir=tmp_path / "emp_out",
        decision=None,
        config=config,
        summary=summary,
        metadata=metadata,
        fixture_sha256=str(summary["fixture_sha256"]),
        config_version=str(summary["config_version"]),
        generation_timestamp=str(summary["generation_timestamp"]),
        frame=frame,
        base_episodes=pd.DataFrame(),
        exclusion_counts={},
        gate_result=gate_result,
        fixture_validation_fail_codes=[],
    )

    monkeypatch.setattr(closure, "run_rg09_harness_internal", lambda **_kwargs: run_data)
    monkeypatch.setattr(closure, "_derive_episodes", lambda *_args, **_kwargs: (pd.DataFrame(), {}))
    monkeypatch.setattr(
        closure,
        "_precondition_fail_codes",
        lambda *_args, **_kwargs: ["FAIL_INSUFFICIENT_EPISODES"],
    )

    out = tmp_path / "emp_out"
    result = closure.run_rg09_empirical_closure(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=out,
    )

    summary_payload = _read_json(out / "rg09_empirical_summary.json")
    assert result.base_decision is None
    assert summary_payload["base_decision"] is None
    assert summary_payload["supports_non_exchangeability"] == "inconclusive"
    assert "FAIL_NULL_DISTRIBUTION_INVALID" in summary_payload["fail_codes"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_run_local_provenance_may_differ_by_output_dir(
    tmp_path: Path, deterministic_seed: int
) -> None:
    paths = _write_fixture_bundle(tmp_path)
    from pysrc.meta.rg09_empirical_closure import run_rg09_empirical_closure

    out_a = tmp_path / "pa"
    out_b = tmp_path / "pb"
    run_rg09_empirical_closure(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=out_a,
    )
    run_rg09_empirical_closure(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=out_b,
    )
    mm_a = _read_json(out_a / "machine_manifest.json")
    mm_b = _read_json(out_b / "machine_manifest.json")
    assert mm_a["artifact_class"] == "run_local_provenance"
    ext_a = mm_a["extends_ii0a_machine_manifest"]["outputs"][0]["path"]
    ext_b = mm_b["extends_ii0a_machine_manifest"]["outputs"][0]["path"]
    assert ext_a != ext_b


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_malformed_successor_hypotheses_fail_closed(
    tmp_path: Path, deterministic_seed: int
) -> None:
    paths = _write_fixture_bundle(tmp_path)
    bad_cfg = _pilot_config_payload()
    bad_cfg["successor_hypotheses"] = {"enabled": "not-a-bool"}
    cfg_path = _write_json(tmp_path / "bad_succ.json", bad_cfg)
    out = tmp_path / "out"
    from pysrc.meta.rg09_empirical_closure import run_rg09_empirical_closure

    result = run_rg09_empirical_closure(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=cfg_path,
        output_dir=out,
    )
    assert result.fail_closed is True
    assert "FAIL_EMPIRICAL_SUCCESSOR_CONFIG_INVALID" in result.fail_codes
    rc = _read_json(out / "run_config_resolved.json")
    assert rc["successor_hypotheses"]["status"] == "invalid"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_validate_successor_overlay_valid_merges(tmp_path: Path, deterministic_seed: int) -> None:
    from pysrc.meta.rg09_empirical_closure import validate_successor_hypotheses_overlay

    merged, codes = validate_successor_hypotheses_overlay({})
    assert not codes
    assert merged["enabled"] is False

    full = {
        "successor_hypotheses": {
            "enabled": False,
            "allow_single_axis_only": True,
            "candidates": [
                {"id": "RG09-H2", "axis": "cross_sectional_universe"},
                {"id": "RG09-H3", "axis": "granularity_shift"},
                {"id": "RG09-H4", "axis": "market_class_shift"},
            ],
        }
    }
    merged2, codes2 = validate_successor_hypotheses_overlay(full)
    assert not codes2
    assert merged2["enabled"] is False

    dup = {
        "successor_hypotheses": {
            "enabled": False,
            "allow_single_axis_only": True,
            "candidates": [
                {"id": "RG09-H2", "axis": "cross_sectional_universe"},
                {"id": "RG09-H2", "axis": "granularity_shift"},
                {"id": "RG09-H4", "axis": "market_class_shift"},
            ],
        }
    }
    assert validate_successor_hypotheses_overlay(dup)[1] == [
        "FAIL_EMPIRICAL_SUCCESSOR_CONFIG_INVALID"
    ]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_empirical_fail_closed_missing_bocpd_column(
    tmp_path: Path, deterministic_seed: int
) -> None:

    from scripts.generate_rg09_fixture import _compute_fixture_sha256
    from tests.python.unit.meta.test_rg09_harness import (
        _episode_frame,
        _write_json,
    )

    frame = _episode_frame().drop(columns=["diag_regime_class_bocpd_gated"])
    fixture_path = tmp_path / "rg09_fixture_v1.parquet"
    write_rg09_fixture_parquet(frame, fixture_path, index=False)
    sha = _compute_fixture_sha256(frame)
    summary = {
        "amendment": "MLN-02-AMD-01",
        "config_version": "rg09_v1.0.2",
        "crisis_episodes_after_cold_start": 0,
        "crisis_label_agreement_rate": 1.0,
        "date_range_end": str(frame["decision_ts"].max().date()),
        "date_range_start": str(frame["decision_ts"].min().date()),
        "entity_id": "ES",
        "fixture_sha256": sha,
        "generation_timestamp": "2026-03-31T00:00:00+00:00",
        "projection_rule": "vol_hi AND severity_flag (p90)",
        "row_count": len(frame),
        "row_counts_by_class": {
            str(k): int(v) for k, v in frame.groupby("regime_class", sort=True).size().items()
        },
        "row_counts_by_class_bocpd_gated": {"bull": 0},
        "source_dataset_id": "synthetic",
    }
    metadata = {
        "amendment": "MLN-02-AMD-01",
        "fixture_sha256": sha,
        "generation_timestamp": "2026-03-31T00:00:00+00:00",
        "config_hash": "hmac-sha256:test",
        "source_hash": "sha256:test",
    }
    summary_path = _write_json(tmp_path / "rg09_fixture_summary.json", summary)
    metadata_path = _write_json(tmp_path / "rg09_fixture_metadata.json", metadata)
    config_path = _write_json(tmp_path / "cfg.json", _pilot_config_payload())
    from pysrc.meta.rg09_empirical_closure import run_rg09_empirical_closure

    out = tmp_path / "emp"
    result = run_rg09_empirical_closure(
        fixture_path=fixture_path,
        fixture_summary_path=summary_path,
        fixture_metadata_path=metadata_path,
        config_path=config_path,
        output_dir=out,
    )
    assert result.fail_closed is True
    assert "FAIL_REFERENCE_REGIME_COLUMN_MISSING" in result.fail_codes
    detail = _read_json(out / "rg09_empirical_detail.json")
    assert "FAIL_REFERENCE_REGIME_COLUMN_MISSING" in detail["fail_codes"]
    for c in detail["fail_codes"]:
        assert "EMPRICAL" not in c


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_derive_episodes_rejects_missing_regime_column(
    tmp_path: Path, deterministic_seed: int
) -> None:
    from pysrc.meta.rg09_harness import _derive_episodes, load_rg09_config

    paths = _write_fixture_bundle(tmp_path)
    config = load_rg09_config(paths["config"])

    bad = read_rg09_fixture_parquet(paths["fixture"]).drop(columns=["regime_class"])
    with pytest.raises(ValueError, match="missing regime class column"):
        _derive_episodes(bad, config, regime_class_column="regime_class")


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_load_run_config_overlay_successor_defaults(
    tmp_path: Path, deterministic_seed: int
) -> None:
    paths = _write_fixture_bundle(tmp_path)
    from pysrc.meta.rg09_empirical_closure import (
        DEFAULT_SUCCESSOR_HYPOTHESES,
        load_run_config_overlay,
    )

    overlay = load_run_config_overlay(paths["config"])
    assert "successor_hypotheses" not in overlay or isinstance(
        overlay.get("successor_hypotheses"), dict
    )
    merged = {**DEFAULT_SUCCESSOR_HYPOTHESES, **overlay.get("successor_hypotheses", {})}
    assert merged["allow_single_axis_only"] is True
