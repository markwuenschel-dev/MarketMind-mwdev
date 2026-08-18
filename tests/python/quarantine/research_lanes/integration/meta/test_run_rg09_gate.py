"""Integration tests for the RG-09 gate CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.python.unit.meta.test_rg09_harness import (
    _dummy_episode_manifest_frame,
    _write_fixture_bundle,
)

from marketmind_gate.gates.meta_learner_scaffold import (
    REASON_META_VALIDITY_OVERALL_REDACTED,
    evaluate_meta_learner_scaffold,
)
from pysrc.meta.phase2_artifact_contract import PHASE2_ARTIFACT_CONTRACT_VERSION
from pysrc.meta.seed_policy import build_run_identity, scaffold_int_seed_from_content_tag
from pysrc.meta.threshold_governance import clear_threshold_register_cache
from pysrc.meta_learning.task_generator import compute_task_id


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _validated_register_payload() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    payload = json.loads(
        (root / "pysrc/meta/threshold_register.mln07.v1.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload["records"], list)
    for row in payload["records"]:
        assert isinstance(row, dict)
        if row["threshold_id"] in {"THR-RG09-V03", "THR-RG09-V15", "THR-RG09-V19"}:
            row["state"] = "VALIDATED"
    return payload


def _install_validated_register(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pysrc.meta.threshold_governance as tg

    register_path = tmp_path / "threshold_register.mln07.v1.json"
    register_path.write_text(json.dumps(_validated_register_payload(), indent=2), encoding="utf-8")
    clear_threshold_register_cache()
    monkeypatch.setattr(tg, "default_register_path", lambda: register_path)


def _install_register_threshold_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    threshold_id: str,
    state: str,
) -> None:
    import pysrc.meta.threshold_governance as tg

    payload = _validated_register_payload()
    for row in payload["records"]:
        assert isinstance(row, dict)
        if row["threshold_id"] == threshold_id:
            row["state"] = state
    register_path = tmp_path / "threshold_register_override.json"
    register_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    clear_threshold_register_cache()
    monkeypatch.setattr(tg, "default_register_path", lambda: register_path)


@pytest.mark.integration
@pytest.mark.determinism("d2")
def test_run_rg09_gate_emits_required_artifacts(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    monkeypatch = pytest.MonkeyPatch()
    _install_validated_register(tmp_path, monkeypatch)
    from scripts.run_rg09_gate import run_rg09_gate

    output_dir = tmp_path / "out"
    result = run_rg09_gate(
        fixture_path=Path("fixtures/rg09/v1/rg09_fixture_v1.parquet"),
        fixture_summary_path=Path("fixtures/rg09/v1/rg09_fixture_summary.json"),
        fixture_metadata_path=Path("fixtures/rg09/v1/rg09_fixture_metadata.json"),
        config_path=Path("docs/rg09/rg09_pilot_config_v1.json"),
        output_dir=output_dir,
    )
    assert result.output_dir == output_dir
    assert (output_dir / "rg09_gate_result.json").exists()
    assert (output_dir / "rg09_surface_manifest.json").exists()
    assert (output_dir / "task_manifest.json").exists()
    assert (output_dir / "meta_validity_report.json").exists()
    assert (output_dir / "execution_assumptions.json").exists()
    assert (output_dir / "rg09_diagnostics.json").exists()
    assert (output_dir / "implementation_brief.machine.json").exists()
    gate_result = _read_json(output_dir / "rg09_gate_result.json")
    assert gate_result["hypothesis_id"] == "RG09-H1"
    assert (
        gate_result["fixture_sha256"]
        == "sha256:3ba47a1ef9445f1694c411151219bd2935f2d6b1b075e33a6c5fe13f903c4553"
    )
    assert gate_result["gate_executed"] is False
    assert gate_result["decision"] is None
    assert "FAIL_INSUFFICIENT_EPISODES" in gate_result["fail_codes"]
    diagnostics = _read_json(output_dir / "rg09_diagnostics.json")
    assert diagnostics["threshold_governance_preflight"]["passed"] is True
    mv_report = _read_json(output_dir / "meta_validity_report.json")
    assert mv_report["overall_result"] == "INSUFFICIENT"
    sc_res = evaluate_meta_learner_scaffold(output_dir)
    sc_payload = sc_res.to_json_dict()
    assert sc_payload["status"] == "SCAFFOLD_INCOMPLETE"
    assert sc_payload["promotable_claim_emitted"] is False
    assert REASON_META_VALIDITY_OVERALL_REDACTED in sc_payload["incomplete_reason_codes"]
    assert sc_payload["meta_validity_report"]["overall_result_shell_visible"] is None
    assert sc_payload["meta_validity_report"]["overall_result_non_scaffold_redacted"] is True
    assert "PASS" not in json.dumps(sc_payload, sort_keys=True)
    monkeypatch.undo()


@pytest.mark.integration
@pytest.mark.determinism("d2")
def test_rg09_governed_task_manifest_triple_contract_fixture_backed(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase II task_manifest frozen fields and MLN-06 triple identity match meta/execution (fixture v1)."""
    _ = deterministic_seed
    _install_validated_register(tmp_path, monkeypatch)
    from scripts.run_rg09_gate import run_rg09_gate

    output_dir = tmp_path / "out_triple"
    run_rg09_gate(
        fixture_path=Path("fixtures/rg09/v1/rg09_fixture_v1.parquet"),
        fixture_summary_path=Path("fixtures/rg09/v1/rg09_fixture_summary.json"),
        fixture_metadata_path=Path("fixtures/rg09/v1/rg09_fixture_metadata.json"),
        config_path=Path("docs/rg09/rg09_pilot_config_v1.json"),
        output_dir=output_dir,
    )
    gate_result = _read_json(output_dir / "rg09_gate_result.json")
    fixture_sha = str(gate_result["fixture_sha256"])
    expected_seed = scaffold_int_seed_from_content_tag(fixture_sha)
    expected_rid = build_run_identity(expected_seed)

    task_doc = _read_json(output_dir / "task_manifest.json")
    meta_doc = _read_json(output_dir / "meta_validity_report.json")
    exec_doc = _read_json(output_dir / "execution_assumptions.json")

    for label, doc in (
        ("task_manifest.json", task_doc),
        ("meta_validity_report.json", meta_doc),
        ("execution_assumptions.json", exec_doc),
    ):
        assert doc.get("artifact_version") == PHASE2_ARTIFACT_CONTRACT_VERSION, label
        assert doc.get("run_id") == expected_rid.run_id, label
        assert isinstance(doc.get("timestamp"), str) and doc["timestamp"].strip(), label

    ts0 = task_doc["timestamp"]
    assert meta_doc["timestamp"] == ts0
    assert exec_doc["timestamp"] == ts0

    for doc in (task_doc, meta_doc, exec_doc):
        ri = doc["run_identity"]
        assert isinstance(ri, dict)
        assert ri["seed"] == expected_seed
        assert ri["run_id"] == expected_rid.run_id

    tasks = task_doc["tasks"]
    assert isinstance(tasks, list)
    assert len(tasks) >= 1
    row = tasks[0]
    frozen = (
        "task_id",
        "regime_id",
        "regime_class",
        "t0",
        "t1",
        "pit_boundary",
        "signal_ids_hash",
        "signal_set_version",
    )
    for key in frozen:
        assert key in row, key
        assert row[key] is not None and str(row[key]).strip(), key

    assert row["task_id"] == compute_task_id(
        regime_id=str(row["regime_id"]),
        t0=str(row["t0"]),
        t1=str(row["t1"]),
        signal_ids_hash=str(row["signal_ids_hash"]),
    )
    assert row["pit_boundary"] == meta_doc["pit_boundary"]
    assert row["signal_ids_hash"] == meta_doc["signal_ids_hash"]
    surface = meta_doc["signal_surface"]
    assert isinstance(surface, dict)
    assert surface["signal_set_version"] == row["signal_set_version"]
    assert surface["signal_ids_hash"] == row["signal_ids_hash"]
    monkeypatch.undo()


@pytest.mark.integration
@pytest.mark.determinism("d2")
def test_base_needs_more_evidence_emits_successor_authorization(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    _install_validated_register(tmp_path, monkeypatch)
    from scripts.run_rg09_gate import run_rg09_gate

    import pysrc.meta.rg09_harness as harness

    paths = _write_fixture_bundle(tmp_path)
    bundle, _ = harness._load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
    )

    def _fake_evaluation(*_: Any, **__: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        evidence = {
            "leakage_geometry": {"clean": True},
            "non_exchangeability": {
                "statistical_pass": True,
                "structural_pass": True,
                "functional_pass": False,
                "structural_contamination": False,
                "structural_separability_ratio": 1.5,
            },
            "null_collapse": {},
        }
        diagnostics: dict[str, Any] = {"authorized_null_families": []}
        return evidence, diagnostics

    monkeypatch.setattr(harness, "_load_and_validate_fixture", lambda **_kwargs: (bundle, []))
    monkeypatch.setattr(
        harness, "_derive_episodes", lambda *_args, **_kwargs: (_dummy_episode_manifest_frame(), {})
    )
    monkeypatch.setattr(harness, "_precondition_fail_codes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(harness, "_evaluate_base_hypothesis", _fake_evaluation)
    output_dir = tmp_path / "out"
    result = run_rg09_gate(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=output_dir,
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    assert gate_result["decision"] == "NEEDS_MORE_EVIDENCE"
    assert gate_result["gate_executed"] is True
    assert gate_result["successor_hypotheses"]["eligible"] is True
    assert gate_result["executed_hypotheses"] == ["RG09-H1"]
    manifest = _read_json(result.output_dir / "implementation_brief.machine.json")
    assert manifest["outputs"][0]["path"].startswith(str(output_dir))
    assert manifest["tests"][0]["file"] == "tests/python/unit/meta/test_rg09_harness.py"


@pytest.mark.integration
@pytest.mark.determinism("d0")
def test_governed_harness_fails_when_mln06_emit_raises(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    _install_validated_register(tmp_path, monkeypatch)
    from scripts.run_rg09_gate import run_rg09_gate

    import pysrc.meta.rg09_phase2_bridge as phase2_bridge
    from pysrc.meta.phase2_artifact_contract import PhaseIIArtifactError

    def _boom(*_: object, **__: object) -> None:
        raise PhaseIIArtifactError("forced failure for integration test")

    monkeypatch.setattr(phase2_bridge, "emit_phase2_artifacts", _boom)
    output_dir = tmp_path / "out"
    with pytest.raises(PhaseIIArtifactError):
        run_rg09_gate(
            fixture_path=Path("fixtures/rg09/v1/rg09_fixture_v1.parquet"),
            fixture_summary_path=Path("fixtures/rg09/v1/rg09_fixture_summary.json"),
            fixture_metadata_path=Path("fixtures/rg09/v1/rg09_fixture_metadata.json"),
            config_path=Path("docs/rg09/rg09_pilot_config_v1.json"),
            output_dir=output_dir,
        )
    monkeypatch.undo()


@pytest.mark.integration
@pytest.mark.determinism("d2")
def test_invalid_null_distribution_emits_executed_null_decision(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    _install_validated_register(tmp_path, monkeypatch)
    from scripts.run_rg09_gate import run_rg09_gate

    import pysrc.meta.rg09_harness as harness

    paths = _write_fixture_bundle(tmp_path)
    bundle, _ = harness._load_and_validate_fixture(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
    )

    def _invalid_eval(*_: Any, **__: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        evidence = {
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
        }
        diagnostics: dict[str, Any] = {
            "authorized_null_families": ["shuffled_label"],
            "fold_0": {
                "shuffled_label": {
                    "distinct_draw_count": 1,
                    "null_range": 0.0,
                }
            },
        }
        return evidence, diagnostics

    monkeypatch.setattr(harness, "_load_and_validate_fixture", lambda **_kwargs: (bundle, []))
    monkeypatch.setattr(
        harness, "_derive_episodes", lambda *_args, **_kwargs: (_dummy_episode_manifest_frame(), {})
    )
    monkeypatch.setattr(harness, "_precondition_fail_codes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(harness, "_evaluate_base_hypothesis", _invalid_eval)

    output_dir = tmp_path / "out"
    result = run_rg09_gate(
        fixture_path=paths["fixture"],
        fixture_summary_path=paths["summary"],
        fixture_metadata_path=paths["metadata"],
        config_path=paths["config"],
        output_dir=output_dir,
    )
    gate_result = _read_json(result.output_dir / "rg09_gate_result.json")
    assert gate_result["gate_executed"] is True
    assert gate_result["decision"] is None
    assert gate_result["fail_codes"] == ["FAIL_NULL_DISTRIBUTION_INVALID"]
    assert gate_result["successor_hypotheses"]["reason"] == "invalid_evidence_surface"


@pytest.mark.integration
@pytest.mark.determinism("d2")
def test_run_rg09_gate_fails_closed_when_gate_critical_thresholds_remain_provisional(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    _install_register_threshold_state(
        tmp_path,
        monkeypatch,
        threshold_id="THR-RG09-V03",
        state="PROVISIONAL",
    )
    from scripts.run_rg09_gate import run_rg09_gate

    from pysrc.meta.threshold_governance import ThresholdGovernanceError

    with pytest.raises(ThresholdGovernanceError):
        run_rg09_gate(
            fixture_path=Path("fixtures/rg09/v1/rg09_fixture_v1.parquet"),
            fixture_summary_path=Path("fixtures/rg09/v1/rg09_fixture_summary.json"),
            fixture_metadata_path=Path("fixtures/rg09/v1/rg09_fixture_metadata.json"),
            config_path=Path("docs/rg09/rg09_pilot_config_v1.json"),
            output_dir=tmp_path / "out_fail",
        )
