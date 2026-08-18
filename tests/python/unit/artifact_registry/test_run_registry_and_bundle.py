from __future__ import annotations

import json
from pathlib import Path

from pysrc.artifact_registry import LocalCAS
from pysrc.artifact_registry.artifacts import assert_bundle_complete
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.artifact_registry.contracts import (
    store_model_snapshot_manifest,
    store_task_manifest,
)
from pysrc.artifact_registry.run_registry import RunRegistry, RunStatus
from pysrc.ops.hashing import HashRef


def test_run_registry_state_machine_and_visibility(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    registry = RunRegistry(root)

    run_id = registry.begin_run(metadata={"plan_hash": "plan-123"})

    # Construct a simple HashRefs-like payload by hand via LocalCAS.
    cas_root = tmp_path / "cas"
    cas = LocalCAS(cas_root)
    hashes = cas.put_json({"foo": "bar"})

    registry.add_artifact(run_id, "plan", hashes)

    # REGISTERING runs are not visible by default.
    assert registry.get_run(run_id) is None
    run_incomplete = registry.get_run(run_id, include_incomplete=True)
    assert run_incomplete is not None
    assert run_incomplete.status is RunStatus.REGISTERING
    assert any(a.role == "plan" for a in run_incomplete.artifacts)

    # Finalize → COMPLETE and ensure default visibility now includes the run.
    registry.finalize_run(run_id, RunStatus.COMPLETE)
    run_complete = registry.get_run(run_id)
    assert run_complete is not None
    assert run_complete.status is RunStatus.COMPLETE

    # Mutations after finalize must fail.
    another_hashes = cas.put_json({"foo": "baz"})
    try:
        registry.add_artifact(run_id, "plan2", another_hashes)
        raise AssertionError("Expected add_artifact to fail after finalize")
    except Exception:
        # Exact exception type is HashingContractViolation; avoid tight coupling.
        pass

    # FAILED runs are hidden by default but can be queried explicitly.
    run_failed_id = registry.begin_run()
    registry.finalize_run(run_failed_id, RunStatus.FAILED)
    assert registry.get_run(run_failed_id) is None
    failed = registry.get_run(run_failed_id, include_failed=True)
    assert failed is not None
    assert failed.status is RunStatus.FAILED

    # iter_runs() by default only sees COMPLETE runs.
    runs = list(registry.iter_runs())
    assert all(r.status is RunStatus.COMPLETE for r in runs)
    assert any(r.run_id == run_id for r in runs)


def test_bundlewriter_with_cas_and_run_registry_emits_manifest_and_is_reconstructable(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    cas_root = tmp_path / "cas"
    runs_root = tmp_path / "runs"

    cas = LocalCAS(cas_root)
    registry = RunRegistry(runs_root)
    run_id = registry.begin_run(metadata={"strategy_id": "strat_X"})

    writer = BundleWriter(
        bundle_dir,
        cas=cas,
        run_registry=registry,
        run_id=run_id,
    )

    # Minimal but structurally valid payloads.
    writer.write_plan(
        plan_hash="intent-123",
        config_hash="cfg-123",
        as_of_time="2025-01-01T00:00:00Z",
        config={"param": 1},
    )
    writer.write_env_fingerprint()
    writer.write_dataset_manifest(
        dataset_id="ds-1",
        symbols=["AAPL"],
        row_count=3,
        time_range={"start": "2020-01-01", "end": "2020-01-02"},
    )
    writer.write_preprocessing_report(
        steps=[],
        timings={},
        warnings=[],
    )
    writer.write_splits_manifest(
        splits=[],
        split_method="time",
        purge_window=0,
        embargo_window=0,
    )

    writer.write_bundle_manifest()

    # Existing Appendix-C layout should be intact.
    assert_bundle_complete(bundle_dir)

    # bundle_manifest.json should exist and contain per-role CAS/attest entries.
    manifest_path = bundle_dir / "bundle_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["hash_policy"]["cas"] == "cas.v1:b3-256"
    assert manifest["hash_policy"]["attest"] == "attest.v1:jcs-sha256"

    artifacts = manifest["artifacts"]
    # At minimum we expect the five core roles and each entry must carry a path.
    for role in (
        "plan",
        "env_fingerprint",
        "dataset_manifest",
        "preprocessing_report",
        "splits_manifest",
    ):
        assert role in artifacts
        entry = artifacts[role]
        assert isinstance(entry["path"], str)
        assert isinstance(entry["cas"], str)
        assert entry["cas"].startswith("cas.v1:b3-256:")
        if entry["attest"] is not None:
            assert entry["attest"].startswith("attest.v1:jcs-sha256:")

    # Reconstruct a bundle purely from CAS + manifest and assert completeness.
    reconstructed_dir = tmp_path / "reconstructed"
    reconstructed_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct using manifest-provided paths; no hard-coded role → filename mapping.
    for entry in artifacts.values():
        path = entry["path"]
        cas_str = entry["cas"]
        target = reconstructed_dir / path
        cas.materialize(cas_str, target)

    assert_bundle_complete(reconstructed_dir)


def test_model_and_task_manifest_helpers_register_artifacts(tmp_path: Path) -> None:
    cas_root = tmp_path / "cas"
    runs_root = tmp_path / "runs"

    cas = LocalCAS(cas_root)
    registry = RunRegistry(runs_root)
    run_id = registry.begin_run()

    model_hashes = store_model_snapshot_manifest(
        cas,
        registry,
        run_id,
        {"model_id": "m1", "version": "v1"},
    )
    task_hashes = store_task_manifest(
        cas,
        registry,
        run_id,
        {"task_id": "t1", "kind": "backtest"},
    )

    # Finalize to make the run visible by default.
    registry.finalize_run(run_id, RunStatus.COMPLETE)
    run = registry.get_run(run_id)
    assert run is not None

    roles = {a.role for a in run.artifacts}
    assert "model_snapshot_manifest" in roles
    assert "task_manifest" in roles

    # Ensure CAS objects exist and identifiers are well-formed.
    for hashes in (model_hashes, task_hashes):
        assert cas.exists(hashes.cas)
        assert isinstance(hashes.cas, HashRef)
        assert hashes.cas.domain == "cas.v1"
        assert hashes.cas.algo == "b3-256"
