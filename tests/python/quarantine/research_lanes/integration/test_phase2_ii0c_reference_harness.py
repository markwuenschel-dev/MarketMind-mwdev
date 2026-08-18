"""End-to-end II-0C reference harness replay (non-promotable wiring stability only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysrc.meta_learning.phase2_ii0c_encoder import II0CEncoderStub
from pysrc.meta_learning.phase2_ii0c_reference import (
    II0C_REFERENCE_SEED,
    II0C_REFERENCE_TIMESTAMP_UTC,
    build_ii0c_reference_encoder_config,
    build_ii0c_reference_meta_task_request,
)
from pysrc.meta_learning.phase2_ii0c_tasks import II0CMetaTaskPayload, build_ii0c_meta_task_payload
from pysrc.pipeline.phase2_ii0c_runner import (
    II0C_DRY_RUN_SUMMARY_FILENAME,
    II0C_DRY_RUN_SUMMARY_LEGACY_FILENAME,
    II0C_PILOT_REPORT_FILENAME,
    Phase2II0CProviders,
    Phase2II0CRunRequest,
    run_phase2_ii0c_dry_run,
    run_phase2_ii0c_pilot,
)

pytestmark = [pytest.mark.integration, pytest.mark.determinism("d0")]


def _reference_task_payload() -> II0CMetaTaskPayload:
    return build_ii0c_meta_task_payload(build_ii0c_reference_meta_task_request())


def test_phase2_ii0c_reference_dry_run_is_stable_across_output_roots(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"

    def _task_provider(*, seed: int, timestamp_utc: str) -> II0CMetaTaskPayload:
        _ = seed, timestamp_utc
        return _reference_task_payload()

    first = run_phase2_ii0c_dry_run(
        output_dir=root_a,
        seed=II0C_REFERENCE_SEED,
        timestamp_utc=II0C_REFERENCE_TIMESTAMP_UTC,
        task_provider=_task_provider,
        encoder_provider=lambda *, task_payload, seed: II0CEncoderStub(
            build_ii0c_reference_encoder_config()
        ).encode_task(task_payload.meta_task),
    )
    second = run_phase2_ii0c_dry_run(
        output_dir=root_b,
        seed=II0C_REFERENCE_SEED,
        timestamp_utc=II0C_REFERENCE_TIMESTAMP_UTC,
        task_provider=_task_provider,
        encoder_provider=lambda *, task_payload, seed: II0CEncoderStub(
            build_ii0c_reference_encoder_config()
        ).encode_task(task_payload.meta_task),
    )

    tm_a = json.loads(
        (first.governed_evidence_dir / "task_manifest.json").read_text(encoding="utf-8")
    )
    tm_b = json.loads(
        (second.governed_evidence_dir / "task_manifest.json").read_text(encoding="utf-8")
    )
    assert tm_a["tasks"][0]["task_id"] == tm_b["tasks"][0]["task_id"]
    assert tm_a["content_hash"] == tm_b["content_hash"]

    assert (
        first.summary["cross_section"]["task"]["task_id"]
        == second.summary["cross_section"]["task"]["task_id"]
    )
    disk_a = json.loads((root_a / II0C_DRY_RUN_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    disk_b = json.loads((root_b / II0C_DRY_RUN_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert (
        disk_a["identity_binding_audit"]["task_id"] == disk_b["identity_binding_audit"]["task_id"]
    )
    for root in (root_a, root_b):
        legacy = json.loads(
            (root / II0C_DRY_RUN_SUMMARY_LEGACY_FILENAME).read_text(encoding="utf-8")
        )
        assert "cross_section" not in legacy
        assert legacy["task"]["task_id"] == tm_a["tasks"][0]["task_id"]


def test_phase2_ii0c_reference_pilot_emits_report_and_governed_triple(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    bundle = tmp_path / "bundle"

    def _task(request: Phase2II0CRunRequest) -> II0CMetaTaskPayload:
        _ = request
        return _reference_task_payload()

    result = run_phase2_ii0c_pilot(
        Phase2II0CRunRequest(
            bundle_dir=bundle,
            seed=II0C_REFERENCE_SEED,
            strategy_id="phase2_ii0c_reference",
            run_metadata={"timestamp_utc": II0C_REFERENCE_TIMESTAMP_UTC},
        ),
        providers=Phase2II0CProviders(task_provider=_task),
    )

    assert (bundle / II0C_PILOT_REPORT_FILENAME).is_file()
    assert (result.governed_evidence_dir / "task_manifest.json").is_file()
    assert (
        result.pilot_report["incumbent_baseline_identity"]["baseline_kind"] == "xgboost_incumbent"
    )
    assert (
        result.task_payload.meta_task.task_id
        == result.pilot_report["identity_binding_audit"]["task_id"]
    )
