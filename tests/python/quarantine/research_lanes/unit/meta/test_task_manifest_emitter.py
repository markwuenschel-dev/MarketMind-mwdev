"""Tests for scaffold and governed task-manifest emission."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from pysrc.meta.scaffold_emitter_guard import ScaffoldEmitterForbiddenError
from pysrc.meta.seed_policy import build_run_identity
from pysrc.meta.task import MetaTask
from pysrc.meta.task_manifest_config import TaskManifestConfig
from pysrc.meta.task_manifest_emitter import (
    GOVERNED_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TaskManifestTaskInput,
    emit_governed_task_manifest,
    emit_task_manifest,
    recompute_task_manifest_content_hash_from_document,
)
from pysrc.meta.task_manifest_errors import TaskManifestFieldError, TaskManifestIdentityError
from pysrc.meta_learning.dynamic_k_contract import build_fixed_slot_surface_from_sparse_slots
from pysrc.meta_learning.task_generator import derive_signal_ids_hash

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _meta_task(
    *, task_id: str, regime_class: str = "bull", signal_ids_hash: str | None = None
) -> MetaTask:
    signal_ids, signal_mask = build_fixed_slot_surface_from_sparse_slots({0: f"{task_id}.signal"})
    return MetaTask(
        task_id=task_id,
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class=regime_class,
        t0="2024-01-01T00:00:00+00:00",
        t1="2024-01-20T00:00:00+00:00",
        pit_boundary="2024-01-05T00:00:00+00:00",
        support_set=(
            "2024-01-01T00:00:00+00:00",
            "2024-01-02T00:00:00+00:00",
            "2024-01-03T00:00:00+00:00",
            "2024-01-04T00:00:00+00:00",
            "2024-01-05T00:00:00+00:00",
        ),
        query_set=(
            "2024-01-10T00:00:00+00:00",
            "2024-01-11T00:00:00+00:00",
        ),
        signal_ids=signal_ids,
        signal_mask=signal_mask,
        signal_set_version="rg09.v1",
        signal_ids_hash=signal_ids_hash
        or derive_signal_ids_hash(signal_ids=signal_ids, signal_mask=signal_mask),
        horizon=1,
        active_k=1,
    )


def _config() -> TaskManifestConfig:
    return TaskManifestConfig(run_id="run.sha256:" + "1" * 64, signal_set_version="rg09.v1")


@pytest.mark.determinism("d1")
def test_scaffold_emit_task_manifest_forbidden_when_governed_lane_set(
    tmp_path: Path, deterministic_seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = deterministic_seed
    monkeypatch.setenv("MARKETMIND_GOVERNED_LANE", "1")
    task = TaskManifestTaskInput(
        regime_id="trend_bull__stable",
        regime_class="bull",
        t0="2020-01-01T00:00:00+00:00",
        t1="2020-03-01T00:00:00+00:00",
        signal_ids_hash="sha256:signals",
        signal_set_version="rg09.v1",
        support_last_timestamp="2020-02-15T00:00:00+00:00",
    )
    with pytest.raises(ScaffoldEmitterForbiddenError):
        emit_task_manifest(tmp_path / "blocked.json", tasks=[task], seed=1)


@pytest.mark.determinism("d1")
def test_scaffold_schema_version_present(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    task = TaskManifestTaskInput(
        regime_id="trend_bull__stable",
        regime_class="bull",
        t0="2020-01-01T00:00:00+00:00",
        t1="2020-03-01T00:00:00+00:00",
        signal_ids_hash="sha256:signals",
        signal_set_version="rg09.v1",
        support_last_timestamp="2020-02-15T00:00:00+00:00",
    )
    doc = emit_task_manifest(tmp_path / "task_manifest.json", tasks=[task], seed=99)
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["run_identity"] == build_run_identity(99).to_block()


@pytest.mark.determinism("d1")
def test_schema_version_present(deterministic_seed: int) -> None:
    _ = deterministic_seed
    report = emit_governed_task_manifest([_meta_task(task_id="task-1")], _config())
    assert report.document["schema_version"] == GOVERNED_SCHEMA_VERSION


@pytest.mark.determinism("d1")
def test_all_required_fields_per_row(deterministic_seed: int) -> None:
    _ = deterministic_seed
    report = emit_governed_task_manifest([_meta_task(task_id="task-1")], _config())
    row = report.document["tasks"][0]
    for key in (
        "task_id",
        "regime_id",
        "regime_class",
        "t0",
        "t1",
        "pit_boundary",
        "signal_ids_hash",
        "signal_set_version",
    ):
        assert key in row


@pytest.mark.determinism("d1")
def test_missing_task_id_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    task = SimpleNamespace(
        task_id=None,
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class="bull",
        t0="2024-01-01T00:00:00+00:00",
        t1="2024-01-20T00:00:00+00:00",
        pit_boundary="2024-01-05T00:00:00+00:00",
        signal_ids_hash="sha256:" + "2" * 64,
    )
    with pytest.raises(TaskManifestFieldError, match="task_id"):
        emit_governed_task_manifest([task], _config())


@pytest.mark.determinism("d1")
def test_duplicate_task_id_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    task = _meta_task(task_id="dup-task")
    with pytest.raises(TaskManifestIdentityError, match="task_id"):
        emit_governed_task_manifest([task, task], _config())


@pytest.mark.determinism("d1")
def test_content_hash_determinism(deterministic_seed: int) -> None:
    _ = deterministic_seed
    tasks = [_meta_task(task_id="task-1"), _meta_task(task_id="task-2", regime_class="bear")]
    a = emit_governed_task_manifest(tasks, _config())
    b = emit_governed_task_manifest(tasks, _config())
    assert a.content_hash == b.content_hash


@pytest.mark.determinism("d1")
def test_content_hash_excludes_wall_clock(deterministic_seed: int) -> None:
    _ = deterministic_seed
    report = emit_governed_task_manifest([_meta_task(task_id="task-1")], _config())
    mutated = deepcopy(report.document)
    mutated["emitted_at"] = "2026-04-23T11:00:00Z"
    assert recompute_task_manifest_content_hash_from_document(mutated) == report.content_hash


@pytest.mark.determinism("d1")
def test_content_hash_format(deterministic_seed: int) -> None:
    _ = deterministic_seed
    report = emit_governed_task_manifest([_meta_task(task_id="task-1")], _config())
    assert report.content_hash.startswith("sha256:")
    assert len(report.content_hash) == 71


@pytest.mark.determinism("d1")
def test_task_count_matches(deterministic_seed: int) -> None:
    _ = deterministic_seed
    report = emit_governed_task_manifest(
        [_meta_task(task_id="task-1"), _meta_task(task_id="task-2", regime_class="bear")],
        _config(),
    )
    assert report.document["task_count"] == 2


@pytest.mark.determinism("d1")
def test_empty_task_list_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(TaskManifestFieldError, match="tasks"):
        emit_governed_task_manifest([], _config())


@pytest.mark.determinism("d1")
def test_signal_ids_hash_reads_from_task(deterministic_seed: int) -> None:
    _ = deterministic_seed
    task = _meta_task(task_id="task-1", signal_ids_hash="sha256:" + "3" * 64)
    report = emit_governed_task_manifest([task], _config())
    assert report.document["tasks"][0]["signal_ids_hash"] == "sha256:" + "3" * 64


@pytest.mark.determinism("d1")
def test_signal_ids_hash_none_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    task = SimpleNamespace(
        task_id="task-1",
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class="bull",
        t0="2024-01-01T00:00:00+00:00",
        t1="2024-01-20T00:00:00+00:00",
        pit_boundary="2024-01-05T00:00:00+00:00",
        signal_ids_hash=None,
    )
    with pytest.raises(TaskManifestFieldError, match="signal_ids_hash"):
        emit_governed_task_manifest([task], _config())


@pytest.mark.determinism("d1")
def test_committed_example_artifact_schema_and_hash() -> None:
    path = Path("artifacts/phase_ii/task_manifest/task_manifest_example.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["_notice"] == "synthetic-example-non-promotable"
    assert doc["schema_version"] == "task_manifest.v1"
    assert doc["task_count"] == 3
    assert len(doc["tasks"]) == 3
    assert doc["content_hash"]["algorithm"] == "sha256"
    assert doc["content_hash"]["canonicalization"] == "json.sort_keys.no_ws.omit_content_hash.v1"
    assert recompute_task_manifest_content_hash_from_document(doc) == doc["content_hash"]["value"]
