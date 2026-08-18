"""Integration coverage for the governed task-manifest lane."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysrc.meta.seed_policy import build_run_identity
from pysrc.meta.task import MetaTask
from pysrc.meta.task_manifest_config import TaskManifestConfig
from pysrc.meta.task_manifest_emitter import emit_governed_task_manifest
from pysrc.meta.task_manifest_errors import TaskManifestImmutabilityError
from pysrc.meta.task_manifest_io import load_task_manifest, write_task_manifest
from pysrc.meta_learning.dynamic_k_contract import build_fixed_slot_surface_from_sparse_slots
from pysrc.meta_learning.task_generator import derive_signal_ids_hash


def _meta_task(task_id: str, regime_class: str) -> MetaTask:
    signal_ids, signal_mask = build_fixed_slot_surface_from_sparse_slots({0: f"{task_id}.signal"})
    return MetaTask(
        task_id=task_id,
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class=regime_class,
        t0="2024-02-01T00:00:00+00:00",
        t1="2024-02-20T00:00:00+00:00",
        pit_boundary="2024-02-05T00:00:00+00:00",
        support_set=(
            "2024-02-01T00:00:00+00:00",
            "2024-02-02T00:00:00+00:00",
            "2024-02-03T00:00:00+00:00",
            "2024-02-04T00:00:00+00:00",
            "2024-02-05T00:00:00+00:00",
        ),
        query_set=(
            "2024-02-10T00:00:00+00:00",
            "2024-02-11T00:00:00+00:00",
        ),
        signal_ids=signal_ids,
        signal_mask=signal_mask,
        signal_set_version="rg09.v1",
        signal_ids_hash=derive_signal_ids_hash(signal_ids=signal_ids, signal_mask=signal_mask),
        horizon=1,
        active_k=1,
    )


@pytest.mark.integration
@pytest.mark.determinism("d2")
def test_governed_task_manifest_round_trip_and_immutability(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    cfg = TaskManifestConfig(run_id=build_run_identity(123).run_id, signal_set_version="rg09.v1")
    tasks = [
        _meta_task("task-1", "bull"),
        _meta_task("task-2", "bear"),
        _meta_task("task-3", "high_vol"),
    ]
    report = emit_governed_task_manifest(tasks, cfg)
    out_path = write_task_manifest(report, tmp_path)
    loaded = load_task_manifest(out_path)
    assert loaded == report.document
    assert loaded["task_count"] == len(tasks)
    with pytest.raises(TaskManifestImmutabilityError, match="governed artifact already exists"):
        write_task_manifest(report, tmp_path)
