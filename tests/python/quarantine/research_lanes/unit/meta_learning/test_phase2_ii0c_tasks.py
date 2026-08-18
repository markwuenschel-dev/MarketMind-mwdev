"""II-0C MetaTask scaffold tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.seed_policy import build_run_identity
from pysrc.meta.task_manifest_emitter import build_task_manifest_document
from pysrc.meta_learning.dynamic_k_contract import build_fixed_slot_surface_from_sparse_slots
from pysrc.meta_learning.phase2_ii0c_tasks import (
    build_ii0c_meta_task,
    build_ii0c_signal_surface,
    build_ii0c_task_inputs,
    build_ii0c_task_manifest_input,
)
from pysrc.meta_learning.task_generator import meta_task_to_task_manifest_input

pytestmark = pytest.mark.unit


def _base_kwargs() -> dict[str, object]:
    purge = pd.Timedelta(0)
    embargo = pd.Timedelta(seconds=1)
    return {
        "regime_id": "ii0c__trend_bull",
        "regime_class": "bull",
        "regime_embedding": None,
        "support_set": ("2024-01-01T00:00:00+00:00",),
        "query_set": ("2024-01-02T00:00:00+00:00",),
        "horizon": 8,
        "signal_set_version": "ii0c.v1",
        "t0": "2024-01-01T00:00:00+00:00",
        "t1": "2024-01-02T00:00:00+00:00",
        "purge_window": purge,
        "embargo_window": embargo,
    }


@pytest.mark.determinism("d0")
def test_build_ii0c_signal_surface_is_deterministic_by_slot_order(deterministic_seed: int) -> None:
    _ = deterministic_seed
    a = build_ii0c_signal_surface({5: "sig_b", 1: "sig_a"})
    b = build_ii0c_signal_surface({1: "sig_a", 5: "sig_b"})
    canonical = build_fixed_slot_surface_from_sparse_slots({1: "sig_a", 5: "sig_b"})
    assert a == b == canonical
    assert a[0][1] == "sig_a"
    assert a[0][5] == "sig_b"
    assert a[1][1] is True
    assert a[1][5] is True


@pytest.mark.determinism("d0")
def test_build_ii0c_task_inputs_propagates_pit_boundary_and_serializes(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    kwargs = _base_kwargs()
    meta_task, task_row = build_ii0c_task_inputs(
        **kwargs,
        slot_to_signal_id={7: "sig_z", 0: "sig_a"},
        pit_boundary="2024-01-01T00:00:00+00:00",
    )
    canonical_row = meta_task_to_task_manifest_input(meta_task)
    assert task_row == build_ii0c_task_manifest_input(
        **kwargs,
        slot_to_signal_id={0: "sig_a", 7: "sig_z"},
        pit_boundary="2024-01-01T00:00:00+00:00",
    )
    assert task_row.pit_boundary == meta_task.pit_boundary
    assert canonical_row.support_last_timestamp == task_row.support_last_timestamp
    assert canonical_row.pit_boundary is None
    assert meta_task.signal_ids_hash == task_row.signal_ids_hash

    doc = build_task_manifest_document(tasks=[task_row], run_identity=build_run_identity(11))
    out = tmp_path / "task_manifest.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["tasks"][0]["pit_boundary"] == meta_task.pit_boundary
    assert loaded["tasks"][0]["task_id"] == meta_task.task_id


@pytest.mark.determinism("d0")
def test_build_ii0c_meta_task_fails_closed_on_boundary_violations(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    overlap_kwargs = dict(_base_kwargs())
    overlap_kwargs["support_set"] = ("2024-01-01T00:00:00+00:00",)
    overlap_kwargs["query_set"] = ("2024-01-01T00:00:00+00:00",)
    with pytest.raises(DataPreconditionError, match="disjoint"):
        build_ii0c_meta_task(**overlap_kwargs, slot_to_signal_id={0: "sig_a"})
    pit_kwargs = dict(_base_kwargs())
    with pytest.raises(DataPreconditionError, match="pit_boundary"):
        build_ii0c_meta_task(
            **pit_kwargs,
            slot_to_signal_id={0: "sig_a"},
            pit_boundary="2024-01-03T00:00:00+00:00",
        )
    with pytest.raises(DataPreconditionError, match="purge_window"):
        gap_kwargs = dict(_base_kwargs())
        gap_kwargs["query_set"] = ("2024-01-01T00:00:01+00:00",)
        gap_kwargs["purge_window"] = pd.Timedelta(seconds=5)
        gap_kwargs["embargo_window"] = pd.Timedelta(seconds=5)
        build_ii0c_task_manifest_input(
            **gap_kwargs,
            slot_to_signal_id={0: "sig_a"},
        )


@pytest.mark.determinism("d0")
def test_build_ii0c_task_inputs_preserves_canonical_task_lowering(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    kwargs = _base_kwargs()
    meta_task, task_row = build_ii0c_task_inputs(
        **kwargs,
        slot_to_signal_id={3: "sig_c"},
        pit_boundary="2024-01-01T00:00:00+00:00",
    )
    assert task_row.regime_id == meta_task.regime_id
    assert task_row.regime_class == meta_task.regime_class
    assert task_row.t0 == meta_task.t0
    assert task_row.t1 == meta_task.t1
    assert task_row.signal_ids_hash == meta_task.signal_ids_hash
    assert task_row.signal_set_version == meta_task.signal_set_version
    assert task_row.support_last_timestamp == meta_task.pit_boundary
    assert task_row.pit_boundary == meta_task.pit_boundary
