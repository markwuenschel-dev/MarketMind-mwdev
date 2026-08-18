"""MLN-01 MetaTask generator and TaskRegistry invariants."""

from __future__ import annotations

import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.rg09_phase2_bridge import meta_task_for_rg09_harness_bundle
from pysrc.meta.task_manifest_emitter import compute_task_id as scaffold_compute_task_id
from pysrc.meta_learning.contracts.task_registry import (
    TaskRegistryDuplicateError,
    TaskRegistryProtocol,
)
from pysrc.meta_learning.dynamic_k_contract import build_fixed_slot_surface_from_sparse_slots
from pysrc.meta_learning.in_memory_task_registry import AppendOnlyTaskRegistry
from pysrc.meta_learning.task_generator import (
    TASK_ID_HMAC_KEY_MATERIAL,
    TASK_ID_HMAC_KEY_VERSION,
    build_meta_task,
    compute_task_id,
    derive_signal_ids_hash,
    meta_task_from_record,
    meta_task_to_record,
    meta_task_to_task_manifest_input,
)


def _base_kwargs() -> dict[str, object]:
    purge = __import__("pandas").Timedelta(0)
    embargo = __import__("pandas").Timedelta(seconds=1)
    sig_ids, sig_mask = build_fixed_slot_surface_from_sparse_slots({0: "sig_a"})
    return {
        "regime_id": "r1",
        "regime_class": "bull",
        "regime_embedding": None,
        "support_set": ("2024-01-01T00:00:00+00:00",),
        "query_set": ("2024-01-02T00:00:00+00:00",),
        "horizon": 8,
        "signal_ids": sig_ids,
        "signal_mask": sig_mask,
        "signal_set_version": "v1",
        "t0": "2024-01-01T00:00:00+00:00",
        "t1": "2024-01-02T00:00:00+00:00",
        "purge_window": purge,
        "embargo_window": embargo,
    }


@pytest.mark.determinism("d0")
def test_task_id_deterministic_and_includes_signal_hash(deterministic_seed: int) -> None:
    _ = deterministic_seed
    kw = _base_kwargs()
    a = build_meta_task(**kw)
    b = build_meta_task(**kw)
    assert a.task_id == b.task_id
    h = derive_signal_ids_hash(signal_ids=a.signal_ids, signal_mask=a.signal_mask)
    alt = compute_task_id(
        regime_id=a.regime_id,
        t0=a.t0,
        t1=a.t1,
        signal_ids_hash="sha256:different",
    )
    assert alt != a.task_id
    assert a.signal_ids_hash == h


@pytest.mark.determinism("d0")
def test_scaffold_compute_task_id_delegates_to_canonical(deterministic_seed: int) -> None:
    _ = deterministic_seed
    x = compute_task_id(
        regime_id="r",
        t0="t0",
        t1="t1",
        signal_ids_hash="sha256:x",
    )
    y = scaffold_compute_task_id(regime_id="r", t0="t0", t1="t1", signal_ids_hash="sha256:x")
    assert x == y


@pytest.mark.determinism("d0")
def test_compute_task_id_rejects_empty_signal_hash(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(DataPreconditionError):
        compute_task_id(regime_id="r", t0="t0", t1="t1", signal_ids_hash="  ")


@pytest.mark.determinism("d0")
def test_fixed_slot_slot_position_affects_hash_not_alphabetical_sort(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    purge = __import__("pandas").Timedelta(0)
    embargo = __import__("pandas").Timedelta(seconds=1)
    base = {
        "regime_id": "r1",
        "regime_class": "bull",
        "regime_embedding": None,
        "support_set": ("2024-01-01T00:00:00+00:00",),
        "query_set": ("2024-01-02T00:00:00+00:00",),
        "horizon": 8,
        "signal_set_version": "v1",
        "t0": "2024-01-01T00:00:00+00:00",
        "t1": "2024-01-02T00:00:00+00:00",
        "purge_window": purge,
        "embargo_window": embargo,
    }
    z_slot0, m0 = build_fixed_slot_surface_from_sparse_slots({0: "z"})
    a_slot1, m1 = build_fixed_slot_surface_from_sparse_slots({1: "a"})
    ta = build_meta_task(**base, signal_ids=z_slot0, signal_mask=m0)
    tb = build_meta_task(**base, signal_ids=a_slot1, signal_mask=m1)
    assert ta.signal_ids_hash != tb.signal_ids_hash
    assert ta.active_k == 1
    assert tb.active_k == 1


@pytest.mark.determinism("d0")
def test_build_meta_task_rejects_invalid_regime_class(deterministic_seed: int) -> None:
    _ = deterministic_seed
    kw = _base_kwargs()
    kw["regime_class"] = "Bull"
    with pytest.raises(DataPreconditionError, match="regime_class"):
        build_meta_task(**kw)


@pytest.mark.determinism("d0")
def test_support_query_overlap_rejected(deterministic_seed: int) -> None:
    _ = deterministic_seed
    kw = _base_kwargs()
    ts = "2024-01-01T12:00:00+00:00"
    kw["support_set"] = (ts,)
    kw["query_set"] = (ts,)
    with pytest.raises(DataPreconditionError, match="disjoint"):
        build_meta_task(**kw)


@pytest.mark.determinism("d0")
def test_purge_embargo_violation_rejected(deterministic_seed: int) -> None:
    _ = deterministic_seed
    import pandas as pd

    kw = _base_kwargs()
    kw["support_set"] = ("2024-01-01T00:00:00+00:00",)
    kw["query_set"] = ("2024-01-01T00:00:01+00:00",)
    kw["purge_window"] = pd.Timedelta(seconds=5)
    kw["embargo_window"] = pd.Timedelta(seconds=5)
    with pytest.raises(DataPreconditionError, match="purge_window"):
        build_meta_task(**kw)


@pytest.mark.determinism("d0")
def test_pit_boundary_mismatch_rejected(deterministic_seed: int) -> None:
    _ = deterministic_seed
    kw = _base_kwargs()
    kw["pit_boundary"] = "2099-01-01T00:00:00+00:00"
    with pytest.raises(DataPreconditionError, match="pit_boundary"):
        build_meta_task(**kw)


@pytest.mark.determinism("d0")
def test_registry_duplicate_rejected(deterministic_seed: int) -> None:
    _ = deterministic_seed
    t = build_meta_task(**_base_kwargs())
    reg = AppendOnlyTaskRegistry()
    reg.append(t)
    with pytest.raises(TaskRegistryDuplicateError):
        reg.append(t)
    alt = _base_kwargs()
    o_ids, o_mask = build_fixed_slot_surface_from_sparse_slots({0: "other"})
    alt["signal_ids"] = o_ids
    alt["signal_mask"] = o_mask
    t2 = build_meta_task(**alt)
    with pytest.raises(TaskRegistryDuplicateError):
        reg.append(t2)


@pytest.mark.determinism("d0")
def test_registry_query_since_iso_structurally_matches_protocol(deterministic_seed: int) -> None:
    _ = deterministic_seed
    reg: TaskRegistryProtocol = AppendOnlyTaskRegistry()
    t = build_meta_task(**_base_kwargs())
    reg.append(t)
    assert reg.query(since="2099-01-01T00:00:00+00:00") == []
    assert len(reg.query(since="2024-01-01T00:00:00+00:00")) == 1


@pytest.mark.determinism("d0")
def test_meta_task_record_roundtrip(deterministic_seed: int) -> None:
    _ = deterministic_seed
    t = build_meta_task(**_base_kwargs())
    rec = meta_task_to_record(t)
    assert rec["task_id_hmac_key_version"] == TASK_ID_HMAC_KEY_VERSION
    assert TASK_ID_HMAC_KEY_MATERIAL == b""
    assert rec["signal_surface"]["kind"] == "fixed_slot_masked"
    assert rec["signal_surface"]["max_signals"] == 64
    t2 = meta_task_from_record(rec)
    assert t2 == t


@pytest.mark.determinism("d0")
def test_lower_to_task_manifest_preserves_identity_fields(deterministic_seed: int) -> None:
    _ = deterministic_seed
    t = build_meta_task(**_base_kwargs())
    row = meta_task_to_task_manifest_input(t)
    assert row.regime_id == t.regime_id
    assert row.signal_ids_hash == t.signal_ids_hash
    assert row.support_last_timestamp == t.pit_boundary


@pytest.mark.determinism("d1")
def test_rg09_bridge_yields_meta_task_and_manifest_row(deterministic_seed: int) -> None:
    _ = deterministic_seed
    import pandas as pd

    episodes = pd.DataFrame(
        {
            "regime_id": ["rid"],
            "regime_class": ["bull"],
            "start_ts": [pd.Timestamp("2024-06-01T00:00:00+00:00")],
            "end_ts": [pd.Timestamp("2024-06-05T00:00:00+00:00")],
        }
    )
    summary = {"date_range_start": "2024-06-01", "date_range_end": "2024-06-05"}
    mt = meta_task_for_rg09_harness_bundle(
        episodes=episodes,
        summary=summary,
        fixture_sha256="sha256:abc",
        null_seed_namespace="rg09.v1",
        label_horizon_bars=16,
    )
    row = meta_task_to_task_manifest_input(mt)
    assert row.regime_id == mt.regime_id
    assert row.signal_ids_hash == mt.signal_ids_hash
