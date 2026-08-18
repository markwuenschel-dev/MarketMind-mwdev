"""MLC-0 · Unit tests for the promotable :class:`pysrc.meta.task_registry.TaskRegistry`.

Covers brief §5 Step 7 acceptance:
- Append behavior on new ``(regime_id, t0)`` keys
- Duplicate ``(regime_id, t0)`` raises :class:`TaskRegistryDuplicateError`
- Task is retrievable after registration
- No mutation of registered tasks
- Durable store stub interface exercised (in-memory ``NullDurableStore`` +
  file-backed ``JsonLinesDurableStore``)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta.task_registry import (
    JsonLinesDurableStore,
    NullDurableStore,
    TaskNotFoundError,
    TaskRegistry,
    TaskRegistryDuplicateError,
)


def _ids() -> tuple[str, ...]:
    return tuple(f"sig_{i:02d}" for i in range(MAX_SIGNALS))


def _mask_of(k: int) -> tuple[bool, ...]:
    return tuple(i < k for i in range(MAX_SIGNALS))


def _make_task(
    *,
    regime_id: str = "trend_hi__vol_med__bocpd_stable",
    t0: str = "2024-01-01T00:00:00+00:00",
    task_id: str | None = None,
) -> MetaTask:
    suffix = task_id or f"hmac-sha256:{regime_id}-{t0}"
    return MetaTask(
        task_id=suffix,
        regime_id=regime_id,
        regime_class="bull",
        t0=t0,
        t1="2024-01-10T00:00:00+00:00",
        pit_boundary="2024-01-03T00:00:00+00:00",
        support_set=(
            "2024-01-01T00:00:00+00:00",
            "2024-01-02T00:00:00+00:00",
            "2024-01-03T00:00:00+00:00",
        ),
        query_set=(
            "2024-01-05T00:00:00+00:00",
            "2024-01-06T00:00:00+00:00",
        ),
        signal_ids=_ids(),
        signal_mask=_mask_of(3),
        signal_set_version="mln04.v1",
        signal_ids_hash="sha256:deadbeef",
        horizon=1,
        active_k=3,
    )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_append_new_task_succeeds() -> None:
    reg = TaskRegistry()
    task = _make_task()
    reg.append(task)
    assert len(reg) == 1
    assert reg.get(task.task_id) is task


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_duplicate_stable_key_raises() -> None:
    reg = TaskRegistry()
    t1 = _make_task(task_id="hmac-sha256:one")
    t2 = _make_task(task_id="hmac-sha256:two")  # same (regime_id, t0)
    reg.append(t1)
    with pytest.raises(TaskRegistryDuplicateError):
        reg.append(t2)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_duplicate_task_id_raises_even_if_stable_key_differs() -> None:
    reg = TaskRegistry()
    t1 = _make_task(task_id="hmac-sha256:one", regime_id="trend_hi__vol_med__bocpd_stable")
    t2 = _make_task(task_id="hmac-sha256:one", t0="2023-12-31T00:00:00+00:00")
    reg.append(t1)
    with pytest.raises(TaskRegistryDuplicateError):
        reg.append(t2)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_get_missing_raises() -> None:
    reg = TaskRegistry()
    with pytest.raises(TaskNotFoundError):
        reg.get("no-such-id")


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_get_by_stable_roundtrip() -> None:
    reg = TaskRegistry()
    task = _make_task()
    reg.append(task)
    found = reg.get_by_stable(regime_id=task.regime_id, t0=task.t0)
    assert found is task


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_get_by_stable_missing_raises() -> None:
    reg = TaskRegistry()
    with pytest.raises(TaskNotFoundError):
        reg.get_by_stable(regime_id="trend_lo__vol_hi__bocpd_cp", t0="2024-02-01T00:00:00+00:00")


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_contains_stable_reports_presence() -> None:
    reg = TaskRegistry()
    task = _make_task()
    assert reg.contains_stable(regime_id=task.regime_id, t0=task.t0) is False
    reg.append(task)
    assert reg.contains_stable(regime_id=task.regime_id, t0=task.t0) is True


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_query_filters_by_regime_id() -> None:
    reg = TaskRegistry()
    t1 = _make_task(
        regime_id="trend_hi__vol_med__bocpd_stable",
        task_id="id-1",
    )
    t2 = _make_task(
        regime_id="trend_lo__vol_hi__bocpd_cp",
        task_id="id-2",
    )
    reg.append(t1)
    reg.append(t2)
    hits = reg.query(regime_id="trend_lo__vol_hi__bocpd_cp")
    assert [t.task_id for t in hits] == ["id-2"]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_query_filters_by_since() -> None:
    reg = TaskRegistry()
    t1 = _make_task(regime_id="trend_hi__vol_med__bocpd_stable", task_id="id-1")
    reg.append(t1)
    assert [t.task_id for t in reg.query(since="2023-01-01T00:00:00+00:00")] == ["id-1"]
    assert reg.query(since="2099-01-01T00:00:00+00:00") == []


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_iteration_preserves_insertion_order() -> None:
    reg = TaskRegistry()
    task_ids = []
    for i, regime in enumerate(
        (
            "trend_hi__vol_med__bocpd_stable",
            "trend_lo__vol_hi__bocpd_cp",
            "trend_flat__vol_lo__bocpd_transition",
        )
    ):
        t = _make_task(regime_id=regime, task_id=f"id-{i}")
        reg.append(t)
        task_ids.append(t.task_id)
    assert [t.task_id for t in reg] == task_ids


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_iter_stable_keys_returns_all() -> None:
    reg = TaskRegistry()
    reg.append(_make_task())
    keys = list(reg.iter_stable_keys())
    assert keys == [("trend_hi__vol_med__bocpd_stable", "2024-01-01T00:00:00+00:00")]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_append_rejects_non_metatask() -> None:
    reg = TaskRegistry()
    with pytest.raises(DataPreconditionError):
        reg.append("not a task")  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_registered_task_is_frozen() -> None:
    reg = TaskRegistry()
    task = _make_task()
    reg.append(task)
    found = reg.get(task.task_id)
    with pytest.raises(Exception):
        found.regime_class = "bear"  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_null_durable_store_default() -> None:
    reg = TaskRegistry()
    assert isinstance(reg.durable_store, NullDurableStore)
    reg.append(_make_task())
    assert list(reg.durable_store.iter_records()) == []


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_jsonlines_durable_store_persists(tmp_path: Path) -> None:
    store = JsonLinesDurableStore(tmp_path / "task_registry.jsonl")
    reg = TaskRegistry(durable_store=store)
    task = _make_task()
    reg.append(task)
    records: list[dict[str, Any]] = list(store.iter_records())
    assert len(records) == 1
    assert records[0]["task_id"] == task.task_id
    assert records[0]["record"]["regime_id"] == task.regime_id
    # file actually written on disk
    raw = (tmp_path / "task_registry.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 1
    parsed = json.loads(raw[0])
    assert parsed["task_id"] == task.task_id


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_jsonlines_store_rejects_empty_task_id(tmp_path: Path) -> None:
    store = JsonLinesDurableStore(tmp_path / "r.jsonl")
    with pytest.raises(DataPreconditionError):
        store.persist("", {"x": 1})


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_jsonlines_store_rejects_non_dict_record(tmp_path: Path) -> None:
    store = JsonLinesDurableStore(tmp_path / "r.jsonl")
    with pytest.raises(DataPreconditionError):
        store.persist("task-1", "not a dict")  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_jsonlines_store_iter_on_missing_file(tmp_path: Path) -> None:
    # construct without auto-touch by supplying a subdir path
    path = tmp_path / "fresh" / "absent.jsonl"
    store = JsonLinesDurableStore(path)
    # file was created as empty by ctor
    assert list(store.iter_records()) == []


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_duplicate_does_not_persist_second_record(tmp_path: Path) -> None:
    store = JsonLinesDurableStore(tmp_path / "r.jsonl")
    reg = TaskRegistry(durable_store=store)
    t1 = _make_task(task_id="id-1")
    reg.append(t1)
    with pytest.raises(TaskRegistryDuplicateError):
        reg.append(_make_task(task_id="id-2"))  # same stable key
    # only the first append wrote a line
    raw = (tmp_path / "r.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 1
