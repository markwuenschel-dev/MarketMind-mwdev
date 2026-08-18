"""MLC-0 · Unit tests for the promotable :class:`pysrc.meta.task.MetaTask`.

Covers brief §5 Step 7 acceptance:
- Schema field presence and type correctness
- Immutability enforcement
- ``pit_boundary`` equals last support timestamp
- Support / query disjointness
- ``signal_ids`` / ``signal_mask`` length consistency
- ``regime_embedding`` stays ``None`` (MLC-1 not yet online) or is an ``np.ndarray``
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any

import numpy as np
import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.task import MAX_SIGNALS, META_TASK_SCHEMA_VERSION, MetaTask


def _mask_of(active_k: int) -> tuple[bool, ...]:
    return tuple(i < active_k for i in range(MAX_SIGNALS))


def _ids() -> tuple[str, ...]:
    return tuple(f"sig_{i:02d}" for i in range(MAX_SIGNALS))


def _valid_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": "hmac-sha256:abc123",
        "regime_id": "trend_hi__vol_med__bocpd_stable",
        "regime_class": "bull",
        "t0": "2024-01-01T00:00:00+00:00",
        "t1": "2024-01-10T00:00:00+00:00",
        "pit_boundary": "2024-01-03T00:00:00+00:00",
        "support_set": (
            "2024-01-01T00:00:00+00:00",
            "2024-01-02T00:00:00+00:00",
            "2024-01-03T00:00:00+00:00",
        ),
        "query_set": (
            "2024-01-05T00:00:00+00:00",
            "2024-01-06T00:00:00+00:00",
        ),
        "signal_ids": _ids(),
        "signal_mask": _mask_of(3),
        "signal_set_version": "mln04.v1",
        "signal_ids_hash": "sha256:deadbeef",
        "horizon": 1,
        "active_k": 3,
    }
    base.update(overrides)
    return base


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_schema_version_constant() -> None:
    assert META_TASK_SCHEMA_VERSION == "mlc0.v2.0"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_valid_metatask_constructs() -> None:
    t = MetaTask(**_valid_kwargs())
    assert t.regime_class == "bull"
    assert t.active_k == 3
    assert t.pit_boundary == t.support_set[-1]
    assert t.regime_embedding is None
    assert t.has_regime_embedding() is False


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_all_required_fields_present() -> None:
    """MLC-0 AC-1: every field from MetaLearningCore §2.1 YAML is present."""
    required = {
        "task_id",
        "regime_id",
        "regime_class",
        "regime_embedding",
        "t0",
        "t1",
        "pit_boundary",
        "support_set",
        "query_set",
        "signal_ids",
        "signal_mask",
        "signal_set_version",
        "signal_ids_hash",
        "horizon",
        "active_k",
    }
    names = {f.name for f in fields(MetaTask)}
    assert required <= names


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_immutable_after_construction() -> None:
    t = MetaTask(**_valid_kwargs())
    with pytest.raises((FrozenInstanceError, AttributeError)):
        t.regime_class = "bear"  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_pit_boundary_must_equal_last_support() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(pit_boundary="2024-01-02T00:00:00+00:00"))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_support_query_overlap_raises() -> None:
    bad_query = (
        "2024-01-03T00:00:00+00:00",  # overlaps support max
        "2024-01-05T00:00:00+00:00",
    )
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(query_set=bad_query))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_query_must_lie_strictly_after_pit_boundary() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(
            **_valid_kwargs(
                query_set=("2024-01-03T00:00:00+00:00", "2024-01-05T00:00:00+00:00"),
            )
        )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_support_set_must_be_strictly_sorted() -> None:
    bad_support = (
        "2024-01-02T00:00:00+00:00",
        "2024-01-01T00:00:00+00:00",
        "2024-01-03T00:00:00+00:00",
    )
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(support_set=bad_support))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_query_set_must_be_strictly_sorted() -> None:
    bad_query = (
        "2024-01-06T00:00:00+00:00",
        "2024-01-05T00:00:00+00:00",
    )
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(query_set=bad_query))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_signal_ids_mask_length_must_equal_max_signals() -> None:
    short_ids = _ids()[:63]
    short_mask = _mask_of(3)[:63]
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(signal_ids=short_ids, signal_mask=short_mask, active_k=3))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_active_k_must_match_mask() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(active_k=4))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_active_k_lower_bound() -> None:
    zero_mask = tuple([False] * MAX_SIGNALS)
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(signal_mask=zero_mask, active_k=0))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_horizon_must_be_positive() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(horizon=0))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_regime_class_rejects_unknown_label() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(regime_class="euphoria"))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_regime_id_rejects_empty() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(regime_id=""))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_task_id_must_be_nonempty_string() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(task_id=""))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_signal_ids_hash_must_be_nonempty() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(signal_ids_hash=""))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_signal_set_version_must_be_nonempty() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(signal_set_version=""))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_regime_embedding_defaults_to_none() -> None:
    t = MetaTask(**_valid_kwargs())
    assert t.regime_embedding is None
    assert t.has_regime_embedding() is False


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_regime_embedding_accepts_1d_ndarray() -> None:
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    t = MetaTask(**_valid_kwargs(regime_embedding=emb))
    assert isinstance(t.regime_embedding, np.ndarray)
    assert t.has_regime_embedding() is True


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_regime_embedding_rejects_non_ndarray() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(regime_embedding=[0.1, 0.2, 0.3]))  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_regime_embedding_rejects_non_1d() -> None:
    emb = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(regime_embedding=emb))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_t0_must_bracket_support_start() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(t0="2024-01-01T12:00:00+00:00"))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_t1_must_bracket_query_end() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(t1="2024-01-05T23:59:00+00:00"))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_support_set_must_be_nonempty() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(support_set=()))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_query_set_must_be_nonempty() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(query_set=()))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_signal_mask_must_be_tuple() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(signal_mask=list(_mask_of(3))))  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_support_entries_must_be_strings() -> None:
    bad = (
        "2024-01-01T00:00:00+00:00",
        2,
        "2024-01-03T00:00:00+00:00",
    )
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(support_set=bad))  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_query_entries_must_be_strings() -> None:
    bad = (
        "2024-01-05T00:00:00+00:00",
        123,
    )
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(query_set=bad))  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_signal_ids_entries_must_be_strings() -> None:
    bad_ids = (0, *(_ids()[1:]))
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(signal_ids=bad_ids))  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_signal_mask_entries_must_be_bool() -> None:
    bad_mask = (1, *(_mask_of(3)[1:]))
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(signal_mask=bad_mask))  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_as_record_shape_and_version() -> None:
    t = MetaTask(**_valid_kwargs())
    rec = t.as_record()
    assert rec["schema_version"] == META_TASK_SCHEMA_VERSION
    assert rec["task_id"] == t.task_id
    assert rec["regime_embedding"] is None
    assert rec["support_set"] == list(t.support_set)
    assert rec["query_set"] == list(t.query_set)
    assert rec["signal_ids"] == list(t.signal_ids)
    assert rec["signal_mask"] == list(t.signal_mask)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_as_record_embeds_ndarray_as_list() -> None:
    emb = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    t = MetaTask(**_valid_kwargs(regime_embedding=emb))
    rec = t.as_record()
    assert rec["regime_embedding"] == [1.0, 2.0, 3.0]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_horizon_type_must_be_int() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(horizon=1.5))  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_active_k_type_must_be_int() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(active_k=3.0))  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_active_k_upper_bound() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(active_k=MAX_SIGNALS + 1))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_support_set_must_be_tuple() -> None:
    with pytest.raises(DataPreconditionError):
        MetaTask(**_valid_kwargs(support_set=["2024-01-01T00:00:00+00:00"]))  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_full_active_surface() -> None:
    all_on = tuple([True] * MAX_SIGNALS)
    t = MetaTask(**_valid_kwargs(signal_mask=all_on, active_k=MAX_SIGNALS))
    assert t.active_k == MAX_SIGNALS
