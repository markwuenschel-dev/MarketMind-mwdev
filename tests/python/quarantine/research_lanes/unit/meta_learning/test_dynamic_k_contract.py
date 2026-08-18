"""MLN-04 Dynamic-K fixed-slot contract validation."""

from __future__ import annotations

import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.dynamic_k_contract import (
    MAX_SIGNALS,
    build_fixed_slot_surface_from_sparse_slots,
    validate_fixed_slot_task_surface,
    validate_signal_set_version,
)
from pysrc.registry.signal_catalog import SignalCatalog


class _MinimalSignal:
    """Satisfies :class:`SignalProtocol` for catalog registration tests."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def signal_embedding(self) -> None:
        return None

    @property
    def slot_index(self) -> int:
        return -1

    def __call__(self, data: object, params: dict[str, object]) -> list[float]:
        return []


@pytest.mark.determinism("d0")
def test_more_than_max_slots_in_sparse_map_fails(deterministic_seed: int) -> None:
    _ = deterministic_seed
    m = {i: f"s{i}" for i in range(MAX_SIGNALS + 1)}
    with pytest.raises(DataPreconditionError, match="more distinct slots"):
        build_fixed_slot_surface_from_sparse_slots(m)


@pytest.mark.determinism("d0")
def test_slot_out_of_range_fails(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(DataPreconditionError, match="out of range"):
        build_fixed_slot_surface_from_sparse_slots({MAX_SIGNALS: "bad"})


@pytest.mark.determinism("d0")
def test_inactive_slot_must_not_carry_non_empty_id(deterministic_seed: int) -> None:
    _ = deterministic_seed
    ids, mask = build_fixed_slot_surface_from_sparse_slots({0: "a"})
    bad = list(ids)
    bad[1] = "leak"
    with pytest.raises(DataPreconditionError, match="inactive slot"):
        validate_fixed_slot_task_surface(
            signal_ids=tuple(bad),
            signal_mask=mask,
            active_k=1,
        )


@pytest.mark.determinism("d0")
def test_active_k_mismatch_fails(deterministic_seed: int) -> None:
    _ = deterministic_seed
    ids, mask = build_fixed_slot_surface_from_sparse_slots({0: "a", 1: "b"})
    with pytest.raises(DataPreconditionError, match="active_k"):
        validate_fixed_slot_task_surface(signal_ids=ids, signal_mask=mask, active_k=99)


@pytest.mark.determinism("d0")
def test_mask_length_mismatch_fails(deterministic_seed: int) -> None:
    _ = deterministic_seed
    ids, _ = build_fixed_slot_surface_from_sparse_slots({0: "a"})
    short_mask = (True, False)
    with pytest.raises(DataPreconditionError, match="MAX_SIGNALS"):
        validate_fixed_slot_task_surface(
            signal_ids=ids[:2],
            signal_mask=tuple(bool(x) for x in short_mask),
            active_k=1,
        )


@pytest.mark.determinism("d0")
def test_duplicate_active_id_two_slots_fails(deterministic_seed: int) -> None:
    _ = deterministic_seed
    ids = [""] * MAX_SIGNALS
    mask = [False] * MAX_SIGNALS
    ids[0] = "dup"
    ids[1] = "dup"
    mask[0] = True
    mask[1] = True
    with pytest.raises(DataPreconditionError, match="duplicate signal id"):
        validate_fixed_slot_task_surface(
            signal_ids=tuple(ids),
            signal_mask=tuple(mask),
            active_k=2,
        )


@pytest.mark.determinism("d0")
def test_signal_catalog_rejects_65th_slot(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cat = SignalCatalog()
    for i in range(MAX_SIGNALS):
        cat.register(_MinimalSignal(f"sig_{i}"), spec_hash=f"spec_{i}", signal_name=f"n{i}")
    with pytest.raises(DataPreconditionError, match="MAX_SIGNALS"):
        cat.register(_MinimalSignal("overflow"), spec_hash="spec_overflow", signal_name="overflow")


@pytest.mark.determinism("d0")
def test_validate_signal_set_version_empty_fails(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(DataPreconditionError):
        validate_signal_set_version("  ")
