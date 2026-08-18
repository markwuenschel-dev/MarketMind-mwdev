"""Tests for SignalCatalog: slot assignment, idempotent register, no slot reuse."""

from __future__ import annotations

import pytest

from pysrc.registry.signal_catalog import (
    SignalCatalog,
    _compute_spec_hash,
    _StubSignal,
    get_catalog,
)


@pytest.mark.determinism("d0")
def test_slot_assignment_monotonic() -> None:
    """Slot indices are assigned in order 0, 1, 2, ..."""
    catalog = SignalCatalog()
    s0 = _StubSignal("a")
    s1 = _StubSignal("b")
    s2 = _StubSignal("c")
    assert catalog.register(s0, signal_name="a") == 0
    assert catalog.register(s1, signal_name="b") == 1
    assert catalog.register(s2, signal_name="c") == 2
    assert catalog.get_by_slot(0) is not None
    assert catalog.get_by_slot(0).slot_index == 0
    assert catalog.get_by_slot(1) is not None
    assert catalog.get_by_slot(1).slot_index == 1
    assert catalog.get_by_slot(2) is not None
    assert catalog.get_by_slot(2).slot_index == 2


@pytest.mark.determinism("d0")
def test_register_idempotent_by_spec_hash() -> None:
    """Re-registering the same spec_hash returns existing slot_index and does not increment."""
    catalog = SignalCatalog()
    spec_hash = _compute_spec_hash("same_signal")
    s = _StubSignal("same_signal")
    slot1 = catalog.register(s, spec_hash=spec_hash, signal_name="same_signal")
    slot2 = catalog.register(s, spec_hash=spec_hash, signal_name="same_signal")
    assert slot1 == slot2
    assert len(catalog) == 1


@pytest.mark.determinism("d0")
def test_slot_immutable_after_assignment() -> None:
    """Once assigned, a signal's slot_index does not change (via RegisteredSignal)."""
    catalog = SignalCatalog()
    s = _StubSignal("x")
    slot = catalog.register(s, signal_name="x")
    reg = catalog.get_by_spec_hash(_compute_spec_hash("x"))
    assert reg is not None
    assert reg.slot_index == slot
    assert reg.spec_hash == _compute_spec_hash("x")


@pytest.mark.determinism("d0")
def test_no_slot_reuse_phase_i() -> None:
    """Phase I: we do not reassign slots; each new signal gets next slot."""
    catalog = SignalCatalog()
    catalog.register(_StubSignal("first"), signal_name="first")
    catalog.register(_StubSignal("second"), signal_name="second")
    assert catalog.get_by_slot(0).signal_name == "first"
    assert catalog.get_by_slot(1).signal_name == "second"


@pytest.mark.determinism("d0")
def test_dedup_by_spec_hash() -> None:
    """Different signals with same spec_hash are treated as one (idempotent)."""
    catalog = SignalCatalog()
    h = _compute_spec_hash("dup")
    catalog.register(_StubSignal("dup"), spec_hash=h, signal_name="dup")
    catalog.register(_StubSignal("dup2"), spec_hash=h, signal_name="dup2")
    assert len(catalog) == 1
    assert catalog.get_by_spec_hash(h) is not None


@pytest.mark.determinism("d0")
def test_five_starters_get_slots_0_4() -> None:
    """get_catalog() registers 5 starter signals with slots 0-4."""
    catalog = get_catalog()
    assert len(catalog) >= 5
    for i in range(5):
        reg = catalog.get_by_slot(i)
        assert reg is not None, f"slot {i} should be assigned"
