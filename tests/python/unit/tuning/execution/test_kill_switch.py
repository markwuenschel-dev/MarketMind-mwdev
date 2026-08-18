"""Unit tests for kill switch state."""

from __future__ import annotations

import pytest

from pysrc.tuning.execution.kill_switch import KillSwitchState


@pytest.mark.determinism("d1")
def test_kill_switch_blocks_new_orders_when_engaged(deterministic_seed: int) -> None:
    _ = deterministic_seed
    switch = KillSwitchState()
    assert switch.allows_new_orders() is True
    switch.engage("reconciliation mismatch")
    assert switch.block_new_orders is True
    assert switch.allows_new_orders() is False
    assert switch.reason == "reconciliation mismatch"


@pytest.mark.determinism("d1")
def test_kill_switch_release_clears_block(deterministic_seed: int) -> None:
    _ = deterministic_seed
    switch = KillSwitchState()
    switch.engage("broker stream stale")
    switch.release()
    assert switch.block_new_orders is False
    assert switch.reason == ""
    assert switch.allows_new_orders() is True


@pytest.mark.determinism("d1")
def test_kill_switch_engage_requires_reason(deterministic_seed: int) -> None:
    _ = deterministic_seed
    switch = KillSwitchState()
    with pytest.raises(ValueError, match="non-empty reason"):
        switch.engage("")
