"""MLC-0 · Unit tests for BOCPD service ``cold_start`` flag and determinism.

Covers brief §5 Step 7 acceptance:
- Determinism: same inputs + same snapshot → same labels
- ``effective_at`` is availability time (==``decision_ts``), not change-point time
- Cold-start labels carry ``cold_start=True`` (boolean field)
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from pysrc.meta.bocpd_service import BOCPDRegimeService, RegimeLabelRecord
from pysrc.meta.regime_config import BOCPDConfig


def _synthetic_log_rv(n: int, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=-7.5, scale=0.4, size=n).astype(np.float64)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_cold_start_field_present_on_record() -> None:
    assert any(f == "cold_start" for f in RegimeLabelRecord.__dataclass_fields__)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_cold_start_true_during_burn_in() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=50, max_run_length=64)
    svc = BOCPDRegimeService(cfg)
    hist = _synthetic_log_rv(30)
    svc.initialize(hist)

    returns = np.zeros(30, dtype=np.float64)
    ts = datetime(2024, 1, 2, tzinfo=UTC)
    rec = svc.update(
        decision_ts=ts,
        log_rv=float(hist[-1]),
        log_return=0.0,
        pit_boundary_idx=10,
        log_rv_history=hist,
        returns_history=returns,
    )
    assert rec.cold_start is True
    assert rec.boundary_flag == "cold_start"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_cold_start_false_after_burn_in() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=10, max_run_length=64)
    svc = BOCPDRegimeService(cfg)
    hist = _synthetic_log_rv(300)
    svc.initialize(hist)
    returns = np.zeros(300, dtype=np.float64)
    # feed enough observations so the posterior has mass
    for i in range(40):
        svc.update(
            decision_ts=datetime(2024, 1, 2, tzinfo=UTC),
            log_rv=float(hist[i]),
            log_return=0.0,
            pit_boundary_idx=i,
            log_rv_history=hist,
            returns_history=returns,
        )
    rec = svc.update(
        decision_ts=datetime(2024, 2, 1, tzinfo=UTC),
        log_rv=float(hist[100]),
        log_return=0.001,
        pit_boundary_idx=100,
        log_rv_history=hist,
        returns_history=returns,
    )
    assert rec.cold_start is False
    assert rec.boundary_flag != "cold_start"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_effective_at_is_decision_ts_not_cp_time() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=10)
    svc = BOCPDRegimeService(cfg)
    hist = _synthetic_log_rv(60)
    svc.initialize(hist)
    returns = np.zeros(60, dtype=np.float64)
    ts = datetime(2024, 3, 15, 9, 30, tzinfo=UTC)
    rec = svc.update(
        decision_ts=ts,
        log_rv=float(hist[30]),
        log_return=0.0,
        pit_boundary_idx=30,
        log_rv_history=hist,
        returns_history=returns,
    )
    assert rec.effective_at == ts


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_determinism_same_inputs_same_labels() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=10, max_run_length=64)
    hist = _synthetic_log_rv(100)
    returns = np.zeros(100, dtype=np.float64)

    def run() -> list[str]:
        svc = BOCPDRegimeService(cfg)
        svc.initialize(hist)
        out: list[str] = []
        for i in range(30):
            rec = svc.update(
                decision_ts=datetime(2024, 1, 2, tzinfo=UTC),
                log_rv=float(hist[i]),
                log_return=0.0,
                pit_boundary_idx=i,
                log_rv_history=hist,
                returns_history=returns,
            )
            out.append(rec.regime_id)
        return out

    assert run() == run()


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_snapshot_restore_continues_deterministically() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=5, max_run_length=32)
    hist = _synthetic_log_rv(100)
    returns = np.zeros(100, dtype=np.float64)

    svc_a = BOCPDRegimeService(cfg)
    svc_a.initialize(hist)
    for i in range(20):
        svc_a.update(
            decision_ts=datetime(2024, 1, 2, tzinfo=UTC),
            log_rv=float(hist[i]),
            log_return=0.0,
            pit_boundary_idx=i,
            log_rv_history=hist,
            returns_history=returns,
        )
    snapshot = svc_a.snapshot()

    svc_b = BOCPDRegimeService.from_snapshot(snapshot, cfg)

    tail_a = [
        svc_a.update(
            decision_ts=datetime(2024, 1, 2, tzinfo=UTC),
            log_rv=float(hist[i]),
            log_return=0.0,
            pit_boundary_idx=i,
            log_rv_history=hist,
            returns_history=returns,
        ).regime_id
        for i in range(20, 30)
    ]
    tail_b = [
        svc_b.update(
            decision_ts=datetime(2024, 1, 2, tzinfo=UTC),
            log_rv=float(hist[i]),
            log_return=0.0,
            pit_boundary_idx=i,
            log_rv_history=hist,
            returns_history=returns,
        ).regime_id
        for i in range(20, 30)
    ]
    assert tail_a == tail_b
