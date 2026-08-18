"""Stub vs real W1 lane isolation (Agent F inventory)."""

from __future__ import annotations

import pytest

pytest.importorskip("xgboost")

import pysrc.meta.w1_baseline_runner as w1r
from pysrc.meta.w1_baseline_config import W1BaselineConfig
from pysrc.meta.w1_baseline_errors import W1BaselineEvidenceError
from pysrc.meta.w1_baseline_incumbent import XGBoostIncumbentBaseline
from pysrc.meta.w1_baseline_io import build_w1_synthetic_task_pool
from pysrc.meta.w1_baseline_runner import run_w1_baseline_evidence


@pytest.mark.determinism("d1")
def test_synthetic_lane_rejects_xgb_incumbent(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)
    xgb = XGBoostIncumbentBaseline()
    with pytest.raises(W1BaselineEvidenceError, match="synthetic_stub"):
        run_w1_baseline_evidence(cfg, pool, incumbent=xgb)


@pytest.mark.determinism("d1")
def test_real_lane_rejects_stub_class_instance(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)
    stub = w1r._XGBoostIncumbentStub()
    with pytest.raises(W1BaselineEvidenceError, match="stub"):
        run_w1_baseline_evidence(cfg, pool, evidence_lane="governed_real", incumbent=stub)
