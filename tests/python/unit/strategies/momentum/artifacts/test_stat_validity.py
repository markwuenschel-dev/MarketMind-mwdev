from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysrc.backtesting.contracts.types import PitMeta
from pysrc.strategies.momentum.alpha_ir import AlphaIR
from pysrc.strategies.momentum.artifacts.signal_card import RunMeta
from pysrc.strategies.momentum.artifacts.stat_validity import build_stat_validity_payload

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def test_stat_validity_payload_is_v1_and_includes_pbo() -> None:
    alpha_ir = AlphaIR(
        signal=pd.Series([0.1, -0.1], index=["A", "B"]),
        information_coefficient=0.2,
        realized_vol=None,
        task_embedding=np.zeros(64, dtype=np.float32),
        pit_provenance=PitMeta(as_of="2024-01-01"),
        variant="xsec",
        diagnostics={"n_assets": 2},
    )
    payload = build_stat_validity_payload(
        alpha_ir=alpha_ir,
        run_meta=RunMeta(run_id="run-1"),
        returns=[0.01, -0.02, 0.015, 0.005, -0.001, 0.004],
        n_trials=2,
    )
    assert payload["schema_version"] == "v1"
    assert "pbo" in payload
    assert payload["validation_profile"]["profile_id"] == "production_v1"


def test_stat_validity_payload_normalizes_and_computes_pbo_when_pairs_provided(
    monkeypatch,
) -> None:
    alpha_ir = AlphaIR(
        signal=pd.Series([0.1, -0.1], index=["A", "B"]),
        information_coefficient=0.2,
        realized_vol=None,
        task_embedding=np.zeros(64, dtype=np.float32),
        pit_provenance=None,
        variant="xsec",
        diagnostics={"n_assets": 2},
    )
    captured: dict[str, object] = {}

    def _fake_build(raw_pairs):
        captured["raw_pairs"] = raw_pairs
        return [{"trial_id": "t1", "path_id": "p1", "net_sharpe": 0.1, "selected": True}]

    def _fake_compute(pairs, *, mode):
        captured["pairs"] = pairs
        captured["mode"] = mode
        return {"value": 0.25, "gate_result": {"passed": True}}

    monkeypatch.setattr(
        "pysrc.backtesting.validation.statistical.pbo_bridge.build_pbo_path_pairs",
        _fake_build,
    )
    monkeypatch.setattr(
        "pysrc.backtesting.validation.statistical.pbo.compute_pbo",
        _fake_compute,
    )

    payload = build_stat_validity_payload(
        alpha_ir=alpha_ir,
        run_meta=RunMeta(run_id="run-2"),
        returns=[0.01, -0.02, 0.015, 0.005, -0.001, 0.004],
        n_trials=3,
        pbo_path_pairs=[{"trial_id": "t1", "path_id": "p1", "net_sharpe": 0.1, "selected": True}],
    )

    assert captured["raw_pairs"] == [
        {"trial_id": "t1", "path_id": "p1", "net_sharpe": 0.1, "selected": True}
    ]
    assert captured["pairs"] == [
        {"trial_id": "t1", "path_id": "p1", "net_sharpe": 0.1, "selected": True}
    ]
    assert payload["pit_compliant"] is False
