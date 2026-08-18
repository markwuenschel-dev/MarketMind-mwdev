from __future__ import annotations

import json

import polars as pl
import pytest

from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.backtesting.contracts.plan import BacktestPlan, DeterminismTier, EngineConfig
from pysrc.backtesting.data.pit import PitUnsafeFrame
from pysrc.backtesting.engines.vectorized.engine import VectorizedBacktestEngine


@pytest.mark.determinism("d1")
def test_zero_cost_assumptions_are_explicit(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "returns": [0.01, 0.02, -0.01, 0.01],
            "sma_5": [1.0, 2.0, 3.0, 4.0],
            "sma_10": [1.0, 1.5, 2.5, 3.5],
        }
    )
    plan = BacktestPlan(
        engine_id="vectorized.sma",
        execution_model_id="fill.identity",
        cost_model_id="fees.zero",
        ledger_id="ledger.simple",
        validator_ids=[],
        determinism=DeterminismTier.D1,
        seed=5,
        pit_required=False,
        engine_config=EngineConfig(params={"fast_sma": 5, "slow_sma": 10}),
    )
    bundle_dir = tmp_path / "bundle"
    store = BundleBacktestArtifactStore(BundleWriter(bundle_dir))

    VectorizedBacktestEngine().run(
        plan, PitUnsafeFrame(payload_ref="raw", metadata={"frame": frame}), store
    )
    payload = json.loads((bundle_dir / "execution_assumptions.json").read_text(encoding="utf-8"))

    assert payload["cost_model_id"] == "fees.zero"
    assert payload["commission_bps"] == 0.0
