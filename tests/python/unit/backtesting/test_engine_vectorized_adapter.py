from __future__ import annotations

import polars as pl
import pytest

from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.backtesting.contracts.plan import BacktestPlan, DeterminismTier, EngineConfig
from pysrc.backtesting.data.pit import PitUnsafeFrame
from pysrc.backtesting.engines.vectorized.engine import VectorizedBacktestEngine


@pytest.mark.determinism("d1")
def test_vectorized_engine_emits_execution_assumptions(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "returns": [0.01, -0.01, 0.02, 0.01],
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
    store = BundleBacktestArtifactStore(BundleWriter(tmp_path / "bundle"))

    result = VectorizedBacktestEngine().run(
        plan,
        PitUnsafeFrame(payload_ref="raw", metadata={"frame": frame}),
        store,
    )

    assert "execution_assumptions.json" in result.artifacts
    assert (tmp_path / "bundle" / "execution_assumptions.json").exists()
    assert "total_return" in result.metrics
