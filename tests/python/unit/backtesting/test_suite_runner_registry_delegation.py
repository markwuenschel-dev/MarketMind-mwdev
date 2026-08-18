from __future__ import annotations

import polars as pl
import pytest

from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.backtesting.contracts.plan import BacktestPlan, DeterminismTier, EngineConfig
from pysrc.backtesting.data.pit import PitUnsafeFrame
from pysrc.backtesting.orchestration.plan import BacktestSuitePlan
from pysrc.backtesting.orchestration.suite_runner import BacktestSuiteRunner


@pytest.mark.determinism("d1")
def test_suite_runner_delegates_through_registry(tmp_path) -> None:
    frame = pl.DataFrame(
        {"returns": [0.01, 0.02, -0.01, 0.01], "sma_5": [1, 2, 3, 4], "sma_10": [1, 1, 2, 3]}
    )
    plan = BacktestPlan(
        engine_id="vectorized.sma",
        execution_model_id="fill.identity",
        cost_model_id="fees.zero",
        ledger_id="ledger.simple",
        validator_ids=["mechanical.v1"],
        determinism=DeterminismTier.D1,
        seed=1,
        pit_required=False,
        engine_config=EngineConfig(params={"fast_sma": 5, "slow_sma": 10}),
        run_id="suite-run",
    )
    store = BundleBacktestArtifactStore(BundleWriter(tmp_path / "bundle"))
    suite_plan = BacktestSuitePlan(
        plans=[plan],
        bundle_path=str(tmp_path / "bundle"),
        store=store,
        context={"data": PitUnsafeFrame(payload_ref="raw", metadata={"frame": frame})},
    )

    bundle_ref = BacktestSuiteRunner().execute(suite_plan)

    assert bundle_ref.run_id == "suite-run"
    assert (tmp_path / "bundle" / "execution_assumptions.json").exists()
