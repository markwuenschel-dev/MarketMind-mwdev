from __future__ import annotations

import pytest

from pysrc.backtesting.contracts.errors import DeterminismTierMissingError, PitUnsafeInputError
from pysrc.backtesting.contracts.plan import BacktestPlan, DeterminismTier, EngineConfig
from pysrc.backtesting.data.pit import PitUnsafeFrame
from pysrc.backtesting.engines.vectorized.engine import VectorizedBacktestEngine


class StubStore:
    def put_json(self, role, payload):
        return None

    def put_bytes(self, role, payload, media_type):
        return None

    def get_json(self, ref):
        return {}


@pytest.mark.determinism("d1")
def test_pit_required_plan_rejects_unsafe_input() -> None:
    plan = BacktestPlan(
        engine_id="vectorized.sma",
        execution_model_id="fill.identity",
        cost_model_id="fees.zero",
        ledger_id="ledger.simple",
        validator_ids=[],
        determinism=DeterminismTier.D1,
        seed=1,
        pit_required=True,
        engine_config=EngineConfig(params={"fast_sma": 5, "slow_sma": 10}),
    )

    with pytest.raises(PitUnsafeInputError):
        VectorizedBacktestEngine().run(plan, PitUnsafeFrame(payload_ref="raw"), StubStore())


@pytest.mark.determinism("d1")
def test_missing_determinism_tier_fails_closed() -> None:
    with pytest.raises(DeterminismTierMissingError):
        BacktestPlan(
            engine_id="vectorized.sma",
            execution_model_id="fill.identity",
            cost_model_id="fees.zero",
            ledger_id="ledger.simple",
            validator_ids=[],
            determinism=None,
            seed=1,
            pit_required=False,
            engine_config=None,
        )
