from __future__ import annotations

import json

import pytest

from pysrc.backtesting.contracts.plan import BacktestPlan, DeterminismTier, EngineConfig
from pysrc.backtesting.contracts.protocols import AsOfView
from pysrc.backtesting.contracts.types import (
    ArtifactRef,
    MarketSlice,
    PitMeta,
    RunBundleRef,
    to_primitive,
)


class StubView:
    def as_of(self, ts):
        return MarketSlice(as_of=ts.isoformat())

    def pit_meta(self):
        return PitMeta(as_of="2026-01-01T00:00:00+00:00")


@pytest.mark.determinism("d1")
def test_contract_dataclasses_round_trip_to_json() -> None:
    plan = BacktestPlan(
        engine_id="vectorized.sma",
        execution_model_id="fill.identity",
        cost_model_id="fees.zero",
        ledger_id="ledger.simple",
        validator_ids=["statistical.v1"],
        determinism=DeterminismTier.D1,
        seed=7,
        pit_required=False,
        engine_config=EngineConfig(params={"fast_sma": 5}),
        run_id="run-1",
    )
    bundle_ref = RunBundleRef(
        bundle_path="bundles/run-1",
        run_id="run-1",
        manifest_ref=ArtifactRef(role="bundle_manifest.json", path="bundle_manifest.json"),
    )

    json_blob = json.dumps({"plan": to_primitive(plan), "bundle": to_primitive(bundle_ref)})
    payload = json.loads(json_blob)

    assert payload["plan"]["engine_id"] == "vectorized.sma"
    assert payload["plan"]["determinism"] == "d1"
    assert payload["bundle"]["manifest_ref"]["role"] == "bundle_manifest.json"


@pytest.mark.determinism("d1")
def test_asof_view_protocol_is_runtime_checkable() -> None:
    assert isinstance(StubView(), AsOfView)


@pytest.mark.determinism("d1")
def test_types_to_primitive_normalizes_tuple_datetime_and_enum() -> None:
    from datetime import datetime

    from pysrc.backtesting.contracts.types import ValidationStatus, to_primitive

    payload = {
        "status": ValidationStatus.PASS,
        "points": (1, datetime(2024, 1, 2, 3, 4, 5)),
    }

    out = to_primitive(payload)

    assert out["status"] == "PASS"
    assert out["points"] == [1, "2024-01-02T03:04:05"]
