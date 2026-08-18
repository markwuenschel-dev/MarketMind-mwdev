from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.backtesting.contracts.plan import BacktestPlan, DeterminismTier, EngineConfig
from pysrc.backtesting.contracts.protocols import AsOfView
from pysrc.backtesting.contracts.types import BacktestResult, MarketSlice
from pysrc.backtesting.data.pit import PITSafeDataView
from pysrc.backtesting.engines.vectorized.engine import VectorizedBacktestEngine
from pysrc.data.dataview import DataView
from pysrc.data.dataview_asof_adapter import DataViewAsOfAdapter
from pysrc.pipeline.stages.market_data.sources.file import FileSource


@pytest.mark.integration
@pytest.mark.determinism("d1")
def test_source_to_pit_chain_uses_adapted_file_source(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/sample_spy.csv")
    file_source = FileSource(str(fixture_path))
    frame = asyncio.run(file_source.get_historical("", "2024-01-01", "2024-01-31", eager=True))

    assert "valid_time" in frame.columns
    assert "knowledge_time" in frame.columns

    enriched_pdf = frame.to_pandas()
    enriched_pdf["returns"] = enriched_pdf["close"].pct_change().fillna(0.0)
    enriched_pdf["sma_5"] = enriched_pdf["close"].rolling(window=5, min_periods=1).mean()
    enriched_pdf["sma_10"] = enriched_pdf["close"].rolling(window=10, min_periods=1).mean()

    dataview = DataView()
    dataview.register_source(enriched_pdf, seed_fixture_membership=True)

    symbol = str(enriched_pdf["symbol"].iloc[0])
    adapter = DataViewAsOfAdapter(
        dataview=dataview,
        symbols=[symbol],
        fields=["returns", "sma_5", "sma_10"],
    )
    knowledge_dates = [
        datetime.combine(value, datetime.min.time()).replace(tzinfo=UTC)
        for value in enriched_pdf["knowledge_time"].tolist()
    ]
    pit_view = PITSafeDataView(
        view=adapter,
        metadata={"pit_enforced": True, "knowledge_dates": knowledge_dates},
    )

    assert isinstance(adapter, AsOfView)
    assert isinstance(pit_view, PITSafeDataView)

    snapshot = adapter.as_of(datetime(2024, 1, 10, tzinfo=UTC))
    assert isinstance(snapshot, MarketSlice)
    assert snapshot.prices
    assert adapter.pit_meta() is not None

    writer = BundleWriter(tmp_path / "bundle")
    store = BundleBacktestArtifactStore(writer)
    engine = VectorizedBacktestEngine()
    plan = BacktestPlan(
        engine_id="vectorized.sma",
        execution_model_id="fill.identity",
        cost_model_id="fees.zero",
        ledger_id="ledger.simple",
        validator_ids=[],
        determinism=DeterminismTier.D1,
        seed=42,
        pit_required=True,
        engine_config=EngineConfig(params={"fast_sma": 5, "slow_sma": 10}),
        run_id="phase-ib-source-chain",
    )

    result = engine.run(plan, pit_view, store)
    assert isinstance(result, BacktestResult)
    assert "execution_assumptions.json" in result.artifacts
    assert set(result.metrics) >= {
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "num_trades",
    }
