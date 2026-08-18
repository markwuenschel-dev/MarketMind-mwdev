from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pysrc.backtesting.contracts.plan import BacktestPlan, DeterminismTier, EngineConfig
from pysrc.backtesting.contracts.protocols import AsOfView
from pysrc.backtesting.contracts.types import MarketSlice
from pysrc.backtesting.data.pit import PITSafeDataView, PitUnsafeFrame
from pysrc.backtesting.engines.vectorized.engine import VectorizedBacktestEngine
from pysrc.data.dataview import DataView
from pysrc.data.dataview_asof_adapter import DataViewAsOfAdapter
from pysrc.pipeline.orchestrator import (
    OrchestratorConfig,
    _normalize_strategy_context,
    run_orchestration,
)
from pysrc.strategies.pipeline_strategy import StrategyContext


class _CapturingEngine:
    def __init__(self) -> None:
        self.seen_plan: BacktestPlan | None = None
        self.seen_data: Any = None

    def run(self, plan: BacktestPlan, data: Any, store: Any) -> Any:
        self.seen_plan = plan
        self.seen_data = data

        class _Result:
            metrics: dict[str, float] = {}

        return _Result()


@pytest.mark.determinism("d1")
def test_canonical_plan_sets_pit_required(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "close": [100.0, 101.0],
            "volume": [1_000_000, 1_000_100],
        }
    )

    def _fake_load_ohlcv(path: Path) -> pl.DataFrame:  # type: ignore[unused-argument]
        return df

    monkeypatch.setattr("pysrc.pipeline.orchestrator.load_ohlcv", _fake_load_ohlcv)

    capturing = _CapturingEngine()
    monkeypatch.setattr("pysrc.pipeline.orchestrator.resolve_engine", lambda _engine_id: capturing)

    cfg = OrchestratorConfig(input_path=tmp_path / "data.csv", bundle_dir=tmp_path / "bundle")
    exit_code, _ = run_orchestration(cfg)

    assert exit_code in (0, 1)
    assert capturing.seen_plan is not None
    assert capturing.seen_plan.pit_required is True


@pytest.mark.determinism("d1")
def test_canonical_data_is_pit_safe_view(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "close": [100.0, 101.0],
            "volume": [1_000_000, 1_000_100],
        }
    )

    def _fake_load_ohlcv(path: Path) -> pl.DataFrame:  # type: ignore[unused-argument]
        return df

    monkeypatch.setattr("pysrc.pipeline.orchestrator.load_ohlcv", _fake_load_ohlcv)

    capturing = _CapturingEngine()
    monkeypatch.setattr("pysrc.pipeline.orchestrator.resolve_engine", lambda _engine_id: capturing)

    cfg = OrchestratorConfig(input_path=tmp_path / "data.csv", bundle_dir=tmp_path / "bundle")
    run_orchestration(cfg)

    assert isinstance(capturing.seen_data, PITSafeDataView)


@pytest.mark.determinism("d1")
def test_adapter_satisfies_asof_protocol() -> None:
    dv = DataView()
    from datetime import date

    import pandas as pd  # type: ignore[import-untyped]

    pdf = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv.register_source(pdf)

    adapter = DataViewAsOfAdapter(dataview=dv, symbols=["A"], fields=["close"])
    assert isinstance(adapter, AsOfView)


@pytest.mark.determinism("d1")
def test_adapter_pit_meta_populated() -> None:
    dv = DataView()
    from datetime import date

    import pandas as pd  # type: ignore[import-untyped]

    pdf = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv.register_source(pdf)
    adapter = DataViewAsOfAdapter(dataview=dv, symbols=["A"], fields=["close"])

    ts = datetime(2024, 1, 2, tzinfo=UTC)
    slice_ = adapter.as_of(ts)
    assert isinstance(slice_, MarketSlice)
    meta = adapter.pit_meta()
    assert meta is not None
    assert meta.source == "pysrc.data.dataview.DataView"


@pytest.mark.determinism("d1")
def test_engine_rejects_plain_frame_when_pit_required(tmp_path: Path) -> None:
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
        engine_config=EngineConfig(),
        run_id="test",
    )
    df = pl.DataFrame({"returns": [0.0], "sma_5": [0.0], "sma_10": [0.0]})
    frame = PitUnsafeFrame(payload_ref="test", metadata={"frame": df})
    store = None  # type: ignore[assignment]

    with pytest.raises(Exception):
        engine.run(plan, frame, store)


@pytest.mark.determinism("d1")
def test_fixture_plan_may_remain_pit_optional(tmp_path: Path) -> None:
    engine = VectorizedBacktestEngine()
    plan = BacktestPlan(
        engine_id="vectorized.sma",
        execution_model_id="fill.identity",
        cost_model_id="fees.zero",
        ledger_id="ledger.simple",
        validator_ids=[],
        determinism=DeterminismTier.D1,
        seed=42,
        pit_required=False,
        engine_config=EngineConfig(),
        run_id="test",
    )
    df = pl.DataFrame({"returns": [0.0], "sma_5": [0.0], "sma_10": [0.0]})
    frame = PitUnsafeFrame(payload_ref="test", metadata={"frame": df})
    store = None  # type: ignore[assignment]

    result = engine.run(plan, frame, store)
    assert result.metrics is not None


@pytest.mark.determinism("d1")
def test_strategy_route_normalizes_pit_safe_input(tmp_path: Path) -> None:
    from datetime import date

    import pandas as pd  # type: ignore[import-untyped]

    dv = DataView(pit_required=True)
    pdf = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 2)],
            "knowledge_time": [date(2024, 1, 1), date(2024, 1, 2)],
            "close": [100.0, 101.0],
        }
    )
    dv.register_source(pdf)
    adapter = DataViewAsOfAdapter(dataview=dv, symbols=["A"], fields=["close"])
    knowledge_dates = [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    ]
    pit_view = PITSafeDataView(view=adapter, metadata={"knowledge_dates": knowledge_dates})

    normalized = _normalize_strategy_context(
        StrategyContext(prices=pd.DataFrame({"close": [1.0]}), cache_dir=tmp_path),
        pit_input=pit_view,
        knowledge_dates=None,
    )

    assert isinstance(normalized.prices, pd.DataFrame)
    assert list(normalized.prices.columns) == ["close", "valid_time", "knowledge_time"]
    assert normalized.pit_provenance is not None


@pytest.mark.determinism("d1")
def test_strategy_route_rejects_missing_knowledge_dates(tmp_path: Path) -> None:
    from datetime import date

    import pandas as pd  # type: ignore[import-untyped]

    dv = DataView(pit_required=True)
    pdf = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv.register_source(pdf)
    adapter = DataViewAsOfAdapter(dataview=dv, symbols=["A"], fields=["close"])
    pit_view = PITSafeDataView(view=adapter, metadata={})

    with pytest.raises(Exception):
        _normalize_strategy_context(
            StrategyContext(prices=pd.DataFrame({"close": [1.0]}), cache_dir=tmp_path),
            pit_input=pit_view,
            knowledge_dates=None,
        )
