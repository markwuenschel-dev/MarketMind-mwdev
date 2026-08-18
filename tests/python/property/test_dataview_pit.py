from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pysrc.data.dataview import DataView
from pysrc.data.dataview_asof_adapter import DataViewAsOfAdapter


@st.composite
def _rows_strategy(draw, min_rows: int = 1, max_rows: int = 20) -> pd.DataFrame:
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    base = date(2024, 1, 1)
    symbols = ["S1"] * n
    deltas = [draw(st.integers(min_value=0, max_value=10)) for _ in range(n)]
    valid_times = [base + timedelta(days=d) for d in deltas]
    knowledge_times = [base + timedelta(days=d) for d in deltas]
    prices = [float(draw(st.floats(min_value=50.0, max_value=150.0))) for _ in range(n)]
    return pd.DataFrame(
        {
            "symbol": symbols,
            "valid_time": valid_times,
            "knowledge_time": knowledge_times,
            "close": prices,
        }
    )


@pytest.mark.determinism("d2")
@given(rows=_rows_strategy(), offset_days=st.integers(min_value=0, max_value=10))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_adapter_preserves_pit_invariant(rows: pd.DataFrame, offset_days: int) -> None:
    dv = DataView()
    dv.register_source(rows)
    adapter = DataViewAsOfAdapter(dataview=dv, symbols=["S1"], fields=["close"])

    T = date(2024, 1, 1) + timedelta(days=offset_days)
    ts = datetime(T.year, T.month, T.day)
    snapshot = adapter.as_of(ts)
    for rec in snapshot.prices:
        assert rec["valid_time"] <= T


@pytest.mark.determinism("d2")
@given(rows=_rows_strategy())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_adapter_pit_meta_always_set(rows: pd.DataFrame) -> None:
    dv = DataView()
    dv.register_source(rows)
    adapter = DataViewAsOfAdapter(dataview=dv, symbols=["S1"], fields=["close"])

    T = date(2024, 1, 5)
    ts = datetime(T.year, T.month, T.day)
    _ = adapter.as_of(ts)
    meta = adapter.pit_meta()
    assert meta is not None


def _make_store(symbol: str, valid: date, know: date, **fields: float) -> pd.DataFrame:
    d = {"symbol": [symbol], "valid_time": [valid], "knowledge_time": [know]}
    d.update(fields)
    return pd.DataFrame(d)


@pytest.mark.determinism("d1")
@given(
    T=st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)),
    future_days=st.integers(min_value=1, max_value=365),
)
@settings(max_examples=50, deadline=5000)
def test_future_valid_time_never_observable_at_T(T: date, future_days: int) -> None:
    """Rows with valid_time > T must never appear in as_of(..., T)."""
    dv = DataView()
    future_valid = T + timedelta(days=future_days)
    df = _make_store("A", future_valid, T, close=100.0)
    dv.register_source(df)
    out = dv.as_of(["A"], ["close"], T)
    assert out.empty


@pytest.mark.determinism("d1")
@given(
    T=st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)),
    future_days=st.integers(min_value=1, max_value=365),
)
@settings(max_examples=50, deadline=5000)
def test_future_knowledge_time_never_observable_at_T(T: date, future_days: int) -> None:
    """Rows with knowledge_time > T must never appear in as_of(..., T)."""
    dv = DataView()
    future_know = T + timedelta(days=future_days)
    df = _make_store("A", T, future_know, close=100.0)
    dv.register_source(df)
    out = dv.as_of(["A"], ["close"], T)
    assert out.empty


@pytest.mark.determinism("d1")
def test_future_restatements_do_not_alter_earlier_query_results() -> None:
    """Query at T1 then add a restatement with knowledge_time > T1; query at T1 again must be unchanged."""
    dv = DataView()
    df1 = _make_store("A", date(2024, 1, 1), date(2024, 1, 2), close=100.0)
    dv.register_source(df1)
    out1 = dv.as_of(["A"], ["close"], date(2024, 1, 2))
    assert len(out1) == 1
    assert out1["close"].iloc[0] == 100.0
    df2 = _make_store("A", date(2024, 1, 1), date(2024, 1, 5), close=101.0)
    dv.register_source(df2)
    out1_again = dv.as_of(["A"], ["close"], date(2024, 1, 2))
    assert len(out1_again) == 1
    assert out1_again["close"].iloc[0] == 100.0


@pytest.mark.determinism("d1")
def test_later_queries_see_superset_or_equal_visible_rows() -> None:
    """Query at T1 then T2 >= T1; symbols visible at T1 are a subset of (or equal to) visible at T2 for same store."""
    dv = DataView()
    df = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 2), date(2024, 1, 3)],
            "close": [100.0, 200.0],
        }
    )
    dv.register_source(df)
    out_jan2 = dv.as_of(["A", "B"], ["close"], date(2024, 1, 2))
    out_jan4 = dv.as_of(["A", "B"], ["close"], date(2024, 1, 4))
    syms_jan2 = set(out_jan2["symbol"].tolist())
    syms_jan4 = set(out_jan4["symbol"].tolist())
    assert syms_jan2 <= syms_jan4
    assert "A" in syms_jan2
    assert "A" in syms_jan4
    assert "B" in syms_jan4


@pytest.mark.determinism("d1")
def test_poison_pill_future_rows_never_leak() -> None:
    """A row with valid_time and knowledge_time both > T must not affect as_of(..., T)."""
    dv = DataView()
    df_visible = _make_store("A", date(2024, 1, 1), date(2024, 1, 1), close=50.0)
    df_poison = _make_store("A", date(2024, 6, 1), date(2024, 6, 1), close=999.0)
    dv.register_source(df_visible)
    dv.register_source(df_poison)
    out = dv.as_of(["A"], ["close"], date(2024, 1, 2))
    assert len(out) == 1
    assert out["close"].iloc[0] == 50.0


@pytest.mark.determinism("d1")
def test_adapter_pit_meta_none_before_first_snapshot() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["S1"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv = DataView()
    dv.register_source(frame)
    adapter = DataViewAsOfAdapter(dataview=dv, symbols=["S1"], fields=["close"])

    assert adapter.pit_meta() is None


@pytest.mark.determinism("d1")
def test_adapter_as_of_emits_market_slice_metadata_contract() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["S1"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [101.0],
        }
    )
    dv = DataView()
    dv.register_source(frame)
    adapter = DataViewAsOfAdapter(dataview=dv, symbols=["S1"], fields=["close"])

    ts = datetime(2024, 1, 2, 9, 30)
    snapshot = adapter.as_of(ts)

    assert snapshot.as_of == ts.isoformat()
    assert snapshot.metadata["pit_front_door"] == "pysrc.data.dataview.DataView"
    assert snapshot.metadata["knowledge_date"] == "2024-01-02"
    assert snapshot.prices[0]["close"] == 101.0
    assert snapshot.pit_meta is not None
    assert snapshot.pit_meta.knowledge_cutoff == "2024-01-02"
