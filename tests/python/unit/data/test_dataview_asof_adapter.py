from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from pysrc.data.dataview_asof_adapter import DataViewAsOfAdapter


class _StubDataView:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[dict[str, object]] = []

    def as_of(self, *, symbols: list[str], fields: list[str], knowledge_date):  # type: ignore[no-untyped-def]
        self.calls.append(
            {
                "symbols": symbols,
                "fields": fields,
                "knowledge_date": knowledge_date,
            }
        )
        return self.frame


@pytest.mark.determinism("d1")
def test_adapter_pit_meta_is_none_before_any_snapshot() -> None:
    adapter = DataViewAsOfAdapter(
        dataview=_StubDataView(pd.DataFrame()),
        symbols=["AAPL"],
        fields=["close"],
    )

    assert adapter.pit_meta() is None


@pytest.mark.determinism("d1")
def test_adapter_as_of_emits_market_slice_and_provenance() -> None:
    snapshot = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "valid_time": ["2024-01-01"],
            "knowledge_time": ["2024-01-02"],
            "close": [101.5],
        }
    )
    dataview = _StubDataView(snapshot)
    adapter = DataViewAsOfAdapter(
        dataview=dataview,
        symbols=["AAPL"],
        fields=["close"],
    )

    ts = datetime(2024, 1, 3, 14, 30, 0)
    market_slice = adapter.as_of(ts)
    pit_meta = adapter.pit_meta()

    assert dataview.calls == [
        {
            "symbols": ["AAPL"],
            "fields": ["close"],
            "knowledge_date": ts.date(),
        }
    ]
    assert market_slice.as_of == ts.isoformat()
    assert market_slice.prices == snapshot.to_dict(orient="records")
    assert market_slice.features == []
    assert market_slice.metadata["pit_enforced"] is True
    assert market_slice.metadata["knowledge_date"] == "2024-01-03"
    assert market_slice.pit_meta is not None
    assert market_slice.pit_meta.knowledge_cutoff == "2024-01-03"
    assert pit_meta is not None
    assert pit_meta.as_of == ts.isoformat()
    assert pit_meta.knowledge_cutoff == "2024-01-03"
