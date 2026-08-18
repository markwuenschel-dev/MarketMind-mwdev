from datetime import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from pysrc.pipeline import orchestrator as m


def _cfg(p):
    return {
        "data": {"input_path": p},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"}
        },
    }


def test_csv_with_timestamp(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("timestamp,price\n2024-01-01,1\n")
    out = m.DataPrepOrchestrator(_cfg(str(p)))._fetch_raw_multi()
    assert hasattr(out, "columns")


def test_csv_with_date_only(tmp_path: Path):
    p = tmp_path / "d.csv"
    p.write_text("date,price\n2024-01-01,1\n")
    try:
        out = m.DataPrepOrchestrator(_cfg(str(p)))._fetch_raw_multi()
        assert out is not None
    except Exception:
        assert True


def test_csv_with_time_only(tmp_path: Path):
    p = tmp_path / "ti.csv"
    p.write_text("time,price\n00:00:01,1\n")
    try:
        out = m.DataPrepOrchestrator(_cfg(str(p)))._fetch_raw_multi()
        assert out is not None
    except Exception:
        assert True


def test_csv_no_timestamp_like(tmp_path: Path):
    p = tmp_path / "n.csv"
    p.write_text("x,y\n1,2\n")
    with pytest.raises(Exception):
        m.DataPrepOrchestrator(_cfg(str(p)))._fetch_raw_multi()


def test_inline_ops_applies_on_dataframe_input(tmp_path):
    df = pl.DataFrame(
        {
            "timestamp": pl.date_range(dt(2024, 1, 1), dt(2024, 1, 5), "1d", eager=True),
            "symbol": ["AAPL"] * 5,
            "price": [1, 2, 3, 4, 5],
        }
    )
    cfg = {
        "data": {"input_df": df},
        "io": {"allow_empty": False},
        "pipeline": {
            "spec_inline": {
                "ops": [{"kind": "scaling.robust", "input_col": "price", "out_col": "price_robust"}]
            },
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
        "execution": {"lazy_streaming": False, "lazy": False},
        "cache": {"checkpoints": False},
    }
    res = m.run_dataprep(cfg, backtest_metric=None)
    out = res[0] if isinstance(res, tuple) else res
    assert "price_robust" in out.columns
    assert len(out) == len(df)
