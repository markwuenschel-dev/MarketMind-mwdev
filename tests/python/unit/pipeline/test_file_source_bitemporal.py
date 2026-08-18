from __future__ import annotations

import asyncio
from contextlib import aclosing
from pathlib import Path

import polars as pl
import pytest

from pysrc.data.dataview import DataView
from pysrc.pipeline.stages.market_data.exceptions import DataFetchError
from pysrc.pipeline.stages.market_data.sources.file import FileSource

pytestmark = pytest.mark.determinism("d1")


def _daily_csv(tmp_path: Path) -> Path:
    p = tmp_path / "daily.csv"
    p.write_text(
        "date,open,high,low,close,volume\n"
        "2024-01-02,100,101,99,100.5,1000000\n"
        "2024-01-03,100.5,102,100,101,1100000\n"
        "2024-01-04,101,103,101,102,1200000\n"
    )
    return p


def _temporal_csv(tmp_path: Path) -> Path:
    p = tmp_path / "daily_with_temporal.csv"
    p.write_text(
        "date,valid_time,knowledge_time,open,high,low,close,volume\n"
        "2024-01-02,2024-02-01,2024-02-02,100,101,99,100.5,1000000\n"
    )
    return p


@pytest.mark.determinism("d1")
def test_loader_emits_valid_time(tmp_path: Path) -> None:
    src = FileSource(str(_daily_csv(tmp_path)))
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    assert isinstance(df, pl.DataFrame)
    assert "valid_time" in df.columns


@pytest.mark.determinism("d1")
def test_loader_emits_knowledge_time(tmp_path: Path) -> None:
    src = FileSource(str(_daily_csv(tmp_path)))
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    assert "knowledge_time" in df.columns


@pytest.mark.determinism("d1")
def test_temporal_columns_are_date_typed(tmp_path: Path) -> None:
    src = FileSource(str(_daily_csv(tmp_path)))
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    assert df["valid_time"].dtype == pl.Date
    assert df["knowledge_time"].dtype == pl.Date


@pytest.mark.determinism("d1")
def test_timestamp_and_date_remain_present(tmp_path: Path) -> None:
    src = FileSource(str(_daily_csv(tmp_path)))
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    assert "timestamp" in df.columns
    assert "date" in df.columns
    assert "symbol" in df.columns


@pytest.mark.determinism("d1")
def test_valid_time_equals_knowledge_time_equals_bar_date_for_daily_fixture(tmp_path: Path) -> None:
    src = FileSource(str(_daily_csv(tmp_path)))
    df = asyncio.run(src.get_historical("", "2024-01-02", "2024-01-05", eager=True))
    for idx in range(len(df)):
        assert df["valid_time"][idx] == df["knowledge_time"][idx]
        assert df["valid_time"][idx] == df["timestamp"][idx].date()


@pytest.mark.determinism("d1")
def test_existing_loader_behavior_unchanged(tmp_path: Path) -> None:
    src = FileSource(str(_daily_csv(tmp_path)))
    df = asyncio.run(src.get_historical("", "2024-01-02", "2024-01-10", eager=True))
    assert not df.is_empty()
    assert "timestamp" in df.columns
    assert df["timestamp"].dtype == pl.Datetime
    assert df["close"].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)


@pytest.mark.determinism("d1")
def test_existing_temporal_columns_are_preserved(tmp_path: Path) -> None:
    src = FileSource(str(_temporal_csv(tmp_path)))
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-12-31", eager=True))
    assert str(df["valid_time"][0]) == "2024-02-01"
    assert str(df["knowledge_time"][0]) == "2024-02-02"


@pytest.mark.determinism("d1")
def test_existing_non_temporal_columns_are_preserved(tmp_path: Path) -> None:
    src = FileSource(str(_daily_csv(tmp_path)))
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    assert {"date", "open", "high", "low", "close", "volume", "timestamp", "symbol"} <= set(
        df.columns
    )


@pytest.mark.determinism("d1")
def test_file_output_registers_into_dataview(tmp_path: Path) -> None:
    src = FileSource(str(_daily_csv(tmp_path)))
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    dataview = DataView()
    dataview.register_source(df.to_pandas())
    snapshot = dataview.as_of(
        symbols=[str(df["symbol"][0])], fields=["close"], knowledge_date=df["valid_time"][0]
    )
    assert not snapshot.empty


@pytest.mark.determinism("d1")
def test_phase_i_approximation_marker_present() -> None:
    source_text = Path("pysrc/pipeline/stages/market_data/sources/file.py").read_text()
    assert "Phase I approximation" in source_text


@pytest.mark.determinism("d1")
@pytest.mark.asyncio
async def test_file_source_context_manager_enter_exit_and_close(tmp_path: Path) -> None:
    """Exercise DataSource __aenter__, __aexit__, and close for contracts.py coverage."""
    src = FileSource(str(_daily_csv(tmp_path)))
    async with src:
        assert src is not None
    await src.close()


@pytest.mark.determinism("d1")
def test_file_source_accepts_dict_config(tmp_path: Path) -> None:
    """FileSource accepts dict config with file_path and optional format/tail."""
    p = _daily_csv(tmp_path)
    src = FileSource({"file_path": str(p), "format": "csv", "tail": False})
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    assert not df.is_empty()
    assert "timestamp" in df.columns


@pytest.mark.determinism("d1")
def test_file_source_unsupported_format_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported file format"):
        FileSource({"file_path": "x.xyz", "format": "xyz"})


@pytest.mark.determinism("d1")
def test_file_source_fallback_symbol_uses_stem_when_symbol_empty(tmp_path: Path) -> None:
    """When symbol is empty, _fallback_symbol returns file stem."""
    p = _daily_csv(tmp_path)
    src = FileSource(str(p))
    assert src._fallback_symbol("") == "daily"
    assert src._fallback_symbol("  ") == "daily"


@pytest.mark.determinism("d1")
def test_file_source_fallback_symbol_uses_requested_when_non_empty(tmp_path: Path) -> None:
    src = FileSource(str(_daily_csv(tmp_path)))
    assert src._fallback_symbol("SPY") == "SPY"
    assert src._fallback_symbol("  AAPL  ") == "AAPL"


@pytest.mark.determinism("d1")
def test_file_source_no_timestamp_or_date_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "no_date.csv"
    p.write_text("a,b,c\n1,2,3\n")
    src = FileSource(str(p))
    with pytest.raises(DataFetchError, match="No timestamp or date column"):
        asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))


@pytest.mark.determinism("d1")
def test_file_source_eager_empty_after_date_filter_raises(tmp_path: Path) -> None:
    """When date filter yields no rows and eager=True, DataFetchError is raised."""
    src = FileSource(str(_daily_csv(tmp_path)))
    with pytest.raises(DataFetchError, match="No historical data found"):
        asyncio.run(src.get_historical("", "2020-01-01", "2020-01-15", eager=True))


@pytest.mark.determinism("d1")
def test_file_source_returns_lazy_when_not_eager(tmp_path: Path) -> None:
    src = FileSource(str(_daily_csv(tmp_path)))
    lf = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=False))
    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    assert not df.is_empty()


@pytest.mark.determinism("d1")
def test_file_source_get_historical_sync(tmp_path: Path) -> None:
    src = FileSource(str(_daily_csv(tmp_path)))
    df = src.get_historical_sync("", "2024-01-01", "2024-01-31", eager=True)
    assert isinstance(df, pl.DataFrame)
    assert "valid_time" in df.columns


@pytest.mark.determinism("d1")
def test_file_source_nonexistent_file_raises_data_fetch_error(tmp_path: Path) -> None:
    src = FileSource(str(tmp_path / "nonexistent.csv"))
    with pytest.raises(DataFetchError, match="Failed to load historical"):
        asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))


@pytest.mark.determinism("d1")
def test_file_source_csv_with_timestamp_column(tmp_path: Path) -> None:
    """Branch: tcol == 'timestamp' uses cast instead of multi-format parse."""
    p = tmp_path / "ts.csv"
    p.write_text("timestamp,open,close,volume\n2024-01-02,100,101,1e6\n2024-01-03,101,102,1e6\n")
    src = FileSource(str(p))
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    assert len(df) == 2
    assert df["timestamp"].dtype == pl.Datetime


@pytest.mark.determinism("d1")
def test_file_source_filter_by_symbol_when_column_present(tmp_path: Path) -> None:
    """When CSV has symbol column and symbol is non-empty, filter by symbol."""
    p = tmp_path / "multi.csv"
    p.write_text(
        "date,symbol,open,high,low,close,volume\n"
        "2024-01-02,A,100,101,99,100.5,1e6\n"
        "2024-01-02,B,200,201,199,200.5,2e6\n"
        "2024-01-03,A,101,102,100,101.5,1e6\n"
    )
    src = FileSource(str(p))
    df = asyncio.run(src.get_historical("A", "2024-01-01", "2024-01-31", eager=True))
    assert df["symbol"].to_list() == ["A", "A"]
    assert len(df) == 2


@pytest.mark.determinism("d1")
def test_file_source_parquet_format(tmp_path: Path) -> None:
    """FileSource reads parquet when format is parquet."""
    csv_p = _daily_csv(tmp_path)
    df_csv = pl.read_csv(str(csv_p), try_parse_dates=True)
    parquet_p = tmp_path / "daily.parquet"
    df_csv.write_parquet(parquet_p)
    src = FileSource(str(parquet_p))
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    assert len(df) == 3
    assert "valid_time" in df.columns


@pytest.mark.determinism("d1")
def test_file_source_json_format(tmp_path: Path) -> None:
    """FileSource reads ndjson when format is json."""
    p = tmp_path / "daily.json"
    p.write_text(
        '{"date":"2024-01-02","open":100,"high":101,"low":99,"close":100.5,"volume":1e6}\n'
        '{"date":"2024-01-03","open":101,"high":102,"low":100,"close":101.5,"volume":1e6}\n'
    )
    src = FileSource({"file_path": str(p), "format": "json"})
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    assert len(df) >= 1
    assert "timestamp" in df.columns


@pytest.mark.determinism("d1")
@pytest.mark.asyncio
async def test_file_source_realtime_raises_without_tail() -> None:
    """get_realtime raises NotImplementedError when tail=False."""
    src = FileSource({"file_path": "/nonexistent.csv", "tail": False})
    with pytest.raises(NotImplementedError, match="Real-time data not supported"):
        async with aclosing(src.get_realtime("X")) as stream:
            async for _ in stream:
                pass


@pytest.mark.determinism("d1")
def test_file_source_csv_with_time_column(tmp_path: Path) -> None:
    """Branch: tcol from 'time' when timestamp/date not present."""
    p = tmp_path / "tm.csv"
    p.write_text("time,open,close,volume\n2024-01-02,100,101,1e6\n2024-01-03,101,102,1e6\n")
    src = FileSource(str(p))
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    assert len(df) == 2
    assert df["timestamp"].dtype == pl.Datetime


@pytest.mark.determinism("d1")
def test_file_source_ipc_format(tmp_path: Path) -> None:
    """FileSource reads IPC when format is ipc."""
    csv_p = _daily_csv(tmp_path)
    df_csv = pl.read_csv(str(csv_p), try_parse_dates=True)
    ipc_p = tmp_path / "daily.ipc"
    df_csv.write_ipc(ipc_p)
    src = FileSource({"file_path": str(ipc_p), "format": "ipc"})
    df = asyncio.run(src.get_historical("", "2024-01-01", "2024-01-31", eager=True))
    assert len(df) == 3
    assert "valid_time" in df.columns


@pytest.mark.determinism("d1")
@pytest.mark.asyncio
async def test_file_source_realtime_tail_yields_when_file_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_realtime with tail=True yields when CSV chunk is read (mocked file growth)."""
    import io as io_module

    p = tmp_path / "tail.csv"
    p.write_text("date,open,high,low,close,volume\n2024-01-02,100,101,99,100.5,1e6\n")
    initial_len = len(p.read_bytes())
    chunk = b"2024-01-03,101,102,100,101.5,1e6\n"
    sizes_iter = iter([initial_len, initial_len + len(chunk)])

    def fake_getsize(path: str) -> int:
        if path == str(p):
            return next(sizes_iter, initial_len + len(chunk))
        return 0

    monkeypatch.setattr(
        "pysrc.pipeline.stages.market_data.sources.file.os.path.getsize", fake_getsize
    )
    original_open = open

    def fake_open(path: str, mode: str = "", **kwargs: object):
        if path == str(p) and "rb" in mode:
            return io_module.BytesIO(b"x" * initial_len + chunk)
        return original_open(path, mode, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    src = FileSource({"file_path": str(p), "format": "csv", "tail": True})
    count = 0
    async with aclosing(src.get_realtime("", interval=0.01)) as stream:
        async for df in stream:
            count += 1
            assert isinstance(df, pl.DataFrame)
            assert not df.is_empty()
            if count >= 1:
                break
    assert count >= 1


@pytest.mark.determinism("d1")
@pytest.mark.asyncio
async def test_file_source_realtime_tail_filters_by_symbol_when_file_stem_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_realtime with tail=True filters by symbol when _fallback_symbol matches (file named Spysrc.csv)."""
    import io as io_module

    p = tmp_path / "Spysrc.csv"
    p.write_text("date,open,high,low,close,volume\n2024-01-02,100,101,99,100.5,1e6\n")
    initial_len = len(p.read_bytes())
    chunk = b"2024-01-03,101,102,100,101.5,1e6\n"
    sizes_iter = iter([initial_len, initial_len + len(chunk)])

    def fake_getsize(path: str) -> int:
        if path == str(p):
            return next(sizes_iter, initial_len + len(chunk))
        return 0

    monkeypatch.setattr(
        "pysrc.pipeline.stages.market_data.sources.file.os.path.getsize", fake_getsize
    )
    original_open = open

    def fake_open(path: str, mode: str = "", **kwargs: object):
        if path == str(p) and "rb" in mode:
            return io_module.BytesIO(b"x" * initial_len + chunk)
        return original_open(path, mode, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    src = FileSource({"file_path": str(p), "format": "csv", "tail": True})
    count = 0
    async with aclosing(src.get_realtime("SPY", interval=0.01)) as stream:
        async for df in stream:
            count += 1
            assert df.filter(pl.col("symbol") == "SPY").height == df.height
            if count >= 1:
                break
    assert count >= 1


@pytest.mark.determinism("d1")
@pytest.mark.asyncio
async def test_file_source_realtime_tail_two_yields_runs_last_size_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_realtime yields twice; after first yield generator runs last_size=size (line 164)."""
    import io as io_module

    p = tmp_path / "twice.csv"
    p.write_text("date,open,close,volume\n2024-01-02,100,100.5,1e6\n")
    initial_len = len(p.read_bytes())
    c1 = b"2024-01-03,101,101.5,1e6\n"
    c2 = b"2024-01-04,102,102.5,1e6\n"
    sizes = [initial_len, initial_len + len(c1), initial_len + len(c1) + len(c2)]
    sizes_iter = iter(sizes)

    def fake_getsize(path: str) -> int:
        if path == str(p):
            return next(sizes_iter, sizes[-1])
        return 0

    monkeypatch.setattr(
        "pysrc.pipeline.stages.market_data.sources.file.os.path.getsize", fake_getsize
    )
    full_content = b"x" * initial_len + c1 + c2

    def fake_open(path: str, mode: str = "", **kwargs: object):
        if path == str(p) and "rb" in mode:
            return io_module.BytesIO(full_content)
        return open(path, mode, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    src = FileSource({"file_path": str(p), "format": "csv", "tail": True})
    collected: list[pl.DataFrame] = []
    async with aclosing(src.get_realtime("", interval=0.01)) as stream:
        async for df in stream:
            collected.append(df)
            if len(collected) >= 2:
                break
    assert len(collected) == 2
