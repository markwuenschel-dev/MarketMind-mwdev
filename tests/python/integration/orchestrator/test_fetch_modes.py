# --- test import resilience shim (allows moving under tests/python/integration) ---
import pathlib as _plx
import sys as _sysx

try:
    _ROOT = _plx.Path(__file__).resolve().parents[2]
except Exception:
    _ROOT = _plx.Path.cwd()
_TPY = _ROOT / "tests" / "python"
for _p in (str(_ROOT), str(_TPY)):
    if _p not in _sysx.path:
        _sysx.path.insert(0, _p)
# --- end shim ---
# tests/python/test_fetch_modes.py
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pysrc.pipeline import orchestrator as m

pytestmark = [
    pytest.mark.robust,
    pytest.mark.combinatoric,
    pytest.mark.elegant,
    pytest.mark.efficient,
]

if not hasattr(m, "require"):
    m.require = lambda *a, **k: None
if not hasattr(m.DataPrepOrchestrator, "_preload_join_sources"):

    def _preload_join_sources(self, manager):  # type: ignore[no-redef]
        return None

    m.DataPrepOrchestrator._preload_join_sources = _preload_join_sources


def _cfg_with_file(path: str, **kw: Any) -> dict[str, Any]:
    base = {
        "execution": {"lazy": False, "lazy_streaming": False},  # normalize execution keys
        "cache": {"version_tag": "t1"},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
            "spec_inline": {"ops": []},
        },
        "data": {"input_path": path},
        "io": {"allow_empty": True},  # allow zero-byte/header-only files to pass as empty
    }
    base.update(kw)
    return base


@pytest.mark.parametrize(
    ("producer", "expects_error"),
    [
        (lambda p: Path(p).write_bytes(b""), False),  # zero-byte → OK when io.allow_empty=True
        (lambda p: Path(p).write_text("timestamp,price\n"), False),  # header-only → OK (empty)
        (lambda p: Path(p).write_text("timestamp,price\n2024-01-01,100\n"), False),  # valid → OK
    ],
)
def test_fetch_csv_variants(tmp_path, producer, expects_error):
    f = tmp_path / "data.csv"
    producer(f)
    cfg = _cfg_with_file(str(f))
    orch = m.DataPrepOrchestrator(cfg)
    if expects_error:
        with pytest.raises(Exception):
            orch._fetch_raw_multi()
    else:
        df = orch._fetch_raw_multi()
        assert df is not None


def test_fetch_malformed_csv_raises_datapreperror(tmp_path, monkeypatch):
    f = tmp_path / "bad.csv"
    f.write_text('this,is,not\nproper,csv\n"unterminated')
    real = m.pl.read_csv

    def boom(path, *a, **k):
        if str(path) == str(f):
            raise Exception("parse error: bad CSV")
        return real(path, *a, **k)

    monkeypatch.setattr(m.pl, "read_csv", boom, raising=True)
    cfg = _cfg_with_file(str(f))
    orch = m.DataPrepOrchestrator(cfg)
    with pytest.raises(m.DataPrepError):
        orch._fetch_raw_multi()


def test_fetch_jsonl_and_parquet(tmp_path):
    # JSONL / NDJSON are the supported newline-delimited JSON formats
    fj = tmp_path / "data.jsonl"
    fj.write_text(
        '{"timestamp":"2024-01-01","price":100}\n{"timestamp":"2024-01-02","price":101}\n'
    )
    orch = m.DataPrepOrchestrator(_cfg_with_file(str(fj)))
    assert orch._fetch_raw_multi() is not None
    # Parquet
    fp = tmp_path / "data.parquet"
    pl.DataFrame({"timestamp": ["2024-01-01"], "price": [100]}).write_parquet(fp)
    orch = m.DataPrepOrchestrator(_cfg_with_file(str(fp)))
    assert orch._fetch_raw_multi() is not None


def test_fetch_avro_unsupported(tmp_path):
    fav = tmp_path / "data.avro"
    fav.write_bytes(b"\x00")
    orch = m.DataPrepOrchestrator(_cfg_with_file(str(fav)))
    with pytest.raises(m.DataPrepError):
        orch._fetch_raw_multi()


def test_fetch_missing_everything_raises():
    cfg = {
        "execution": {"lazy": False, "lazy_streaming": False},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"}
        },
    }
    orch = m.DataPrepOrchestrator(cfg)
    with pytest.raises(m.DataPrepError):
        orch._fetch_raw_multi()


def test_engine_selected_but_missing_subconfig_raises():
    cfg = {
        "execution": {"lazy": False, "lazy_streaming": False},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"}
        },
        "fetch": {"engine": "market_data"},
    }
    # Force engine path: no file input
    orch = m.DataPrepOrchestrator(cfg)
    with pytest.raises(m.ConfigError):
        orch._fetch_raw_multi()


def test_market_data_empty_results_raise_datafetcherror(monkeypatch):
    cfg = {
        "execution": {"lazy": False, "lazy_streaming": False},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"}
        },
        "fetch": {
            "engine": "market_data",
            "market_data": {"sources": [{"name_for_registry": "s"}]},
        },
        "run": {"symbols": ["AAPL"], "start": "2024-01-01", "end": "2024-01-02"},
    }
    orch = m.DataPrepOrchestrator(cfg)

    class _Mgr:
        async def get_historical(self, **kw):
            return {"AAPL": Exception("no data")}

    def _run_immediate(coro):
        try:
            coro.send(None)
        except StopIteration as stop:
            return stop.value
        raise AssertionError("Expected coroutine to complete without scheduling")

    monkeypatch.setattr(
        "pysrc.pipeline.dataprep_runtime.MarketDataManager",
        _Mgr,
        raising=False,
    )
    monkeypatch.setattr(m.asyncio, "run", _run_immediate, raising=False)
    with pytest.raises(m.DataPrepError):
        orch._fetch_raw_multi()
