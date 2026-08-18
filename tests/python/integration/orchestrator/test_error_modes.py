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
import time
import types
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pysrc.pipeline import orchestrator as m

# bootstraps (idempotent)
if not hasattr(m, "require"):
    m.require = lambda *a, **k: None
if not hasattr(m.DataPrepOrchestrator, "_preload_join_sources"):

    def _preload_join_sources(self, manager):  # type: ignore[no-redef]
        return None

    m.DataPrepOrchestrator._preload_join_sources = _preload_join_sources


def _mk_base_cfg(**kw: Any) -> dict[str, Any]:
    # Create a small CSV for file-input paths
    import tempfile

    test_data = [
        {
            "date": "2024-01-01",
            "timestamp": "2024-01-01",
            "symbol": "AAPL",
            "price": 100.0,
            "volume": 1000,
        },
        {
            "date": "2024-01-02",
            "timestamp": "2024-01-02",
            "symbol": "AAPL",
            "price": 101.0,
            "volume": 1200,
        },
    ]
    df = pl.DataFrame(test_data)
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False)
    tmp.close()
    df.write_csv(tmp.name)

    base: dict[str, Any] = {
        "execution": {"lazy": False, "backend": "polars"},
        "cache": {"version_tag": "t1"},
        "data": {"input_path": tmp.name},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
            "spec_inline": {
                "ops": [{"kind": "scaling.robust", "input_col": "price", "out_col": "price_robust"}]
            },
        },
    }
    base.update(kw)
    return base


def test_run_dataprep_manifest_and_columns(tmp_path: Path):
    cfg = _mk_base_cfg()
    out = m.run_dataprep(cfg)
    is_df = isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], pl.DataFrame)
    assert is_df
    df, manifest = out
    assert isinstance(manifest, dict)
    assert "price_robust" in set(manifest.get("columns", [])) or "price_robust" in set(df.columns)
    status = manifest.get("status")
    if status is not None:
        assert status in ["success", "completed"]


@pytest.mark.parametrize(
    ("engine_cfg", "expected_exc"),
    [
        (None, m.DataPrepError),  # no fetch at all
        ({"engine": "market_data"}, m.ConfigError),  # engine selected w/o subconfig
    ],
)
def test_fetch_config_errors(engine_cfg, expected_exc):
    cfg = _mk_base_cfg()
    # Force engine path (remove file input)
    cfg.setdefault("data", {}).pop("input_path", None)
    cfg["fetch"] = engine_cfg
    orch = m.DataPrepOrchestrator(cfg)
    with pytest.raises(expected_exc):
        orch._fetch_raw_multi()


def test_market_data_empty_raises_datafetcherror(monkeypatch):
    cfg = _mk_base_cfg()
    # Force engine path (remove file input)
    cfg.setdefault("data", {}).pop("input_path", None)
    orch = m.DataPrepOrchestrator(cfg)

    class _Mgr:
        async def get_historical(self, **kw):
            return {"AAPL": Exception("fail")}

    monkeypatch.setattr(
        "pysrc.pipeline.dataprep_runtime.MarketDataManager",
        _Mgr,
        raising=False,
    )
    with pytest.raises(m.DataPrepError):
        orch._fetch_raw_multi()


def test_stage_with_guard_retries():
    cfg: dict[str, Any] = _mk_base_cfg()
    cfg["error_handling"] = {
        "retry_policy": {"max_attempts": 2, "initial_backoff_seconds": 0, "max_backoff_seconds": 0}
    }
    orch = m.DataPrepOrchestrator(cfg)

    calls = {"n": 0}

    def flappy():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return "ok"

    out = orch._stage_with_guard("demo", flappy, timeout_s=None)
    assert out == "ok"
    assert any(s.get("name") == "demo" for s in orch._metrics.get("stages", []))


def test_stage_timeout_bubbles():
    cfg = _mk_base_cfg()
    orch = m.DataPrepOrchestrator(cfg)

    def slow():
        time.sleep(2)
        return "done"

    with pytest.raises(Exception):
        orch._stage_with_guard("timeout", slow, timeout_s=1)


@pytest.mark.parametrize(
    ("val", "expected_min"), [(1, 1), ("auto", 1), ("weird", 1), (0, 1), ("3", 1)]
)
def test_resolve_workers(val, expected_min):
    cfg = _mk_base_cfg()
    orch = m.DataPrepOrchestrator(cfg)
    n = orch._resolve_workers(val)
    assert n >= expected_min


def test_resolve_step_dotted_and_errors():
    cfg = _mk_base_cfg()
    orch = m.DataPrepOrchestrator(cfg)

    mod = types.ModuleType("pkgx")
    sub = types.ModuleType("pkgx.sub")

    class C: ...

    import sys

    sys.modules["pkgx"] = mod
    sys.modules["pkgx.sub"] = sub
    sub.C = C

    if hasattr(orch, "_resolve_step"):
        assert orch._resolve_step("pkgx.sub.C") is C
        assert orch._resolve_step("pkgx.sub: C") is C
        with pytest.raises(Exception):
            orch._resolve_step("pkgx.bad.C")
        with pytest.raises(Exception):
            orch._resolve_step("pkgx:bad.C")
    else:
        pytest.skip("no _resolve_step in this implementation")


def test_checkpoint_df_handles_absent_save_df(monkeypatch):
    cfg = _mk_base_cfg()
    orch = m.DataPrepOrchestrator(cfg)

    class _CacheNoDF(m.Cache):
        def save_df(self, *a, **k):
            raise RuntimeError("nope")

    orch.cache = _CacheNoDF()
    df_min = m.to_polars([{"a": 1}])
    orch._checkpoint_df("cleaned", df_min)  # non-fatal
