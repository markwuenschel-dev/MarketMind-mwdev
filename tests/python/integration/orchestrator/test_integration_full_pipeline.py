# tests/integration/test_integration_full_pipeline.py
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from allpairspy import AllPairs
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from pysrc.pipeline import orchestrator as m

# ============================================================================
# Fixtures and Helpers
# ============================================================================


def _minimal_cfg(input_path: str, **overrides) -> dict[str, Any]:
    base = {
        "data": {"input_path": input_path},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"}
        },
        "execution": {"lazy": False, "lazy_streaming": False},
        "cache": {"checkpoints": False, "version_tag": "test"},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base:
            base[k].update(v)
        else:
            base[k] = v
    return base


def _csv_with_data(tmp_path: Path, name: str = "data.csv", rows: int = 2) -> Path:
    p = tmp_path / name
    lines = ["timestamp,symbol,price\n"]
    for i in range(rows):
        lines.append(f"2024-01-{i + 1:02d},TEST,{100 + i}\n")
    p.write_text("".join(lines), encoding="utf-8")
    return p


# ============================================================================
# Unit Tests: run_dataprep with spec_inline (happy path)
# ============================================================================


def test_run_dataprep_csv_spec_inline_returns_tuple(tmp_path):
    # Inline spec path should return (df, manifest) tuple
    csv = _csv_with_data(tmp_path)
    cfg = _minimal_cfg(
        str(csv),
        pipeline={
            "spec_inline": {
                "ops": [{"kind": "scaling.robust", "input_col": "price", "out_col": "price_robust"}]
            },
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    result = m.run_dataprep(cfg, backtest_metric=None)
    assert isinstance(result, tuple), "run_dataprep with spec_inline must return tuple"
    df, manifest = result
    assert isinstance(df, pl.DataFrame), "First element must be DataFrame"
    assert isinstance(manifest, dict), "Second element must be manifest dict"


def test_run_dataprep_spec_inline_creates_transformed_column(tmp_path):
    # Verify transformation actually runs
    csv = _csv_with_data(tmp_path, rows=3)
    cfg = _minimal_cfg(
        str(csv),
        pipeline={
            "spec_inline": {
                "ops": [{"kind": "scaling.robust", "input_col": "price", "out_col": "price_scaled"}]
            },
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, manifest = m.run_dataprep(cfg)
    assert "price_scaled" in df.columns, "Transformed column must exist"
    assert len(df) == 3, "Row count preserved"
    assert manifest["status"] == "success"


def test_run_dataprep_spec_inline_preserves_original_columns(tmp_path):
    # Non-destructive transformation
    csv = _csv_with_data(tmp_path)
    cfg = _minimal_cfg(
        str(csv),
        pipeline={
            "spec_inline": {
                "ops": [{"kind": "scaling.robust", "input_col": "price", "out_col": "new_col"}]
            },
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, _ = m.run_dataprep(cfg)
    assert set(df.columns) >= {"timestamp", "symbol", "price", "new_col"}


def test_run_dataprep_empty_ops_passthrough(tmp_path):
    # Empty ops list = identity transform
    csv = _csv_with_data(tmp_path, rows=5)
    cfg = _minimal_cfg(
        str(csv),
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, manifest = m.run_dataprep(cfg)
    assert len(df) == 5
    assert set(df.columns) == {"timestamp", "symbol", "price"}
    assert manifest["status"] == "success"


# ============================================================================
# Unit Tests: orchestrator.run() returns manifest dict (no spec_inline)
# ============================================================================


def test_orchestrator_run_no_preprocessing_returns_manifest_dict(tmp_path):
    # When no spec_inline and no preset/grid, run() returns manifest only
    csv = _csv_with_data(tmp_path)
    cfg = _minimal_cfg(str(csv))
    orch = m.DataPrepOrchestrator(cfg)
    manifest = orch.run()
    assert isinstance(manifest, dict), "orch.run() without preprocessing returns manifest"
    assert manifest["status"] == "success"


def test_orchestrator_run_manifest_includes_hashes(tmp_path):
    # Provenance: hashes tracked in manifest
    csv = _csv_with_data(tmp_path)
    cfg = _minimal_cfg(str(csv))
    orch = m.DataPrepOrchestrator(cfg)
    manifest = orch.run()
    assert "hashes" in manifest
    hashes = manifest["hashes"]
    assert "raw_hash" in hashes
    assert "clean_hash" in hashes
    assert "processed_hash" in hashes


def test_orchestrator_run_manifest_no_preset_hashes_when_unused(tmp_path):
    # When preset/grid not configured, those hashes should be absent
    csv = _csv_with_data(tmp_path)
    cfg = _minimal_cfg(str(csv))
    orch = m.DataPrepOrchestrator(cfg)
    manifest = orch.run()
    hashes = manifest.get("hashes", {})
    assert "preset_hash" not in hashes, "No preset used, hash should be absent"
    assert "grid_hash" not in hashes, "No grid used, hash should be absent"


def test_orchestrator_run_manifest_includes_run_id(tmp_path):
    # Observability: run_id always present
    csv = _csv_with_data(tmp_path)
    cfg = _minimal_cfg(str(csv))
    orch = m.DataPrepOrchestrator(cfg)
    manifest = orch.run()
    assert "run_id" in manifest
    assert isinstance(manifest["run_id"], str)
    assert len(manifest["run_id"]) > 0


# ============================================================================
# Unit Tests: Input formats (CSV, JSONL, Parquet)
# ============================================================================


def test_run_dataprep_jsonl_input_format(tmp_path):
    # JSONL input through full pipeline
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        '{"timestamp":"2024-01-01","symbol":"TEST","price":100}\n'
        '{"timestamp":"2024-01-02","symbol":"TEST","price":101}\n',
        encoding="utf-8",
    )
    cfg = _minimal_cfg(
        str(jsonl),
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, manifest = m.run_dataprep(cfg)
    assert len(df) == 2
    assert manifest["status"] == "success"


def test_run_dataprep_parquet_input_format(tmp_path):
    # Parquet input through full pipeline
    pq = tmp_path / "data.parquet"
    df_orig = pl.DataFrame(
        {"timestamp": ["2024-01-01", "2024-01-02"], "symbol": ["TEST", "TEST"], "price": [100, 101]}
    )
    df_orig.write_parquet(pq)
    cfg = _minimal_cfg(
        str(pq),
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, manifest = m.run_dataprep(cfg)
    assert len(df) == 2
    assert manifest["status"] == "success"


def test_run_dataprep_dataframe_input_bypasses_fetch(tmp_path):
    # In-memory DataFrame input (no file I/O)
    df_input = pl.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "symbol": ["AAPL"] * 3,
            "price": [100, 101, 102],
        }
    )
    cfg = {
        "data": {"input_df": df_input},
        "pipeline": {
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
        "execution": {"lazy": False, "lazy_streaming": False},
        "cache": {"checkpoints": False},
    }
    df, manifest = m.run_dataprep(cfg)
    assert len(df) == 3
    assert "AAPL" in df["symbol"].unique().to_list()


# ============================================================================
# Edge Cases: Empty data, missing columns, malformed input
# ============================================================================


def test_run_dataprep_empty_csv_allow_empty_succeeds(tmp_path):
    # Empty CSV with io.allow_empty=True should succeed
    csv = tmp_path / "empty.csv"
    csv.write_text("timestamp,price\n", encoding="utf-8")
    cfg = _minimal_cfg(
        str(csv),
        io={"allow_empty": True},
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, manifest = m.run_dataprep(cfg)
    assert len(df) == 0, "Empty input should produce empty output"
    assert manifest["status"] == "success"


def test_run_dataprep_missing_timestamp_raises_datapreperror(tmp_path):
    # CSV without timestamp-like column should fail
    csv = tmp_path / "no_ts.csv"
    csv.write_text("x,y\n1,2\n", encoding="utf-8")
    cfg = _minimal_cfg(
        str(csv),
        io={"allow_empty": False},
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    with pytest.raises(m.DataPrepError, match="timestamp"):
        m.run_dataprep(cfg)


def test_run_dataprep_unsupported_file_extension_raises(tmp_path):
    # .avro or unknown extensions should raise
    avro = tmp_path / "data.avro"
    avro.write_bytes(b"Obj\x01\x04")
    cfg = _minimal_cfg(
        str(avro),
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    with pytest.raises(m.DataPrepError, match="unsupported"):
        m.run_dataprep(cfg)


def test_run_dataprep_missing_input_path_raises_configerror(tmp_path):
    # No input_path and no engine configured = ConfigError
    cfg = {
        "pipeline": {
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
        "execution": {"lazy": False},
        "cache": {"checkpoints": False},
    }
    with pytest.raises((m.ConfigError, m.DataPrepError)):
        m.run_dataprep(cfg)


# ============================================================================
# Pairwise Combinatorial Tests: (backend × lazy × checkpoints)
# ============================================================================


@pytest.mark.parametrize(
    ("backend", "lazy", "checkpoints"),
    list(
        AllPairs(
            [
                ["auto", "cpu", "polars"],  # backend
                [False, True],  # lazy
                [False, True],  # checkpoints
            ]
        )
    ),
)
def test_run_dataprep_pairwise_execution_configs(tmp_path, backend, lazy, checkpoints):
    # Pairwise coverage of execution flags
    csv = _csv_with_data(tmp_path, rows=2)
    cfg = _minimal_cfg(
        str(csv),
        execution={"backend": backend, "lazy": lazy, "lazy_streaming": False},
        cache={"checkpoints": checkpoints, "version_tag": "pw"},
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    result = m.run_dataprep(cfg)
    assert isinstance(result, tuple), f"Failed with {backend=}, {lazy=}, {checkpoints=}"
    df, manifest = result
    assert len(df) == 2
    assert manifest["status"] == "success"


# ============================================================================
# Property-Based Tests: Invariants under diverse inputs
# ============================================================================


@given(
    rows=st.integers(min_value=1, max_value=50), price_base=st.integers(min_value=10, max_value=200)
)
@seed(12345)
@settings(deadline=None, max_examples=20)
def test_run_dataprep_preserves_row_count(tmp_path_factory, rows, price_base):
    # Property: non-destructive ops preserve row count
    tmp = tmp_path_factory.mktemp(f"prop_{rows}")
    csv = tmp / "data.csv"
    lines = ["timestamp,price\n"]
    for i in range(rows):
        lines.append(f"2024-01-{(i % 28) + 1:02d},{price_base + i}\n")
    csv.write_text("".join(lines), encoding="utf-8")

    cfg = _minimal_cfg(
        str(csv),
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, _ = m.run_dataprep(cfg)
    assert len(df) == rows, f"Row count mismatch: expected {rows}, got {len(df)}"


@given(
    col_name=st.text(
        min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))
    )
)
@seed(12345)
@settings(deadline=None, max_examples=15)
def test_run_dataprep_output_includes_input_columns(tmp_path_factory, col_name):
    # Property: original columns always present after additive transforms
    tmp = tmp_path_factory.mktemp(f"col_{hash(col_name) % 10000}")
    csv = tmp / "data.csv"
    csv.write_text(f"timestamp,{col_name}\n2024-01-01,100\n", encoding="utf-8")

    cfg = _minimal_cfg(
        str(csv),
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, _ = m.run_dataprep(cfg)
    assert col_name in df.columns, f"Input column '{col_name}' missing from output"


# ============================================================================
# Error Conditions: ConfigError, DataPrepError
# ============================================================================


def test_configerror_raised_when_market_data_engine_without_subconfig():
    # Engine selected but market_data config missing
    cfg = {
        "fetch": {"engine": "market_data"},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"}
        },
        "execution": {"lazy": False},
        "cache": {"checkpoints": False},
    }
    orch = m.DataPrepOrchestrator(cfg)
    with pytest.raises(m.ConfigError, match="parameters"):
        orch._fetch_raw_multi()


def test_datapreperror_raised_when_market_data_returns_exception(monkeypatch):
    # Market data fetch returns Exception instead of DataFrame
    cfg = {
        "fetch": {
            "engine": "market_data",
            "market_data": {"sources": [{"name_for_registry": "test_source"}]},
        },
        "run": {"symbols": ["AAPL"], "start": "2024-01-01", "end": "2024-01-02"},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"}
        },
        "execution": {"lazy": False},
        "cache": {"checkpoints": False},
    }

    class _MockManager:
        async def get_historical(self, **kwargs):
            return {"AAPL": Exception("fetch failed")}

    monkeypatch.setattr(
        "pysrc.pipeline.dataprep_runtime.MarketDataManager",
        _MockManager,
        raising=False,
    )
    orch = m.DataPrepOrchestrator(cfg)
    with pytest.raises(m.DataPrepError, match="fetch failed"):
        orch._fetch_raw_multi()


def test_datapreperror_raised_when_no_market_data_returned(monkeypatch):
    # Empty result from market data engine
    cfg = {
        "fetch": {"engine": "market_data", "market_data": {"sources": []}},
        "run": {"symbols": ["AAPL"], "start": "2024-01-01", "end": "2024-01-02"},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"}
        },
        "execution": {"lazy": False},
        "cache": {"checkpoints": False},
    }

    class _MockManager:
        async def get_historical(self, **kwargs):
            return {}

    monkeypatch.setattr(
        "pysrc.pipeline.dataprep_runtime.MarketDataManager",
        _MockManager,
        raising=False,
    )
    orch = m.DataPrepOrchestrator(cfg)
    with pytest.raises(m.DataPrepError, match="Market data fetch returned no data for AAPL"):
        orch._fetch_raw_multi()


# ============================================================================
# Manifest Verification: Columns, status, timestamps
# ============================================================================


def test_manifest_columns_field_matches_output_dataframe(tmp_path):
    # Manifest.columns should list actual output columns
    csv = _csv_with_data(tmp_path)
    cfg = _minimal_cfg(
        str(csv),
        pipeline={
            "spec_inline": {
                "ops": [{"kind": "scaling.robust", "input_col": "price", "out_col": "scaled"}]
            },
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, manifest = m.run_dataprep(cfg)
    manifest_cols = set(manifest.get("columns", []))
    df_cols = set(df.columns)
    assert manifest_cols == df_cols, "Manifest columns must match DataFrame columns"


def test_manifest_includes_start_and_end_time(tmp_path):
    # Observability: timing metadata present
    csv = _csv_with_data(tmp_path)
    cfg = _minimal_cfg(str(csv))
    orch = m.DataPrepOrchestrator(cfg)
    manifest = orch.run()
    assert "start_time" in manifest
    assert "end_time" in manifest
    # Basic format check (ISO-like timestamp)
    assert "T" in manifest["start_time"]
    assert "T" in manifest["end_time"]


def test_manifest_status_success_on_happy_path(tmp_path):
    # Status field = "success" when no errors
    csv = _csv_with_data(tmp_path)
    cfg = _minimal_cfg(
        str(csv),
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, manifest = m.run_dataprep(cfg)
    assert manifest["status"] in ("success", "completed")


# ============================================================================
# Cache Interaction: Checkpointing (smoke tests, not deep cache logic)
# ============================================================================


def test_cache_checkpoints_enabled_does_not_error(tmp_path):
    # When checkpoints=True, pipeline should not crash (cache may be no-op)
    csv = _csv_with_data(tmp_path)
    cfg = _minimal_cfg(
        str(csv),
        cache={"checkpoints": True, "version_tag": "ckpt"},
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, manifest = m.run_dataprep(cfg)
    assert manifest["status"] == "success"


def test_cache_checkpoints_disabled_completes_successfully(tmp_path):
    # When checkpoints=False, no cache persistence attempted
    csv = _csv_with_data(tmp_path)
    cfg = _minimal_cfg(
        str(csv),
        cache={"checkpoints": False},
        pipeline={
            "spec_inline": {"ops": []},
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
    )
    df, manifest = m.run_dataprep(cfg)
    assert manifest["status"] == "success"
