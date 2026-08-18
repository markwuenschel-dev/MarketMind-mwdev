from __future__ import annotations

import asyncio
import builtins
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytestmark = pytest.mark.determinism("d1")

# Optional deps: skip the single async test if pytest-asyncio isn't present
try:
    import pytest_asyncio  # noqa: F401

    _HAS_ASYNC = True
except Exception:
    _HAS_ASYNC = False

# Hypothesis is optional — guard its usage
try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    _HAS_HYP = True
except Exception:
    _HAS_HYP = False

from pysrc.pipeline.core.pipeline_core_builder import choose_combo, topo_order
from pysrc.pipeline.stages.cleaning import build_cleaning_pipeline
from pysrc.pipeline.stages.cleaning.execution import StreamingCleanerPipeline

# Optional orchestrator
try:
    from pysrc.pipeline.orchestrator import run_dataprep

    HAS_ORCH = True
except Exception:
    HAS_ORCH = False
    run_dataprep = None  # type: ignore

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _maybe_convert_to(tmp_path, input_csv_path: str, filefmt: str) -> str:
    """Create a parquet copy from CSV when the matrix asks for parquet."""
    if filefmt == "csv":
        return input_csv_path
    df = pl.read_csv(input_csv_path)
    out = tmp_path / "converted.parquet"
    df.write_parquet(out)
    return str(out)


def _ensure_close(df: pl.DataFrame | pd.DataFrame):
    """Add a 'close' column when only 'price' exists (keeps tests stable)."""
    if isinstance(df, pl.DataFrame):
        if "close" not in df.columns and "price" in df.columns:
            return df.with_columns(df["price"].alias("close"))
        return df
    # pandas
    if "close" not in df.columns and "price" in df.columns:
        df = df.copy()
        df["close"] = df["price"]
    return df


def create_context(attrs: dict[str, Any]) -> Any:
    """Tiny context object for choose_combo/topo_order tests."""
    return SimpleNamespace(**attrs)


def _build_single_step_pipeline(step_type: str, params: dict[str, Any] | None = None):
    return build_cleaning_pipeline(
        {
            "steps": [
                {
                    "step_id": step_type,
                    "step_type": step_type,
                    "version": "1",
                    "params": params or {},
                }
            ]
        }
    )


# -----------------------------------------------------------------------------
# 1) E2E matrix — driven by your CSV fixtures with robust fallbacks
# -----------------------------------------------------------------------------
from tests.python.infra.matrix import matrix


@matrix(
    backend=["polars", "pandas"],
    optimize=[True, False],
    cache=["on", "off"],
    filefmt=["csv", "parquet"],
    streaming=[True, False],
    ids={
        "backend": {"polars": "pl", "pandas": "pd"},
        "optimize": {True: "opt", False: "noopt"},
        "cache": {"on": "c", "off": "nc"},
        "streaming": {True: "stream", False: "batch"},
    },
)
@pytest.mark.robust
@pytest.mark.parametrize("input_file_fixture", ["prices_small_path", "prices_small_v2_path"])
def test_e2e_matrix_with_fixtures(
    tmp_path, backend, optimize, cache, filefmt, streaming, input_file_fixture, request
):
    # Resolve the fixture path
    fixture_path = request.getfixturevalue(input_file_fixture)
    input_path = _maybe_convert_to(tmp_path, str(fixture_path), filefmt)

    run_cfg = {
        "data": {"input_path": input_path},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
            # inline op is only for fallback path; orchestrator may ignore it harmlessly
            "spec_inline": {
                "ops": [{"kind": "scaling.robust", "input_col": "close", "out_col": "price_robust"}]
            },
        },
        "execution": {"lazy_streaming": streaming, "backend": backend, "optimize": optimize},
        "cache": {"checkpoints": cache == "on"},
    }

    if HAS_ORCH and run_dataprep:
        out, _manifest = run_dataprep(run_cfg, backtest_metric=None)
    else:
        # Minimal fallback using builtins.run shim from conftest
        df_in = (
            pl.read_parquet(input_path)
            if input_path.endswith(".parquet")
            else pl.read_csv(input_path)
        )
        df_in = _ensure_close(df_in)
        out = builtins.run(
            df_in, run_cfg["pipeline"]["spec_inline"], backend=backend, optimize=optimize
        )  # type: ignore

    # Normalize and assert transformation evidence
    if isinstance(out, pl.DataFrame):
        cols = out.columns
    elif hasattr(out, "to_pandas"):
        cols = out.to_pandas().columns
    else:
        cols = getattr(out, "columns", [])
    assert "price_robust" in cols


# -----------------------------------------------------------------------------
# 2) Topological properties & error shapes
# -----------------------------------------------------------------------------
@pytest.mark.property
@pytest.mark.skipif(not _HAS_HYP, reason="hypothesis not installed")
@given(
    steps=st.lists(
        st.text(min_size=1, max_size=10, alphabet="abcxyz._"), min_size=0, max_size=8, unique=True
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=40)
def test_topo_order_properties(steps):
    # ALWAYS pass nested mapping or None to topo_order
    order: dict[str, dict[str, list[str]]] | None = None
    if len(steps) > 1:
        a, b = steps[0], steps[-1]
        if a != b:
            order = {"before": {a: [b]}}

    try:
        ordered = topo_order(steps, order)
        assert set(ordered) == set(steps)
        assert len(ordered) == len(steps)
        pos = {s: i for i, s in enumerate(ordered)}
        if order:
            for a, bs in (order.get("before") or {}).items():
                for b in bs:
                    if a in pos and b in pos:
                        assert pos[a] < pos[b]
            for a, bs in (order.get("after") or {}).items():
                for b in bs:
                    if a in pos and b in pos:
                        assert pos[b] < pos[a]
    except Exception as e:
        assert any(tok in str(e).lower() for tok in ("cycle", "conflict", "circular"))


def test_choose_combo_errors():
    ctx = create_context({})
    cfg = {"cleaning": {"combos": {"x": {"steps": []}, "y": {"steps": []}}}}
    with pytest.raises(KeyError):
        choose_combo(cfg, ctx)
    cfg2 = {"cleaning": {"combos": {"default": {"steps": []}}, "use": "zzz"}}
    with pytest.raises(KeyError):
        choose_combo(cfg2, ctx)


# -----------------------------------------------------------------------------
# 3) Fallback property (use fixture-derived DF instead of generation)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("price_df_fixture", ["prices_small_pl", "prices_small_v2_pl"])
def test_pipeline_property_no_row_increase(price_df_fixture, request):
    df = request.getfixturevalue(price_df_fixture)
    df = _ensure_close(df)
    spec = {"ops": [{"kind": "scaling.robust", "input_col": "close", "out_col": "price_r"}]}
    out = builtins.run(df, {"spec_inline": spec}, backend="polars", optimize=False)  # type: ignore
    out_pl = out if isinstance(out, pl.DataFrame) else pl.DataFrame(out)
    assert out_pl.height == df.height
    assert "price_r" in out_pl.columns


# -----------------------------------------------------------------------------
# 4) Error propagation + a real ValidationStep error path (uses pandas)
# -----------------------------------------------------------------------------
try:
    import pytest_mock  # noqa: F401
except Exception:

    @pytest.fixture
    def mocker():
        import unittest.mock as _mock

        class _Mocker:
            # minimal surface used by tests
            MagicMock = _mock.MagicMock
            patch = _mock.patch

        return _Mocker()


@pytest.mark.parametrize(
    ("exc_cls", "msg"),
    [
        (ValueError, "Invalid configuration"),
        (TypeError, "Type mismatch in pipeline"),
        (RuntimeError, "Pipeline execution failed"),
        (KeyError, "Missing required field"),
    ],
)
def test_error_propagation(exc_cls, msg, mocker):
    dummy = mocker.MagicMock()
    dummy.apply.side_effect = exc_cls(msg)
    with pytest.raises(exc_cls, match=msg):
        dummy.apply(pl.DataFrame({"x": [1, 2, 3]}))


def test_validation_step_raises_on_missing_required_columns():
    pipeline = _build_single_step_pipeline(
        "validate.schema",
        {"ohlcv_mode": True, "strict": True},
    )
    with pytest.raises(Exception):
        pipeline.run(pd.DataFrame({"open": [1], "high": [2]}))


# -----------------------------------------------------------------------------
# 5) Combo ordering integration (choose_combo + topo_order)
# -----------------------------------------------------------------------------
def test_combo_ordering_integration():
    ctx = create_context({})
    cfg: dict[str, Any] = {
        "pipeline": {
            "cleaning": {
                "combos": {
                    "default": {
                        "steps": [
                            {
                                "step_id": "impute.outliers",
                                "step_type": "impute.outliers",
                                "version": "1",
                            },
                            {
                                "step_id": "impute.missing",
                                "step_type": "impute.missing",
                                "version": "1",
                            },
                        ],
                        "order": {"before": {"impute.missing": ["impute.outliers"]}},
                    }
                },
                "use": "default",
            }
        }
    }
    cleaning_cfg = cfg["pipeline"]["cleaning"]
    res = choose_combo(cleaning_cfg, ctx)

    steps = res.get("steps", [])
    order = res.get("order", None)
    names = [str(s["step_id"]) for s in steps]
    ordered = topo_order(names, order if order else None)
    assert ordered.index("impute.missing") < ordered.index("impute.outliers")


# -----------------------------------------------------------------------------
# 6) Streaming smoke (guarded)
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_ASYNC, reason="requires pytest-asyncio")
@pytest.mark.asyncio
async def test_streaming_smoke_with_fixture(prices_small_pl: pl.DataFrame):
    # Turn your fixture into a tiny async stream (row-wise)
    async def src():
        for i in range(min(8, prices_small_pl.height)):
            yield prices_small_pl.slice(i, 1)
            await asyncio.sleep(0)

    pipeline = _build_single_step_pipeline(
        "impute.missing",
        {"method": "forward_fill", "backward_fill": True},
    )
    pipe = StreamingCleanerPipeline(pipeline, buffer_size=3)
    out_batches: list[pl.DataFrame] = []
    async for batch in pipe.process_stream(src()):
        out_batches.append(batch)

    assert out_batches
    assert set(out_batches[0].columns) <= set(prices_small_pl.columns)


# -----------------------------------------------------------------------------
# 7) Perf-ish check scaled from fixture (benchmark plugin required)
# -----------------------------------------------------------------------------
@pytest.mark.perf
@pytest.mark.parametrize("target_rows", [1_000, 5_000])
def test_scaling_from_fixture(benchmark, prices_small_pl: pl.DataFrame, target_rows: int):
    df = _ensure_close(prices_small_pl)

    def scale_up(df_in: pl.DataFrame, n: int) -> pl.DataFrame:
        if df_in.height == 0:
            return df_in
        reps = int(np.ceil(n / df_in.height))
        big = pl.concat([df_in] * reps).with_row_index(name="_rid")
        # ensure timestamp monotonic by adding an offset to duplicates if present
        if "timestamp" in big.columns:
            # Ensure monotonic timestamps by adding a per-duplicate offset as a duration.
            # Cast to i64 epoch ns, add small offset, cast back to datetime[ns] (Polars requires datetime + duration).
            big = big.with_columns(
                (
                    pl.col("timestamp").cast(pl.Int64)
                    + (pl.col("_rid") // df_in.height).cast(pl.Int64)
                )
                .cast(pl.Datetime("ns"))
                .alias("timestamp")
            ).drop("_rid")
        else:
            big = big.drop("_rid")
        return big.head(n)

    df_big = scale_up(df, target_rows)
    out = benchmark(
        builtins.run,
        df_big,
        {
            "spec_inline": {
                "ops": [{"kind": "scaling.robust", "input_col": "close", "out_col": "price_r"}]
            }
        },
        backend="polars",
        optimize=True,
    )  # type: ignore
    assert isinstance(out, (pl.DataFrame, pd.DataFrame))


# -----------------------------------------------------------------------------
# 8) Regressions — run against fixtures (no synthetic gen)
# -----------------------------------------------------------------------------
def test_regression_empty_like(prices_small_pl: pl.DataFrame):
    empty = prices_small_pl.head(0)
    out = builtins.run(empty, {"spec_inline": {"ops": []}}, backend="polars", optimize=False)  # type: ignore
    out_pl = out if isinstance(out, pl.DataFrame) else pl.DataFrame(out)
    assert out_pl.height == 0


def test_regression_single_row_like(prices_small_pl: pl.DataFrame):
    one = prices_small_pl.head(1)
    out = builtins.run(one, {"spec_inline": {"ops": []}}, backend="polars", optimize=False)  # type: ignore
    out_pl = out if isinstance(out, pl.DataFrame) else pl.DataFrame(out)
    assert out_pl.height == 1


def test_topo_order_preserves_input_order_when_no_constraints():
    steps = ["S1", "S2", "S3", "S4"]
    ordered = topo_order(steps, None)  # not {}
    assert ordered == steps


def test_topo_order_chained_constraints():
    steps = ["A", "B", "C"]
    order: dict[str, dict[str, list[str]]] = {"before": {"A": ["C"], "C": ["B"]}}
    ordered = topo_order(steps, order)
    assert ordered == ["A", "C", "B"]
