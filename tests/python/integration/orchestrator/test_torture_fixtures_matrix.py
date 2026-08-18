# tests/python/integration/orchestrator/test_torture_fixtures_matrix.py
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.errors import EmptyDataError

# Register the plugin
pytest_plugins = ("tests.python.plugins.torture_plugin",)

# ---------------- backend loaders & adapters ----------------


def _load_input_df(csv_path: Path, backend: str, **read_kwargs):
    if backend == "polars":
        pl = pytest.importorskip("polars")
        # Forward only relevant kwargs for polars
        kw = {k: v for k, v in read_kwargs.items() if k in {"sep", "has_header"}}
        return pl.read_csv(str(csv_path), **kw)
    return pd.read_csv(csv_path, **read_kwargs)


def _is_frame_like(obj) -> bool:
    if isinstance(obj, pd.DataFrame):
        return True
    name = getattr(obj, "__class__", None).__name__
    return name in {"DataFrame", "LazyFrame"}


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "collect"):
        obj = obj.collect()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


# ---------------- orchestrator or fallback ----------------


def _maybe_import_orchestrator():
    # Plug your real orchestrator here by listing (module, function)
    candidates = [
        ("pysrc.pipeline.orchestrator", "run_dataprep"),
    ]
    for mod, fn in candidates:
        try:
            m = __import__(mod, fromlist=[fn])
            return "orchestrator", getattr(m, fn)
        except (ImportError, AttributeError):
            continue

    # Fallback: identity runner
    def _fallback_runner(df_in, run_cfg, backend: str, optimize: bool):
        return df_in, {
            "backend": backend,
            "optimize": optimize,
            "ops": run_cfg.get("pipeline", {}).get("spec_inline", {}).get("ops", []),
        }

    return "fallback", _fallback_runner


# ---------------- utility infra ----------------

_NUMERIC_COL_CANDIDATES = ("close", "price", "last", "value", "adj_close", "close_price")


def _discover_input_col(csv_path: Path, **read_kwargs) -> str | None:
    if csv_path.is_dir():
        return None
    try:
        sample = pd.read_csv(csv_path, nrows=50, engine="python", **read_kwargs)
    except EmptyDataError:
        return None
    except (pd.errors.ParserError, OSError, ValueError):
        try:
            sample = pd.read_csv(csv_path, nrows=50, engine="python")
        except (pd.errors.ParserError, OSError, ValueError):
            return None
    for c in _NUMERIC_COL_CANDIDATES:
        if c in sample.columns and pd.api.types.is_numeric_dtype(sample[c]):
            return c
    for c in sample.columns:
        if pd.api.types.is_numeric_dtype(sample[c]):
            return c
    return None


def _sha256_file(path: Path | None) -> str:
    if path is None or path.is_dir():
        return "<DIR>"
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_to_tmp(tmp_root: Path, src: Path) -> Path:
    dst = tmp_root / src.name
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst


# ---------------- the main combinatoric test ----------------


@pytest.mark.robust
@pytest.mark.combinatoric
def test_torture_matrix(tmp_path_factory, torture_case, backend, optimize):
    """
    Dynamic, robust, combinatoric test over discovered fixtures × backends × optimize flags.
    """
    # Resolve data dir from plugin hook
    # Path derivation mirrors plugin's auto path
    start = Path(__file__)
    from tests.python.plugins.torture_plugin import _auto_data_dir

    data_dir = _auto_data_dir(start)

    # Unpack case
    rel = torture_case.relpath
    main_rel, side_rel = (rel + (None,))[:2]
    src_main = data_dir / main_rel
    src_side = data_dir / side_rel if side_rel else None

    # Streaming is handled elsewhere
    if torture_case.expect == "stream_ok" or src_main.is_dir():
        pytest.skip("Streaming validated in dedicated async test")

    assert src_main.exists(), f"Missing fixture: {src_main}"
    if src_side:
        assert src_side.exists(), f"Missing sidecar: {src_side}"

    # Immutability snapshot
    main_hash_before = _sha256_file(src_main)
    side_hash_before = _sha256_file(src_side) if src_side else None

    # Isolated working copies
    tmp_root = tmp_path_factory.mktemp(f"case-{torture_case.id}")
    dst_main = _copy_to_tmp(tmp_root, src_main)
    dst_side = _copy_to_tmp(tmp_root, src_side) if src_side else None

    read_kwargs = dict(torture_case.read_kwargs)
    fmt = torture_case.fmt
    expect = torture_case.expect

    # Fast path: empty/header-only
    if expect == "empty_ok":
        try:
            df = pd.read_csv(dst_main, **read_kwargs)
            assert df.shape[0] == 0, "Expected empty frame"
        except EmptyDataError:
            pass
        assert _sha256_file(src_main) == main_hash_before
        if src_side:
            assert _sha256_file(src_side) == side_hash_before
        return

    # Orchestrator selection + pipeline_config
    mode, run_fn = _maybe_import_orchestrator()

    input_col = _discover_input_col(dst_main, **read_kwargs)
    ops = (
        [{"kind": "scaling.robust", "input_col": input_col, "out_col": "price_robust"}]
        if input_col
        else []
    )

    run_cfg = {
        "data": {
            "input_path": str(dst_main),
            "sidecar": str(dst_side) if dst_side else None,
            "read_kwargs": read_kwargs,
            "format": fmt,
        },
        "pipeline": {"spec_inline": {"ops": ops}},
        "execution": {"backend": backend, "optimize": optimize, "lazy_streaming": False},
    }

    # Expected error path
    if expect == "should_raise":
        with pytest.raises(torture_case.raises or (BaseException,), match=torture_case.match):
            if mode == "orchestrator":
                _out, _manifest = run_fn(run_cfg, backtest_metric=None)
            else:
                _ = _load_input_df(dst_main, backend, **read_kwargs)
                _out, _manifest = run_fn(_, run_cfg, backend=backend, optimize=optimize)

    else:
        # Normal path
        if mode == "orchestrator":
            out, manifest = run_fn(run_cfg, backtest_metric=None)
        else:
            df_in = _load_input_df(dst_main, backend, **read_kwargs)
            out, manifest = run_fn(df_in, run_cfg, backend=backend, optimize=optimize)

        assert _is_frame_like(out), f"Expected frame-like output for {torture_case.id}"
        out_pd = _to_pandas(out)
        assert out_pd.shape[1] > 0, "No columns in result"

        # Checks
        checks = set(torture_case.checks or ())
        if "sorted_ts" in checks and "timestamp" in out_pd.columns:
            assert out_pd["timestamp"].is_monotonic_increasing
        if "dedup_ts_symbol" in checks and {"timestamp", "symbol"} <= set(out_pd.columns):
            assert not out_pd.duplicated(["timestamp", "symbol"]).any()
        if "headers_trimmed" in checks:
            assert all(c.strip() == c for c in out_pd.columns)
        if "volume_numeric" in checks and "volume" in out_pd.columns:
            assert pd.api.types.is_numeric_dtype(out_pd["volume"])
        if "numeric_parse_price" in checks and "price" in out_pd.columns:
            assert pd.api.types.is_numeric_dtype(out_pd["price"])
        if "finite_prices" in checks and "close" in out_pd.columns:
            assert np.isfinite(out_pd["close"]).all()
        if "no_negative_low" in checks and "low" in out_pd.columns:
            assert (out_pd["low"] >= 0).all()
        if "tz_normalized_utc" in checks and "timestamp" in out_pd.columns:
            ts = out_pd["timestamp"]
            assert getattr(getattr(ts.dtype, "tz", None), "key", None) in (None, "UTC")
        if "corp_actions_flags_or_adjust" in checks:
            assert any(
                k in out_pd.columns for k in ["adj_close", "corp_action_flag", "split_factor"]
            )
        if "symbol_continuity" in checks and "symbol" in out_pd.columns:
            assert out_pd["symbol"].isin(["FB", "META"]).any()
        if "staleness_flag_or_metric" in checks:
            assert any(k in out_pd.columns for k in ["stale_flag", "freshness_score"])

        if expect == "crossfile_consistency":
            assert dst_side
            assert Path(dst_side).exists()
            meta_df = pd.read_csv(dst_side)
            sym_col = "symbol" if "symbol" in out_pd.columns else None
            if sym_col and "symbol" in meta_df.columns:
                overlap = set(out_pd[sym_col].unique()) & set(meta_df["symbol"].unique())
                assert overlap, "symbol universes do not overlap"

    # Originals untouched
    assert _sha256_file(src_main) == main_hash_before
    if src_side:
        assert _sha256_file(src_side) == side_hash_before
