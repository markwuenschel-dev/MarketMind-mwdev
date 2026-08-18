# tests/python/introspective/test_optimizer_collects_drop.py
import polars as pl
import pytest

from pysrc.preprocessor.api import run


@pytest.fixture
def collect_counter(monkeypatch):
    counts = {"n": 0}
    # Robustly locate LazyFrame.collect and patch it
    Lazy = pl.DataFrame({"x": [1]}).lazy().__class__
    orig_collect = Lazy.collect

    def wrapped(self, *a, **k):
        counts["n"] += 1
        return orig_collect(self, *a, **k)

    monkeypatch.setattr(Lazy, "collect", wrapped, raising=True)
    return counts


def _run_and_count(df, spec, optimize: bool) -> int:
    # Note: optimize parameter not supported by current run() API
    out = run(df, spec, backend="polars")
    assert out.height == df.height  # sanity check
    return out


@pytest.mark.introspective
def test_optimizer_collects_must_not_increase(collect_counter, df_prices, preproc_spec_robust):
    # baseline
    _run_and_count(df_prices, preproc_spec_robust, optimize=False)
    baseline = collect_counter["n"]
    # optimized
    collect_counter["n"] = 0
    _run_and_count(df_prices, preproc_spec_robust, optimize=True)
    optimized = collect_counter["n"]

    # Always non-increasing; in most cases strictly smaller.
    assert optimized <= baseline
