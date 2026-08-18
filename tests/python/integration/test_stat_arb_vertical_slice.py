from __future__ import annotations

import contextlib
from datetime import date

import pandas as pd
import polars as pl
import pytest

import pysrc.strategies.stat_arb  # noqa: F401 - ensure package __init__ loaded for coverage
from pysrc.core.errors import DataFetchError
from pysrc.strategies.stat_arb.config import PAIRS_DEFAULT
from pysrc.strategies.stat_arb.entry import StatArbRunConfig, run_stat_arb_pairs


def _close_yfinance_peewee_dbs() -> None:
    """yfinance opens peewee/sqlite caches; close before pytest reports ResourceWarning."""
    try:
        import yfinance.cache as yfc
    except ImportError:
        return
    for name in ("_TzDBManager", "_CookieDBManager", "_ISINDBManager"):
        mgr = getattr(yfc, name, None)
        close_db = getattr(mgr, "close_db", None) if mgr is not None else None
        if callable(close_db):
            with contextlib.suppress(Exception):
                close_db()


@pytest.mark.determinism("d1")
def test_stat_arb_pairs_vertical_slice_pit_and_artifacts(tmp_path):
    """Deterministic integration test: PIT-safe path and bundle artifacts for stat_arb_pairs."""
    rows = []
    for i, (spy, qqq) in enumerate(
        [
            (100.0, 50.0),
            (101.0, 50.5),
            (102.0, 51.0),
            (103.0, 51.5),
        ],
        start=0,
    ):
        d = date(2024, 1, 1 + i)
        rows.append(
            {
                "symbol": "SPY",
                "valid_time": d,
                "knowledge_time": d,
                "SPY.close": spy,
                "QQQ.close": float("nan"),
            }
        )
        rows.append(
            {
                "symbol": "QQQ",
                "valid_time": d,
                "knowledge_time": d,
                "SPY.close": float("nan"),
                "QQQ.close": qqq,
            }
        )
    prices = pd.DataFrame(rows)

    cfg = StatArbRunConfig(
        leg_a="SPY",
        leg_b="QQQ",
        start="2024-01-01",
        end="2024-01-04",
        config=PAIRS_DEFAULT,
        bundle_dir=tmp_path / "bundle",
        prices=prices,
    )
    bundle = run_stat_arb_pairs("SPY", "QQQ", run_cfg=cfg)

    # Bundle JSON files should exist, including optional artifacts.
    assert (tmp_path / "bundle" / "execution_assumptions.json").exists()
    assert (tmp_path / "bundle" / "stat_validity_report.json").exists()
    # RunBundle structure is populated.
    assert bundle.plan.schema_version in ("1.0", "1.0.0")
    assert bundle.dataset_manifest.row_count == len(prices)


@pytest.mark.integration
@pytest.mark.net
def test_stat_arb_pairs_yahoo_spy_qqq_smoke(tmp_path):
    """Live-network smoke test for SPY/QQQ via Yahoo PIT-safe path."""
    from pysrc.pipeline.stages.market_data.sources.yahoo_fetcher import YahooFinanceSource

    source = YahooFinanceSource()
    import asyncio

    async def _fetch_pair():
        spy = await source.get_historical("SPY", "2022-01-01", "2022-12-31", eager=True)
        qqq = await source.get_historical("QQQ", "2022-01-01", "2022-12-31", eager=True)
        return spy, qqq

    try:
        try:
            spy_frame, qqq_frame = asyncio.run(_fetch_pair())
        except (DataFetchError, ImportError, OSError) as exc:
            pytest.skip(f"Yahoo optional provider unavailable: {exc}")
        # Join on date: select only date + close and cast date to Date so to_pandas() does not
        # produce datetime64[us, UTC] (pandas/numpy in some envs cannot interpret that dtype).
        spy_df = (
            spy_frame.select(["date", "close"])
            .with_columns(pl.col("date").cast(pl.Date))
            .to_pandas()
        )
        qqq_df = (
            qqq_frame.select(["date", "close"])
            .with_columns(pl.col("date").cast(pl.Date))
            .to_pandas()
        )
        merged = spy_df.merge(
            qqq_df[["date", "close"]].rename(columns={"close": "QQQ.close"}),
            on=["date"],
            how="inner",
            suffixes=("", ""),
        )
        merged = merged.rename(columns={"close": "SPY.close"})

        # Long format: one row per symbol per date for DataView.as_of
        spy_rows = merged.assign(symbol="SPY").rename(columns={"SPY.close": "SPY.close"})
        spy_rows["QQQ.close"] = float("nan")
        qqq_rows = merged.assign(symbol="QQQ")
        qqq_rows["SPY.close"] = float("nan")
        qqq_rows = (
            qqq_rows.rename(columns={"close": "QQQ.close"})
            if "close" in qqq_rows.columns
            else qqq_rows
        )
        if "valid_time" not in merged.columns:
            merged["valid_time"] = pd.to_datetime(merged["date"])
            merged["knowledge_time"] = merged["valid_time"]
        prices = pd.concat(
            [
                pd.DataFrame(
                    {
                        "symbol": "SPY",
                        "valid_time": merged["valid_time"],
                        "knowledge_time": merged["knowledge_time"],
                        "SPY.close": merged["SPY.close"],
                        "QQQ.close": float("nan"),
                    }
                ),
                pd.DataFrame(
                    {
                        "symbol": "QQQ",
                        "valid_time": merged["valid_time"],
                        "knowledge_time": merged["knowledge_time"],
                        "SPY.close": float("nan"),
                        "QQQ.close": merged["QQQ.close"],
                    }
                ),
            ],
            ignore_index=True,
        )

        cfg = StatArbRunConfig(
            leg_a="SPY",
            leg_b="QQQ",
            start="2022-01-01",
            end="2022-12-31",
            config=PAIRS_DEFAULT,
            bundle_dir=tmp_path / "bundle",
            prices=prices,
        )
        bundle = run_stat_arb_pairs("SPY", "QQQ", run_cfg=cfg)

        assert (tmp_path / "bundle" / "execution_assumptions.json").exists()
        assert (tmp_path / "bundle" / "stat_validity_report.json").exists()
        assert bundle.dataset_manifest.row_count == len(prices)
    finally:
        _close_yfinance_peewee_dbs()


# --- entry.py validation and branch coverage ---


def test_run_stat_arb_pairs_requires_non_empty_prices():
    """run_stat_arb_pairs raises when prices is None or empty."""
    with pytest.raises(ValueError, match="run_cfg.prices.*non-empty"):
        run_stat_arb_pairs(
            "SPY",
            "QQQ",
            run_cfg=StatArbRunConfig(
                leg_a="SPY", leg_b="QQQ", start="2024-01-01", end="2024-01-04", prices=None
            ),
        )
    with pytest.raises(ValueError, match="run_cfg.prices.*non-empty"):
        run_stat_arb_pairs(
            "SPY",
            "QQQ",
            run_cfg=StatArbRunConfig(
                leg_a="SPY",
                leg_b="QQQ",
                start="2024-01-01",
                end="2024-01-04",
                prices=pd.DataFrame(),
            ),
        )


def test_run_stat_arb_pairs_requires_symbol_column():
    """run_stat_arb_pairs raises when prices lacks 'symbol'."""
    prices = pd.DataFrame(
        {
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "SPY.close": [100.0],
            "QQQ.close": [50.0],
        }
    )
    with pytest.raises(ValueError, match="'symbol' column"):
        run_stat_arb_pairs(
            "SPY",
            "QQQ",
            run_cfg=StatArbRunConfig(
                leg_a="SPY",
                leg_b="QQQ",
                start="2024-01-01",
                end="2024-01-04",
                bundle_dir=None,
                prices=prices,
            ),
        )


def test_run_stat_arb_pairs_requires_valid_time_and_knowledge_time():
    """run_stat_arb_pairs raises when prices lacks valid_time or knowledge_time."""
    prices = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "SPY.close": [100.0],
            "QQQ.close": [50.0],
        }
    )
    with pytest.raises(ValueError, match="valid_time.*knowledge_time"):
        run_stat_arb_pairs(
            "SPY",
            "QQQ",
            run_cfg=StatArbRunConfig(
                leg_a="SPY",
                leg_b="QQQ",
                start="2024-01-01",
                end="2024-01-04",
                bundle_dir=None,
                prices=prices,
            ),
        )


def test_run_stat_arb_pairs_knowledge_dates_from_iso_string(tmp_path):
    """run_stat_arb_pairs accepts knowledge_time as string (fromisoformat branch)."""
    rows = [
        {
            "symbol": "SPY",
            "valid_time": "2024-01-01",
            "knowledge_time": "2024-01-01",
            "SPY.close": 100.0,
            "QQQ.close": float("nan"),
        },
        {
            "symbol": "QQQ",
            "valid_time": "2024-01-01",
            "knowledge_time": "2024-01-01",
            "SPY.close": float("nan"),
            "QQQ.close": 50.0,
        },
    ]
    prices = pd.DataFrame(rows)
    cfg = StatArbRunConfig(
        leg_a="SPY",
        leg_b="QQQ",
        start="2024-01-01",
        end="2024-01-01",
        config=PAIRS_DEFAULT,
        bundle_dir=tmp_path / "bundle",
        prices=prices,
    )
    bundle = run_stat_arb_pairs("SPY", "QQQ", run_cfg=cfg)
    assert bundle.plan.schema_version in ("1.0", "1.0.0")


def test_run_stat_arb_pairs_uses_default_bundle_dir_when_none(tmp_path):
    """When bundle_dir is None, default path is used."""
    rows = []
    for i in range(2):
        d = date(2024, 1, 1 + i)
        rows.append(
            {
                "symbol": "SPY",
                "valid_time": d,
                "knowledge_time": d,
                "SPY.close": 100.0 + i,
                "QQQ.close": float("nan"),
            }
        )
        rows.append(
            {
                "symbol": "QQQ",
                "valid_time": d,
                "knowledge_time": d,
                "SPY.close": float("nan"),
                "QQQ.close": 50.0 + i,
            }
        )
    prices = pd.DataFrame(rows)
    cfg = StatArbRunConfig(
        leg_a="SPY",
        leg_b="QQQ",
        start="2024-01-01",
        end="2024-01-02",
        config=PAIRS_DEFAULT,
        bundle_dir=None,
        prices=prices,
    )
    bundle = run_stat_arb_pairs("SPY", "QQQ", run_cfg=cfg)
    assert bundle.dataset_manifest.row_count == len(prices)
