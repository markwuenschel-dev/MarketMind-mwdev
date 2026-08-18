from __future__ import annotations

from datetime import date

import pandas as pd  # type: ignore[import-untyped]
import pytest

pytestmark = pytest.mark.determinism("d1")


def test_adjuster_uses_massive_historical_factors_and_preserves_raw_fields(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.corporate_actions import (
        build_adjusted_ohlcv_panel,
    )

    bars = pd.DataFrame(
        [
            {
                "symbol": "XYZ",
                "date": date(2025, 1, 2),
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": 100.0,
                "volume": 1_000,
            },
            {
                "symbol": "XYZ",
                "date": date(2025, 1, 3),
                "open": 52.0,
                "high": 56.0,
                "low": 50.0,
                "close": 55.0,
                "volume": 2_000,
            },
        ]
    )
    splits = pd.DataFrame(
        [
            {
                "ticker": "XYZ",
                "execution_date": "2025-01-03",
                "historical_adjustment_factor": 0.5,
                "split_from": 1.0,
                "split_to": 2.0,
            }
        ]
    )
    dividends = pd.DataFrame(
        [
            {
                "ticker": "XYZ",
                "ex_dividend_date": "2025-01-03",
                "historical_adjustment_factor": 0.98,
                "cash_amount": 1.0,
            }
        ]
    )

    adjusted = build_adjusted_ohlcv_panel(bars, splits=splits, dividends=dividends)

    first = adjusted.iloc[0]
    second = adjusted.iloc[1]
    assert first["raw_open"] == 100.0
    assert first["raw_high"] == 110.0
    assert first["raw_low"] == 90.0
    assert first["raw_close"] == 100.0
    assert first["raw_volume"] == 1_000
    assert first["split_adjustment_factor"] == pytest.approx(0.5)
    assert first["dividend_adjustment_factor"] == pytest.approx(0.98)
    assert first["total_price_adjustment_factor"] == pytest.approx(0.49)
    assert first["adj_open"] == pytest.approx(49.0)
    assert first["adj_high"] == pytest.approx(53.9)
    assert first["adj_low"] == pytest.approx(44.1)
    assert first["adj_close"] == pytest.approx(49.0)
    assert first["adj_volume"] == pytest.approx(2_000.0)
    assert first["corporate_action_flag"] is True

    assert second["split_adjustment_factor"] == pytest.approx(1.0)
    assert second["dividend_adjustment_factor"] == pytest.approx(1.0)
    assert second["adj_close"] == pytest.approx(55.0)
    assert second["raw_return_1d"] == pytest.approx(-0.45)
    assert second["adjusted_return_1d"] == pytest.approx(55.0 / 49.0 - 1.0)
    assert second["raw_vs_adjusted_return_delta"] == pytest.approx(
        second["raw_return_1d"] - second["adjusted_return_1d"]
    )


def test_adjuster_cumulates_multiple_future_action_factors(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.corporate_actions import (
        build_adjusted_ohlcv_panel,
    )

    bars = pd.DataFrame(
        [
            {
                "symbol": "XYZ",
                "date": "2025-01-02",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 100,
            },
            {
                "symbol": "XYZ",
                "date": "2025-01-03",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 100,
            },
            {
                "symbol": "XYZ",
                "date": "2025-01-06",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 100,
            },
        ]
    )
    splits = pd.DataFrame(
        [
            {"ticker": "XYZ", "execution_date": "2025-01-03", "historical_adjustment_factor": 0.5},
            {"ticker": "XYZ", "execution_date": "2025-01-06", "historical_adjustment_factor": 0.25},
        ]
    )

    adjusted = build_adjusted_ohlcv_panel(bars, splits=splits, dividends=pd.DataFrame())

    assert adjusted.loc[0, "split_adjustment_factor"] == pytest.approx(0.125)
    assert adjusted.loc[1, "split_adjustment_factor"] == pytest.approx(0.25)
    assert adjusted.loc[2, "split_adjustment_factor"] == pytest.approx(1.0)


def test_adjuster_declares_no_repair_or_full_quality_detection(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.corporate_actions import (
        build_adjusted_ohlcv_panel,
    )

    bars = pd.DataFrame(
        [
            {
                "symbol": "XYZ",
                "date": "2025-01-02",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "volume": 100,
            }
        ]
    )

    adjusted = build_adjusted_ohlcv_panel(
        bars,
        splits=pd.DataFrame(),
        dividends=pd.DataFrame(),
    )

    row = adjusted.iloc[0]
    assert "repair_mode" not in adjusted.columns
    assert row["bar_quality_mode"] == "none"
    assert row["bar_quality_detector_version"] is None
    assert row["bar_quality_flags"] == []
    assert row["extreme_raw_return_flag"] is False
    assert row["extreme_adjusted_return_flag"] is False


def test_adjuster_exports_price_usage_policy(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.corporate_actions import (
        ADJUSTED_RETURN_PRICE_COLUMN,
        FILTER_PRICE_COLUMN,
        FORWARD_LABEL_PRICE_COLUMN,
        LIQUIDITY_VOLUME_COLUMN,
    )

    assert ADJUSTED_RETURN_PRICE_COLUMN == "adj_close"
    assert FORWARD_LABEL_PRICE_COLUMN == "adj_close"
    assert FILTER_PRICE_COLUMN == "raw_close"
    assert LIQUIDITY_VOLUME_COLUMN == "raw_volume"


def test_corporate_actions_loader_normalizes_splits_and_dividends(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.sources.massive_corporate_actions_loader import (
        normalize_dividend_records,
        normalize_split_records,
    )

    splits = normalize_split_records(
        [
            {
                "ticker": "XYZ",
                "execution_date": "2025-01-03",
                "historical_adjustment_factor": 0.5,
                "split_from": 1,
                "split_to": 2,
                "adjustment_type": "forward_split",
                "id": "split-1",
            }
        ]
    )
    dividends = normalize_dividend_records(
        [
            {
                "ticker": "XYZ",
                "ex_dividend_date": "2025-01-04",
                "historical_adjustment_factor": 0.98,
                "cash_amount": 1.0,
                "distribution_type": "recurring",
                "id": "div-1",
            }
        ]
    )

    assert list(splits.columns) == [
        "ticker",
        "execution_date",
        "historical_adjustment_factor",
        "split_from",
        "split_to",
        "adjustment_type",
        "id",
    ]
    assert list(dividends.columns) == [
        "ticker",
        "ex_dividend_date",
        "historical_adjustment_factor",
        "cash_amount",
        "distribution_type",
        "id",
    ]
    assert splits.iloc[0]["historical_adjustment_factor"] == pytest.approx(0.5)
    assert dividends.iloc[0]["historical_adjustment_factor"] == pytest.approx(0.98)


def test_corporate_actions_loader_parse_args_uses_default_actions(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.sources.massive_corporate_actions_loader import (
        parse_args,
    )

    config = parse_args(["--start", "2025-01-02", "--end", "2025-01-31"])

    assert config.actions == ("splits", "dividends")
    assert config.tickers == ()


def test_corporate_actions_loader_writes_manifest_parent(tmp_path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.sources.massive_corporate_actions_loader import (
        CorporateActionsConfig,
        CorporateActionsFileResult,
        write_manifest,
    )

    manifest_path = write_manifest(
        CorporateActionsConfig(
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 31),
            output_root=tmp_path / "actions",
            actions=("splits", "dividends"),
            tickers=(),
            api_key_env="MASSIVE_API_KEY",
            limit=1000,
            verbose=False,
        ),
        [
            CorporateActionsFileResult(
                action="splits",
                path="splits.parquet",
                rows=0,
                content_blake2b="abc",
                status="written",
                duration_seconds=0.0,
            )
        ],
    )

    assert manifest_path.exists()
    assert manifest_path.parent.name == "_manifests"


def test_daily_panel_builder_writes_one_file_per_day_with_actions_in_row(
    tmp_path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.sources.massive_adjusted_day_panel_builder import (
        AdjustedPanelBuildConfig,
        build_adjusted_daily_panels,
    )

    raw_root = tmp_path / "raw"
    actions_root = tmp_path / "actions"
    output_root = tmp_path / "panel"
    raw_day = raw_root / "year=2025" / "month=01" / "2025-01-02.parquet"
    raw_day.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "XYZ",
                "volume": 100,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "symbol": "XYZ",
                "date": date(2025, 1, 2),
                "valid_time": date(2025, 1, 2),
            },
            {
                "ticker": "ABC",
                "volume": 200,
                "open": 20.0,
                "high": 21.0,
                "low": 19.0,
                "close": 20.0,
                "symbol": "ABC",
                "date": date(2025, 1, 2),
                "valid_time": date(2025, 1, 2),
            },
        ]
    ).to_parquet(raw_day, index=False)
    actions_root.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "XYZ",
                "execution_date": date(2025, 1, 2),
                "historical_adjustment_factor": 0.5,
                "split_from": 1.0,
                "split_to": 2.0,
                "adjustment_type": "forward_split",
                "id": "split-xyz",
            }
        ]
    ).to_parquet(actions_root / "splits_2025-01-02_2025-01-02.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ticker": "XYZ",
                "ex_dividend_date": date(2025, 1, 2),
                "historical_adjustment_factor": 0.98,
                "cash_amount": 1.0,
                "distribution_type": "recurring",
                "id": "div-xyz",
            }
        ]
    ).to_parquet(actions_root / "dividends_2025-01-02_2025-01-02.parquet", index=False)

    results = build_adjusted_daily_panels(
        AdjustedPanelBuildConfig(
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 2),
            raw_root=raw_root,
            corporate_actions_root=actions_root,
            output_root=output_root,
            splits_path=None,
            dividends_path=None,
            overwrite=False,
            verbose=False,
        )
    )

    assert len(results) == 1
    out_path = output_root / "year=2025" / "month=01" / "2025-01-02.parquet"
    assert out_path.exists()
    panel = pd.read_parquet(out_path).sort_values("symbol").reset_index(drop=True)
    assert {"raw_open", "split_execution_date", "dividend_ex_dividend_date", "adj_close"} <= set(
        panel.columns
    )
    xyz = panel.loc[panel["symbol"] == "XYZ"].iloc[0]
    abc = panel.loc[panel["symbol"] == "ABC"].iloc[0]
    assert xyz["split_historical_adjustment_factor"] == pytest.approx(0.5)
    assert xyz["dividend_historical_adjustment_factor"] == pytest.approx(0.98)
    assert xyz["total_price_adjustment_factor"] == pytest.approx(1.0)
    assert bool(xyz["corporate_action_flag"]) is True
    assert abc["split_historical_adjustment_factor"] != abc["split_historical_adjustment_factor"]
    assert (
        abc["dividend_historical_adjustment_factor"] != abc["dividend_historical_adjustment_factor"]
    )
