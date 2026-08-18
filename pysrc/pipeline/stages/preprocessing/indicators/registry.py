"""Indicator registry for the W3-B pandas-ta-classic library."""

from __future__ import annotations

from dataclasses import dataclass

from pysrc.pipeline.stages.preprocessing.indicators.schema import (
    W3B_INDICATOR_IDS,
    IndicatorCategory,
)


@dataclass(frozen=True, slots=True)
class IndicatorDefinition:
    indicator_id: str
    category: IndicatorCategory
    source: str
    parameters: dict[str, int | float | str]


_DEFINITIONS: tuple[IndicatorDefinition, ...] = (
    IndicatorDefinition("rsi_14", "momentum", "pandas_ta_classic.rsi", {"length": 14}),
    IndicatorDefinition("roc_5", "momentum", "pandas_ta_classic.roc", {"length": 5}),
    IndicatorDefinition("roc_10", "momentum", "pandas_ta_classic.roc", {"length": 10}),
    IndicatorDefinition("roc_20", "momentum", "pandas_ta_classic.roc", {"length": 20}),
    IndicatorDefinition("stoch_k_14_3", "momentum", "pandas_ta_classic.stoch", {"k": 14, "d": 3}),
    IndicatorDefinition("stoch_d_14_3", "momentum", "pandas_ta_classic.stoch", {"k": 14, "d": 3}),
    IndicatorDefinition("williams_r_14", "momentum", "pandas_ta_classic.willr", {"length": 14}),
    IndicatorDefinition("cci_20", "momentum", "pandas_ta_classic.cci", {"length": 20}),
    IndicatorDefinition(
        "macd_12_26_9", "trend", "pandas_ta_classic.macd", {"fast": 12, "slow": 26, "signal": 9}
    ),
    IndicatorDefinition(
        "macd_hist_12_26_9",
        "trend",
        "pandas_ta_classic.macd",
        {"fast": 12, "slow": 26, "signal": 9},
    ),
    IndicatorDefinition("adx_14", "trend", "pandas_ta_classic.adx", {"length": 14}),
    IndicatorDefinition("dmp_14", "trend", "pandas_ta_classic.adx", {"length": 14}),
    IndicatorDefinition("dmn_14", "trend", "pandas_ta_classic.adx", {"length": 14}),
    IndicatorDefinition("ema_distance_20", "trend", "pandas_ta_classic.ema", {"length": 20}),
    IndicatorDefinition("ema_distance_50", "trend", "pandas_ta_classic.ema", {"length": 50}),
    IndicatorDefinition(
        "sma_cross_20_50", "trend", "pandas_ta_classic.sma", {"fast": 20, "slow": 50}
    ),
    IndicatorDefinition("atr_14", "volatility", "pandas_ta_classic.atr", {"length": 14}),
    IndicatorDefinition("natr_14", "volatility", "pandas_ta_classic.natr", {"length": 14}),
    IndicatorDefinition(
        "bb_percent_b_20_2", "volatility", "pandas_ta_classic.bbands", {"length": 20, "std": 2}
    ),
    IndicatorDefinition(
        "bb_bandwidth_20_2", "volatility", "pandas_ta_classic.bbands", {"length": 20, "std": 2}
    ),
    IndicatorDefinition("keltner_width_20", "volatility", "pandas_ta_classic.kc", {"length": 20}),
    IndicatorDefinition(
        "donchian_position_20", "volatility", "pandas_ta_classic.donchian", {"length": 20}
    ),
    IndicatorDefinition(
        "obv_change_20", "volume_liquidity", "pandas_ta_classic.obv", {"length": 20}
    ),
    IndicatorDefinition("mfi_14", "volume_liquidity", "pandas_ta_classic.mfi", {"length": 14}),
    IndicatorDefinition(
        "volume_zscore_20", "volume_liquidity", "derived.adjusted_volume_zscore", {"length": 20}
    ),
    IndicatorDefinition(
        "volume_ratio_20", "volume_liquidity", "derived.adjusted_volume_ratio", {"length": 20}
    ),
    IndicatorDefinition(
        "volume_dollar_zscore_20",
        "volume_liquidity",
        "derived.raw_dollar_volume_zscore",
        {"length": 20},
    ),
    IndicatorDefinition(
        "dollar_volume_ratio_20",
        "volume_liquidity",
        "derived.raw_dollar_volume_ratio",
        {"length": 20},
    ),
    IndicatorDefinition(
        "drawdown_60", "risk_state", "derived.adjusted_close_drawdown", {"length": 60}
    ),
    IndicatorDefinition(
        "realized_volatility_20", "risk_state", "derived.adjusted_return_volatility", {"length": 20}
    ),
    IndicatorDefinition(
        "range_ratio_10", "risk_state", "derived.adjusted_range_ratio", {"length": 10}
    ),
)


def indicator_definitions() -> tuple[IndicatorDefinition, ...]:
    """Return indicator definitions in the registered W3-B v1 order."""

    return _DEFINITIONS


def indicator_ids() -> tuple[str, ...]:
    """Return the canonical W3-B v1 indicator IDs."""

    return tuple(definition.indicator_id for definition in _DEFINITIONS)


def indicator_category_map() -> dict[str, IndicatorCategory]:
    return {definition.indicator_id: definition.category for definition in _DEFINITIONS}


def indicator_config_payload() -> dict[str, object]:
    """YAML/JSON-ready payload emitted as indicator_library_v1.yml."""

    return {
        "schema_version": "w3_b.pandas_ta.indicator_library.v1",
        "provider": "pandas-ta-classic",
        "lag_bars": 1,
        "save_long_indicator_rows": False,
        "indicators": [
            {
                "indicator_id": definition.indicator_id,
                "category": definition.category,
                "source": definition.source,
                "parameters": dict(definition.parameters),
            }
            for definition in _DEFINITIONS
        ],
    }


if indicator_ids() != W3B_INDICATOR_IDS:
    raise RuntimeError("W3-B indicator registry does not match schema order")
