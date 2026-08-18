from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pysrc.backtesting.contracts.types import PitMeta
from pysrc.data.dataview import DataView
from pysrc.strategies.pipeline_strategy import (
    FeaturePlan,
    FeatureStep,
    StrategyContext,
    materialize_features,
)

pytestmark = pytest.mark.determinism("d2")

FAST_SETTINGS = settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@st.composite
def _market_rows_strategy(draw, min_rows: int = 6, max_rows: int = 12) -> pd.DataFrame:
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    closes = draw(
        st.lists(
            st.floats(min_value=50.0, max_value=150.0, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    spreads = draw(
        st.lists(
            st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    volumes = draw(
        st.lists(st.integers(min_value=100, max_value=5000), min_size=n_rows, max_size=n_rows)
    )
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_rows)]
    highs = [close + spread for close, spread in zip(closes, spreads, strict=False)]
    lows = [max(1.0, close - spread) for close, spread in zip(closes, spreads, strict=False)]
    return pd.DataFrame(
        {
            "symbol": ["SPY"] * n_rows,
            "valid_time": dates,
            "knowledge_time": dates,
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": [float(v) for v in volumes],
        }
    )


def _pit_meta(query_dates: Sequence[date]) -> PitMeta:
    last = query_dates[-1]
    return PitMeta(
        as_of=f"{last.isoformat()}T00:00:00",
        source="pysrc.data.dataview.DataView",
        knowledge_cutoff=last.isoformat(),
    )


def _query_dates(rows: pd.DataFrame) -> list[date]:
    return sorted(pd.to_datetime(rows["knowledge_time"]).dt.date.unique().tolist())


def _build_governed_history(
    rows: pd.DataFrame, query_dates: Sequence[date] | None = None
) -> pd.DataFrame:
    dataview = DataView()
    dataview.register_source(rows)
    dates = list(query_dates or _query_dates(rows))
    snapshots: list[pd.DataFrame] = []
    for query_date in dates:
        snapshot = dataview.as_of(["SPY"], ["close", "high", "low", "volume"], query_date)
        if snapshot.empty:
            continue
        frame = snapshot.copy()
        frame["price"] = frame["close"].astype(float)
        frame.index = pd.DatetimeIndex([pd.Timestamp(query_date)])
        snapshots.append(frame)
    if not snapshots:
        raise AssertionError("expected at least one governed snapshot")
    return pd.concat(snapshots, axis=0)


def _materialize(
    rows: pd.DataFrame, plan: FeaturePlan, tmp_path, query_dates: Sequence[date] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = list(query_dates or _query_dates(rows))
    history = _build_governed_history(rows, dates)
    ctx = StrategyContext(
        prices=history[["price", "close", "high", "low", "volume"]],
        backend="pandas",
        cache_dir=tmp_path,
        pit_provenance=_pit_meta(dates),
    )
    out = materialize_features(ctx, plan, price_col="price")
    assert isinstance(out, pd.DataFrame)
    return history, out


def _append_poison_row(
    rows: pd.DataFrame, *, future_valid_days: int, future_knowledge_days: int, close: float
) -> pd.DataFrame:
    last_valid = max(rows["valid_time"])
    last_knowledge = max(rows["knowledge_time"])
    poison = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "valid_time": [last_valid + timedelta(days=future_valid_days)],
            "knowledge_time": [last_knowledge + timedelta(days=future_knowledge_days)],
            "close": [close],
            "high": [close + 1.0],
            "low": [close - 1.0],
            "volume": [999999.0],
        }
    )
    return pd.concat([rows, poison], ignore_index=True)


def _rolling_mean(values: np.ndarray, window: int, min_samples: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        finite = values[start : i + 1][np.isfinite(values[start : i + 1])]
        if finite.size >= min_samples:
            out[i] = float(finite.mean())
    return out


def _rolling_std(values: np.ndarray, window: int, min_samples: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        finite = values[start : i + 1][np.isfinite(values[start : i + 1])]
        if finite.size >= min_samples:
            out[i] = float(np.std(finite, ddof=0))
    return out


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.full(len(values), np.nan, dtype=float)
    started = False
    acc = 0.0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            if started:
                out[i] = acc
            continue
        if not started:
            acc = float(value)
            started = True
        else:
            acc = alpha * float(value) + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _wilder_smooth(values: np.ndarray, window: int) -> np.ndarray:
    alpha = 1.0 / max(1, window)
    out = np.full(len(values), np.nan, dtype=float)
    started = False
    acc = 0.0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            if started:
                out[i] = acc
            continue
        if not started:
            acc = float(value)
            started = True
        else:
            acc = alpha * float(value) + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _rsi(values: np.ndarray, window: int) -> np.ndarray:
    delta = np.full(len(values), np.nan, dtype=float)
    delta[1:] = values[1:] - values[:-1]
    gains = np.where(np.isfinite(delta), np.clip(delta, 0.0, None), np.nan)
    losses = np.where(np.isfinite(delta), -np.clip(delta, None, 0.0), np.nan)
    avg_gain = _wilder_smooth(gains, window)
    avg_loss = _wilder_smooth(losses, window)
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(
    values: np.ndarray, fast: int, slow: int, signal: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    macd = _ema(values, fast) - _ema(values, slow)
    signal_line = _ema(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist


def _bollinger(
    values: np.ndarray, window: int, num_std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mid = _rolling_mean(values, window, window)
    std = _rolling_std(values, window, window)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, std, upper, lower


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    tr = np.full(len(close), np.nan, dtype=float)
    for i in range(len(close)):
        if i == 0 or not np.isfinite(close[i - 1]):
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return _wilder_smooth(tr, window)


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    out = np.zeros(len(close), dtype=float)
    for i in range(1, len(close)):
        if not np.isfinite(close[i]) or not np.isfinite(close[i - 1]) or not np.isfinite(volume[i]):
            out[i] = out[i - 1]
        elif close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out


def _zscore(values: np.ndarray, window: int) -> np.ndarray:
    mean = _rolling_mean(values, window, window)
    std = _rolling_std(values, window, window)
    return (values - mean) / np.where(std == 0.0, np.nan, std)


def _assert_close(actual: Iterable[float], expected: Iterable[float]) -> None:
    actual_arr = np.asarray(list(actual), dtype=float)
    expected_arr = np.asarray(list(expected), dtype=float)
    assert np.allclose(actual_arr, expected_arr, equal_nan=True)


def _sma_plan(window: int = 3) -> FeaturePlan:
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                "technical.SMA",
                inputs=("price",),
                kwargs={"input_col": "price", "window": window, "out_col": f"sma{window}"},
            )
        ]
    )


def _rolling_std_plan(window: int = 3) -> FeaturePlan:
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                "stats.rolling_std",
                inputs=("price",),
                kwargs={"col": "price", "window": window, "out_col": f"std{window}"},
            )
        ]
    )


def _ema_plan(span: int = 3) -> FeaturePlan:
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                "technical.EMA",
                inputs=("price",),
                kwargs={"input_col": "price", "span": span, "out_col": f"ema{span}"},
            )
        ]
    )


def _rsi_plan(window: int = 3) -> FeaturePlan:
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                "technical.RSI",
                inputs=("price",),
                kwargs={"input_col": "price", "window": window, "out_col": f"rsi{window}"},
            )
        ]
    )


def _macd_plan() -> FeaturePlan:
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                "technical.MACD_line_signal",
                inputs=("price",),
                kwargs={
                    "input_col": "price",
                    "fast": 2,
                    "slow": 3,
                    "signal": 2,
                    "out_macd": "macd",
                    "out_signal": "macd_signal",
                    "out_hist": "macd_hist",
                },
            )
        ]
    )


def _bollinger_plan() -> FeaturePlan:
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                "technical.Bollinger",
                inputs=("price",),
                kwargs={
                    "input_col": "price",
                    "window": 3,
                    "num_std": 2.0,
                    "out_mid": "mid",
                    "out_std": "std",
                    "out_upper": "upper",
                    "out_lower": "lower",
                },
            )
        ]
    )


def _atr_plan() -> FeaturePlan:
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                "technical.ATR",
                inputs=(),
                kwargs={
                    "high_col": "high",
                    "low_col": "low",
                    "close_col": "close",
                    "window": 3,
                    "out_col": "atr3",
                },
            )
        ]
    )


def _obv_plan() -> FeaturePlan:
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                "technical.OBV",
                inputs=("close",),
                kwargs={"input_col": "close", "volume_col": "volume", "out_col": "obv"},
            )
        ]
    )


def _zscore_plan() -> FeaturePlan:
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                "scaling.zscore_roll",
                inputs=("price",),
                kwargs={"col": "price", "window": 3, "out_col": "z3"},
            )
        ]
    )


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_feature_pit_invariant_historical_frame_never_exceeds_query_cutoff(
    rows: pd.DataFrame,
) -> None:
    history = _build_governed_history(rows)
    for ts, row in history.iterrows():
        cutoff = ts.date()
        assert row["valid_time"] <= cutoff
        assert row["knowledge_time"] <= cutoff


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_governed_sma_poison_pill_future_valid_does_not_change_visible_history(
    rows: pd.DataFrame, tmp_path
) -> None:
    dates = _query_dates(rows)
    _, base = _materialize(rows, _sma_plan(), tmp_path, dates)
    poisoned_rows = _append_poison_row(
        rows, future_valid_days=7, future_knowledge_days=7, close=999.0
    )
    _, poisoned = _materialize(poisoned_rows, _sma_plan(), tmp_path, dates)
    _assert_close(base["sma3"], poisoned["sma3"])


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_governed_rolling_std_poison_pill_future_knowledge_does_not_change_visible_history(
    rows: pd.DataFrame, tmp_path
) -> None:
    dates = _query_dates(rows)
    _, base = _materialize(rows, _rolling_std_plan(), tmp_path, dates)
    poisoned_rows = _append_poison_row(
        rows, future_valid_days=0, future_knowledge_days=7, close=888.0
    )
    _, poisoned = _materialize(poisoned_rows, _rolling_std_plan(), tmp_path, dates)
    _assert_close(base["std3"], poisoned["std3"])


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_governed_ema_matches_visible_history(rows: pd.DataFrame, tmp_path) -> None:
    history, out = _materialize(rows, _ema_plan(), tmp_path)
    expected = _ema(history["price"].to_numpy(dtype=float), span=3)
    _assert_close(out["ema3"], expected)


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_governed_rsi_matches_visible_history(rows: pd.DataFrame, tmp_path) -> None:
    history, out = _materialize(rows, _rsi_plan(), tmp_path)
    expected = _rsi(history["price"].to_numpy(dtype=float), window=3)
    _assert_close(out["rsi3"], expected)


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_governed_macd_outputs_match_visible_history(rows: pd.DataFrame, tmp_path) -> None:
    history, out = _materialize(rows, _macd_plan(), tmp_path)
    macd, signal, hist = _macd(history["price"].to_numpy(dtype=float), fast=2, slow=3, signal=2)
    _assert_close(out["macd"], macd)
    _assert_close(out["macd_signal"], signal)
    _assert_close(out["macd_hist"], hist)


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_governed_macd_poison_pill_preserves_all_emitted_columns(
    rows: pd.DataFrame, tmp_path
) -> None:
    dates = _query_dates(rows)
    _, base = _materialize(rows, _macd_plan(), tmp_path, dates)
    poisoned_rows = _append_poison_row(
        rows, future_valid_days=9, future_knowledge_days=9, close=777.0
    )
    _, poisoned = _materialize(poisoned_rows, _macd_plan(), tmp_path, dates)
    for column in ["macd", "macd_signal", "macd_hist"]:
        _assert_close(base[column], poisoned[column])


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_governed_bollinger_outputs_match_visible_history(rows: pd.DataFrame, tmp_path) -> None:
    history, out = _materialize(rows, _bollinger_plan(), tmp_path)
    mid, std, upper, lower = _bollinger(
        history["price"].to_numpy(dtype=float), window=3, num_std=2.0
    )
    _assert_close(out["mid"], mid)
    _assert_close(out["std"], std)
    _assert_close(out["upper"], upper)
    _assert_close(out["lower"], lower)


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_governed_bollinger_poison_pill_preserves_all_emitted_columns(
    rows: pd.DataFrame, tmp_path
) -> None:
    dates = _query_dates(rows)
    _, base = _materialize(rows, _bollinger_plan(), tmp_path, dates)
    poisoned_rows = _append_poison_row(
        rows, future_valid_days=8, future_knowledge_days=8, close=666.0
    )
    _, poisoned = _materialize(poisoned_rows, _bollinger_plan(), tmp_path, dates)
    for column in ["mid", "std", "upper", "lower"]:
        _assert_close(base[column], poisoned[column])


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_governed_atr_matches_visible_history(rows: pd.DataFrame, tmp_path) -> None:
    history, out = _materialize(rows, _atr_plan(), tmp_path)
    expected = _atr(
        history["high"].to_numpy(dtype=float),
        history["low"].to_numpy(dtype=float),
        history["close"].to_numpy(dtype=float),
        window=3,
    )
    _assert_close(out["atr3"], expected)


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_governed_obv_matches_visible_history(rows: pd.DataFrame, tmp_path) -> None:
    history, out = _materialize(rows, _obv_plan(), tmp_path)
    expected = _obv(history["close"].to_numpy(dtype=float), history["volume"].to_numpy(dtype=float))
    _assert_close(out["obv"], expected)


@FAST_SETTINGS
@given(rows=_market_rows_strategy())
def test_governed_zscore_matches_visible_history(rows: pd.DataFrame, tmp_path) -> None:
    history, out = _materialize(rows, _zscore_plan(), tmp_path)
    expected = _zscore(history["price"].to_numpy(dtype=float), window=3)
    _assert_close(out["z3"], expected)
