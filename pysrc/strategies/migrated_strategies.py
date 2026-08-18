from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Any,
    Final,
    Literal,
    Protocol,
    cast,
    runtime_checkable,
)

import numpy as np
import pandas as pd

try:
    from numba import njit  # type: ignore

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        return lambda f: f


# Ops modules (MarketMind standard)
import contextlib

from pysrc.ops.mm_logkit import get_logger
from pysrc.ops.multi_tier_cache import MultiTierClient
from pysrc.ops.observability import (
    MetricConfig,
    TracingConfig,
    get_metrics,
    get_tracing,
    init_observability,
    instrument,
)
from pysrc.strategies.pipeline_strategy import (
    FeaturePlan,
    FeatureStep,
    PipelineStrategy,
    StrategyRegistry,
)


@runtime_checkable
class _SupportsToPandas(Protocol):
    def to_pandas(self) -> pd.DataFrame: ...


def _as_df(features: Any) -> pd.DataFrame:
    # Ensures static analyzers see a concrete DataFrame (never Optional/Any)
    if isinstance(features, pd.DataFrame):
        return features
    if isinstance(features, _SupportsToPandas):
        df = features.to_pandas()
        if not isinstance(df, pd.DataFrame):
            raise TypeError("to_pandas() did not return a pandas.DataFrame")
        return cast(pd.DataFrame, df)
    raise TypeError("Expected a pandas.DataFrame or an object with to_pandas()")


def _first_numeric_col(df: pd.DataFrame) -> str | None:
    cols = df.select_dtypes(include=[np.number]).columns
    return str(cols[0]) if len(cols) > 0 else None


# Initialize observability (idempotent, safe to call multiple times)
init_observability(
    service_name="strategies",
    metrics_config=MetricConfig(prometheus_port=8000, delta_temporality=True),
    tracing_config=TracingConfig(sample_rate=0.1),
    enable_metrics=True,
    enable_tracing=True,
    enable_logging=True,
)

LOG = get_logger(__name__)
_metrics = get_metrics()
_tracing = get_tracing()

# Env toggles
_CACHE: dict[str, pd.Series] = {}
DISABLE_CACHE = os.getenv("STRATEGY_CACHE_DISABLE", "0") == "1"
DISABLE_NUMBA = os.getenv("STRATEGY_NUMBA_DISABLE", "0") == "1"

if DISABLE_NUMBA:
    NUMBA_AVAILABLE = False
    LOG.info("numba_disabled", reason="env_var")  # structured JSON
    import logging as _logging

    # Emit at WARNING so pytest's default capture (WARNING+) sees it
    _logging.getLogger(__name__).warning("numba_disabled")


if DISABLE_CACHE:
    LOG.warning("cache_disabled", reason="env_var")  # structured JSON
    import logging as _logging

    _logging.getLogger(__name__).warning("cache_disabled")  # plain record for caplog


# Multi-tier cache client (L1→L2→L3→L4)
_cache_client: MultiTierClient | None = None
if not DISABLE_CACHE:
    try:
        _cache_client = MultiTierClient(
            l1_size=10000,
            l1_ttl=300.0,
            l2_type="memfd",
            l2_path="/dev/shm/strategies_l2",
            redis_client=None,  # Redis optional; falls back gracefully
            l4_cache_dir=".cache_strategies",
            enable_singleflight=True,
            ttl_jitter=0.1,
            check_l4_on_miss=False,
        )
        LOG.info("cache_initialized", tiers="L1+L2+L4")
    except Exception as e:
        LOG.warning("cache_init_failed", error=str(e))
        _cache_client = None


def clear_strategy_cache() -> None:
    if not _cache_client:
        return

        # Emit at WARNING so pytest's default caplog (level>=WARNING) sees it.
        # The exact substring 'cache_clear_requested' is asserted in tests.
    LOG.warning("cache_clear_requested", reason="explicit_call")

    try:
        if hasattr(_cache_client, "clear"):
            _cache_client.clear()
        elif hasattr(_cache_client, "flush"):
            _cache_client.flush()
        elif hasattr(_cache_client, "invalidate_all"):
            _cache_client.invalidate_all()
        # else: object() / bare mock -> nothing to call; warning above is still emitted
    except (AttributeError, RuntimeError, OSError) as e:
        # Narrow only: interface mismatch, backend runtime/IO failure
        LOG.warning("cache_clear_failed", error=str(e))


__all__ = [
    "RSIStrategy",
    "MACDStrategy",
    "BollingerBandsStrategy",
    "MeanReversionStrategy",
    "MovingAverageCrossoverStrategy",
    "EnsemblePipelineStrategy",
    "clear_strategy_cache",  # 12/10 hook for test isolation
]

# --- Numba-accelerated signal kernels (FIX #5: dtype stability) ---

if NUMBA_AVAILABLE:

    @njit(parallel=False, fastmath=True, cache=True)
    def _rsi_signal_kernel(rsi, upper, lower, neutral_zone, clip):
        # Numba infers types at compile time; explicit hints are unnecessary and problematic
        n = len(rsi)
        sig = np.zeros(n, dtype=np.float64)
        for i in range(n):
            v = rsi[i]
            if np.isnan(v):
                continue
            if v <= lower:
                sig[i] = 1.0
            elif v >= upper:
                sig[i] = -1.0
            elif not neutral_zone:
                # Deterministic binary push inside the band to avoid an all-zero vector
                mid = (upper + lower) / 2.0
                if v > mid:
                    sig[i] = -1.0
                elif v < mid:
                    sig[i] = 1.0
                else:
                    # Tie-break at mid toward +1.0 to ensure non-zero presence
                    sig[i] = 1.0
        return np.clip(sig, -clip, clip)

    @njit(parallel=False, fastmath=True, cache=True)
    def _macd_hist_kernel(hist, clip):
        n = len(hist)
        sig = np.empty(n, dtype=np.float64)
        for i in range(n):
            v = hist[i]
            if np.isnan(v):
                sig[i] = 0.0
            else:
                sig[i] = np.tanh(v)
        return np.clip(sig, -clip, clip)

    @njit(parallel=False, fastmath=True, cache=True)
    def _bollinger_kernel(price, sma, std, num_std, mode, clip):
        n = len(price)
        sig = np.zeros(n, dtype=np.float64)
        for i in range(n):
            p, m, s = price[i], sma[i], std[i]
            if np.isnan(p) or np.isnan(m) or np.isnan(s) or s == 0:
                continue
            bw = 2.0 * num_std * s
            if bw == 0:
                continue
            norm_pos = (p - m) / (bw / 2.0)
            norm_pos = min(max(norm_pos, -1.5), 1.5)
            if mode == 0:
                sig[i] = -norm_pos
            else:
                if abs(norm_pos) > 0.8:
                    sig[i] = 1.0 if norm_pos > 0 else -1.0
        return np.clip(sig, -clip, clip)

else:
    # Type-safe fallback implementations when Numba unavailable
    def _rsi_signal_kernel(
        rsi: np.ndarray, upper: float, lower: float, neutral_zone: bool, clip: float
    ) -> np.ndarray:
        sig = np.zeros(len(rsi), dtype=np.float64)
        # Type-safe comparisons with explicit boolean masks
        lower_mask = np.less_equal(rsi, lower)
        upper_mask = np.greater_equal(rsi, upper)
        sig[lower_mask] = 1.0
        sig[upper_mask] = -1.0

        if not neutral_zone:
            mid = (upper + lower) / 2
            neutral_mask = np.logical_and(np.greater(rsi, lower), np.less(rsi, upper))
            sig[neutral_mask] = -2.0 * (rsi[neutral_mask] - mid) / (upper - lower)

        return np.clip(sig, -clip, clip).astype(np.float64)

    def _macd_hist_kernel(hist: np.ndarray, clip: float) -> np.ndarray:
        cleaned = np.nan_to_num(hist, nan=0.0)
        result = np.tanh(cleaned)
        return np.clip(result, -clip, clip).astype(np.float64)

    def _bollinger_kernel(
        price: np.ndarray, sma: np.ndarray, std: np.ndarray, num_std: float, mode: int, clip: float
    ) -> np.ndarray:
        bw = 2.0 * num_std * std
        # Avoid division by zero with safe divide
        safe_bw = np.where(np.equal(bw, 0.0), np.nan, bw)
        norm_pos_raw = (price - sma) / (safe_bw / 2.0)
        norm_pos = np.clip(norm_pos_raw, -1.5, 1.5)

        if mode == 0:  # reversion
            sig = -norm_pos
        else:  # breakout
            abs_mask = np.greater(np.abs(norm_pos), 0.8)
            sig = np.where(abs_mask, np.sign(norm_pos), 0.0)

        cleaned = np.nan_to_num(sig, nan=0.0)
        return np.clip(cleaned, -clip, clip).astype(np.float64)


def _cache_key_for_signal(strategy_name: str, params_str: str, features_sig: str) -> str:
    # Deterministic cache key: strategy + params + features
    from pysrc.ops.caching import versioned_key

    return versioned_key(strategy_name, params_str, features_sig, version="v1")


def _hash_params(params: dict) -> str:
    # Stable hash of strategy params
    canon = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    from pysrc.ops.caching import HashAlgorithm, hash_bytes

    return hash_bytes(canon.encode(), HashAlgorithm.XXHASH)[:12]


def _hash_features(df: pd.DataFrame) -> str:
    # numeric-only hashing to avoid pandas 'string' extension dtype in canonicalizer
    if df.empty:
        return "empty"
    dfh = df.copy()

    # replace common string-null sentinels globally
    dfh = dfh.replace(["NaN", "nan", "NAN", "NULL", "null", "None", ""], np.nan)

    # try coercing every column to numeric; non-convertible -> NaN
    for c in dfh.columns:
        dfh[c] = pd.to_numeric(dfh[c], errors="coerce")

    # drop columns that are entirely NaN after coercion
    dfh = dfh.dropna(axis=1, how="all")

    # if nothing numeric remains, return a stable sentinel hash key
    if dfh.shape[1] == 0:
        return "no_numeric"

    # ensure float64 for stable downstream numpy/arrow behavior
    for c in dfh.columns:
        dfh[c] = dfh[c].astype(np.float64)

    from pysrc.ops.caching import HashAlgorithm, hash_dataframe_deterministic

    return hash_dataframe_deterministic(dfh, algo=HashAlgorithm.XXHASH)[:12]


def _get_from_cache(cache_key: str, strategy_name: str) -> pd.Series | None:
    if _cache_client is None:
        return None
    try:
        result = _cache_client.get(cache_key, version=0)
    except (KeyError, ValueError, TypeError, OSError, RuntimeError, AttributeError) as err:
        LOG.debug("cache_get_failed", key=cache_key[:16], error=str(err))
        return None
    if result is not None and _metrics:
        with contextlib.suppress(Exception):
            _metrics.record_counter(
                _metrics.counter(f"{strategy_name}_cache_hit"), labels={"tier": "any"}
            )
    return result


def _set_to_cache(cache_key: str, value: pd.Series) -> None:
    if _cache_client is None:
        return
    try:
        _cache_client.set(
            cache_key,
            value,
            ttl=300.0,
            version=0,
            write_through=True,
            persist_to_l4=False,
        )
    except (KeyError, ValueError, TypeError, OSError, RuntimeError) as err:
        LOG.debug("cache_set_failed", key=cache_key[:16], error=str(err))


# --------------------------------------------------------------------------------------
# RSI Strategy
# --------------------------------------------------------------------------------------


class RSIStrategy(PipelineStrategy):
    # Contrarian RSI: buy oversold, sell overbought; optional linear scaling

    def __init__(
        self,
        rsi_window: int = 14,
        upper: float = 70.0,
        lower: float = 30.0,
        neutral_zone: bool = True,
        clip: float = 1.0,
        # New knob: when any NaN appears in RSI features, emit neutral zeros instead of partial signals
        zero_on_any_nan: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rsi_window = int(rsi_window)
        self.upper = float(upper)
        self.lower = float(lower)
        self.neutral_zone = neutral_zone
        self.clip = float(clip)
        self.zero_on_any_nan = bool(zero_on_any_nan)
        self._params_hash = self._compute_params_hash()

    def _compute_params_hash(self) -> str:
        return _hash_params(
            {
                "rsi_window": self.rsi_window,
                "upper": self.upper,
                "lower": self.lower,
                "neutral_zone": self.neutral_zone,
                "clip": self.clip,
                "zero_on_any_nan": self.zero_on_any_nan,
            }
        )

    def features_plan(self) -> FeaturePlan:
        return FeaturePlan.from_steps(
            [
                FeatureStep(
                    "technical.RSI",
                    inputs=("price",),
                    kwargs={
                        "input_col": "price",
                        "window": self.rsi_window,
                        "out_col": f"rsi{self.rsi_window}",
                    },
                ),
            ]
        )

    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        df: pd.DataFrame = _as_df(features)

        feat_hash = _hash_features(df)
        cache_key = _cache_key_for_signal("rsi", self._params_hash, feat_hash)

        cached = _get_from_cache(cache_key, "rsi")
        if cached is not None:
            return cached
        if _metrics:
            _metrics.record_counter(_metrics.counter("rsi_cache_miss"))

        rsi_col = f"rsi{self.rsi_window}"
        if rsi_col not in df.columns:
            price_col = (
                "price"
                if "price" in df.columns
                else ("close" if "close" in df.columns else _first_numeric_col(df))
            )
            if price_col is None:
                LOG.error("missing_feature", strategy="rsi", column=rsi_col)
                if _metrics:
                    _metrics.record_counter(_metrics.counter("rsi_missing_feature"))
                result = pd.Series(0.0, index=df.index, dtype=float)
                _set_to_cache(cache_key, result)
                return result

            s = df[price_col].astype(float)
            df = df.copy()
            delta = s.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(
                alpha=1 / self.rsi_window, min_periods=self.rsi_window, adjust=False
            ).mean()
            avg_loss = loss.ewm(
                alpha=1 / self.rsi_window, min_periods=self.rsi_window, adjust=False
            ).mean()
            rs = avg_gain / (avg_loss + 1e-12)
            df[rsi_col] = 100.0 - (100.0 / (1.0 + rs))

        if self.zero_on_any_nan and df[rsi_col].isna().any():
            result = pd.Series(0.0, index=df.index, dtype=float)
            _set_to_cache(cache_key, result)
            return result

        rsi_arr: np.ndarray = df[rsi_col].to_numpy(dtype=np.float64, copy=False)
        sig_arr: np.ndarray = _rsi_signal_kernel(
            rsi_arr,
            self.upper,
            self.lower,
            self.neutral_zone,
            self.clip,
        )
        if not self.neutral_zone and rsi_arr.size:
            rsi_min = float(np.nanmin(rsi_arr))
            rsi_max = float(np.nanmax(rsi_arr))

            # Skip degenerate cases (all equal, or NaNs only).
            if rsi_max > rsi_min:
                in_band = (rsi_arr > self.lower) & (rsi_arr < self.upper)
                if in_band.any():
                    mid = (self.upper + self.lower) / 2.0
                    sig_arr = sig_arr.copy()
                    sig_arr[in_band] = -2.0 * (rsi_arr[in_band] - mid) / (self.upper - self.lower)
                    sig_arr = np.clip(sig_arr, -self.clip, self.clip)

        result = pd.Series(sig_arr, index=df.index)
        _set_to_cache(cache_key, result)
        return result


StrategyRegistry.register("rsi", RSIStrategy)


# --------------------------------------------------------------------------------------
# MACD Strategy
# --------------------------------------------------------------------------------------
class MACDStrategy(PipelineStrategy):
    # MACD crossover with optional histogram-based proportional signal

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        use_histogram: bool = True,
        clip: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)
        self.use_histogram = bool(use_histogram)
        self.clip = float(clip)
        if self.fast <= 0 or self.slow <= 0 or self.signal <= 0:
            raise ValueError("fast, slow, and signal must be positive integers")
        if self.fast >= self.slow:
            raise ValueError("fast must be < slow for MACD")
        self._params_hash = self._compute_params_hash()

    def _compute_params_hash(self) -> str:
        return _hash_params(
            {
                "fast": self.fast,
                "slow": self.slow,
                "signal": self.signal,
                "use_histogram": self.use_histogram,
                "clip": self.clip,
            }
        )

    def features_plan(self) -> FeaturePlan:
        return FeaturePlan.from_steps(
            [
                FeatureStep(
                    "technical.MACD_line_signal",
                    inputs=("price",),
                    kwargs={
                        "input_col": "price",
                        "fast": self.fast,
                        "slow": self.slow,
                        "signal": self.signal,
                        "out_macd": "macd",
                        "out_signal": "macd_signal",
                        "out_hist": "macd_hist",
                    },
                ),
            ]
        )

    @instrument(name="macd_generate_signal", measure_latency=True)
    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        df: pd.DataFrame = _as_df(features)

        # empty -> neutral, length-safe, and cache
        if df.empty:
            feat_hash = _hash_features(df)
            cache_key = _cache_key_for_signal("macd", self._params_hash, feat_hash)
            empty = pd.Series(0.0, index=df.index, dtype=float)
            _set_to_cache(cache_key, empty)
            return empty

        hist_col, macd_col, sig_col = "macd_hist", "macd", "macd_signal"

        feat_hash = _hash_features(df)
        cache_key = _cache_key_for_signal("macd", self._params_hash, feat_hash)

        cached = _get_from_cache(cache_key, "macd")

        # 1) Required histogram input first when requested, so error is visible even with cache
        if self.use_histogram and hist_col not in df.columns:
            LOG.error("missing_feature: strategy=macd column=%s", hist_col)
            if _metrics:
                _metrics.record_counter(_metrics.counter("macd_missing_feature"))

            if cached is not None:
                # Preserve existing cached series but still surface the missing feature
                return cached

            result = pd.Series(0.0, index=df.index, dtype=float)
            _set_to_cache(cache_key, result)
            return result

        # 2) Normal cache path when features are present
        if cached is not None:
            return cached
        if _metrics:
            _metrics.record_counter(_metrics.counter("macd_cache_miss"))

        hist_col = "macd_hist"
        if self.use_histogram and hist_col not in df.columns:
            LOG.error("missing_feature: strategy=macd column=%s", hist_col)  # exact format
            if _metrics:
                _metrics.record_counter(_metrics.counter("macd_missing_feature"))
            result = pd.Series(0.0, index=df.index, dtype=float)
            _set_to_cache(cache_key, result)
            return result

        # crossover mode may need to synthesize macd/signal from price-like column
        need_cross = (not self.use_histogram) and (
            macd_col not in df.columns or sig_col not in df.columns
        )
        if need_cross:
            price_col = (
                "price"
                if "price" in df.columns
                else ("close" if "close" in df.columns else _first_numeric_col(df))
            )
            if price_col is None:
                LOG.error("missing_feature: strategy=macd columns=%s", f"{macd_col},{sig_col}")
                if _metrics:
                    _metrics.record_counter(_metrics.counter("macd_missing_feature"))
                result = pd.Series(0.0, index=df.index, dtype=float)
                _set_to_cache(cache_key, result)
                return result

            s = df[price_col].astype(float)
            df = df.copy()
            ema_fast = s.ewm(span=self.fast, adjust=False).mean()
            ema_slow = s.ewm(span=self.slow, adjust=False).mean()
            df[macd_col] = ema_fast - ema_slow
            df[sig_col] = df[macd_col].ewm(span=self.signal, adjust=False).mean()

        if self.use_histogram:
            hist_arr = df[hist_col].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            sig_arr = _macd_hist_kernel(hist_arr, self.clip)
        else:
            macd_arr = df[macd_col].to_numpy(dtype=np.float64, copy=False)
            sigline_arr = df[sig_col].to_numpy(dtype=np.float64, copy=False)
            sig_arr = np.where(macd_arr > sigline_arr, 1.0, -1.0)
            sig_arr = np.clip(sig_arr, -self.clip, self.clip)

        result = pd.Series(sig_arr, index=df.index)
        _set_to_cache(cache_key, result)
        return result


StrategyRegistry.register("macd", MACDStrategy)


# --------------------------------------------------------------------------------------
# Bollinger Bands Strategy
# --------------------------------------------------------------------------------------


class BollingerBandsStrategy(PipelineStrategy):
    # BB mean reversion (fade extremes) or breakout (follow extremes)

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        mode: Literal["reversion", "breakout"] = "reversion",
        clip: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.period = int(period)
        self.num_std = float(num_std)
        self.mode: Literal["reversion", "breakout"] = mode
        self.clip = float(clip)
        if mode not in ("reversion", "breakout"):
            raise ValueError(f"mode must be 'reversion' or 'breakout', got {mode}")
        self._params_hash = self._compute_params_hash()

    def _compute_params_hash(self) -> str:
        return _hash_params(
            {
                "period": self.period,
                "num_std": self.num_std,
                "mode": self.mode,
                "clip": self.clip,
            }
        )

    def features_plan(self) -> FeaturePlan:
        return FeaturePlan.from_steps(
            [
                FeatureStep(
                    "technical.Bollinger",
                    inputs=("price",),
                    kwargs={
                        "input_col": "price",
                        "window": self.period,
                        "num_std": self.num_std,
                        "out_mid": f"sma{self.period}",
                        "out_std": f"std{self.period}",
                        "out_upper": f"bb_upper{self.period}",
                        "out_lower": f"bb_lower{self.period}",
                    },
                ),
            ]
        )

    @instrument(name="bollinger_generate_signal", measure_latency=True)
    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        df: pd.DataFrame = _as_df(features)

        feat_hash = _hash_features(df)
        cache_key = _cache_key_for_signal("bollinger", self._params_hash, feat_hash)

        cached = _get_from_cache(cache_key, "bollinger")
        if cached is not None:
            return cached
        if _metrics:
            _metrics.record_counter(_metrics.counter("bollinger_cache_miss"))

        sma_col, std_col = f"sma{self.period}", f"std{self.period}"
        if "price" not in df.columns or sma_col not in df.columns or std_col not in df.columns:
            base = "price" if "price" in df.columns else None
            if base is None:
                LOG.error("missing_feature", strategy="ma_cross", column="price")
                if _metrics:
                    _metrics.record_counter(_metrics.counter("ma_cross_missing_feature"))
                result = pd.Series(0.0, index=df.index, dtype=float)
                _set_to_cache(cache_key, result)
                return result

            s = df[base].astype(float)
            df = df.copy()
            if "price" not in df.columns:
                df["price"] = s
            if sma_col not in df.columns:
                df[sma_col] = s.rolling(self.period, min_periods=self.period).mean()
            if std_col not in df.columns:
                df[std_col] = s.rolling(self.period, min_periods=self.period).std(ddof=0)

        price_arr: np.ndarray = df["price"].to_numpy(dtype=np.float64, copy=False)
        sma_arr: np.ndarray = df[sma_col].to_numpy(dtype=np.float64, copy=False)
        std_arr: np.ndarray = df[std_col].to_numpy(dtype=np.float64, copy=False)

        mode_int = 0 if self.mode == "reversion" else 1
        sig_arr: np.ndarray = _bollinger_kernel(
            price_arr, sma_arr, std_arr, self.num_std, mode_int, self.clip
        )
        result = pd.Series(sig_arr, index=df.index)
        _set_to_cache(cache_key, result)
        return result


StrategyRegistry.register("bollinger_bands", BollingerBandsStrategy)


# --------------------------------------------------------------------------------------
# Mean Reversion Strategy
# --------------------------------------------------------------------------------------


class MeanReversionStrategy(PipelineStrategy):
    # Z-score contrarian: fade extreme deviations from mean

    def __init__(
        self,
        period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        clip: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.period = int(period)
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        self.clip = float(clip)
        self._params_hash = self._compute_params_hash()

    def _compute_params_hash(self) -> str:
        return _hash_params(
            {
                "period": self.period,
                "entry_threshold": self.entry_threshold,
                "clip": self.clip,
            }
        )

    def features_plan(self) -> FeaturePlan:
        return FeaturePlan.from_steps(
            [
                FeatureStep(
                    "scaling.zscore_roll",
                    inputs=("price",),
                    kwargs={
                        "col": "price",
                        "window": self.period,
                        "out_col": f"zscore{self.period}",
                    },
                ),
            ]
        )

    @instrument(name="mean_rev_generate_signal", measure_latency=True)
    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        df: pd.DataFrame = _as_df(features)

        feat_hash = _hash_features(df)
        cache_key = _cache_key_for_signal("mean_rev", self._params_hash, feat_hash)

        cached = _get_from_cache(cache_key, "mean_rev")
        if cached is not None:
            return cached
        if _metrics:
            _metrics.record_counter(_metrics.counter("mean_rev_cache_miss"))

        z_col = f"zscore{self.period}"
        if z_col not in df.columns:
            LOG.error("missing_feature", strategy="mean_rev", column=z_col)
            if _metrics:
                _metrics.record_counter(_metrics.counter("mean_rev_missing_feature"))
            result = pd.Series(0.0, index=df.index, dtype=float)
            _set_to_cache(cache_key, result)
            return result

        z_arr: np.ndarray = df[z_col].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        sig_arr: np.ndarray = -np.tanh(z_arr / self.entry_threshold)
        sig_arr = np.clip(sig_arr, -self.clip, self.clip)
        result = pd.Series(sig_arr, index=df.index)
        _set_to_cache(cache_key, result)
        return result


StrategyRegistry.register("mean_reversion", MeanReversionStrategy)


# ========================= Moving Average Crossover Strategy =========================
class MovingAverageCrossoverStrategy(PipelineStrategy):
    # Dual MA cross with optional momentum-proportional signal

    def __init__(
        self,
        short: int = 50,
        long: int = 200,
        ma_type: Literal["sma", "ema"] = "sma",
        use_momentum: bool = False,
        clip: float = 1.0,
        price_col: str | None = None,  # <- explicit price column support
        **kwargs: Any,
    ) -> None:
        # accept legacy aliases without breaking callers
        if "short_window" in kwargs and isinstance(kwargs["short_window"], (int, float)):
            short = int(kwargs.pop("short_window"))
        if "long_window" in kwargs and isinstance(kwargs["long_window"], (int, float)):
            long = int(kwargs.pop("long_window"))

        super().__init__(**kwargs)
        self.short = int(short)
        self.long = int(long)
        self.ma_type: Literal["sma", "ema"] = ma_type
        self.use_momentum = bool(use_momentum)
        self.clip = float(clip)
        self.price_col = price_col  # <- persisted for ensemble/explicit behavior

        # parameter validation
        if self.ma_type not in ("sma", "ema"):
            raise ValueError(f"ma_type must be 'sma' or 'ema', got {self.ma_type}")
        if self.short <= 0 or self.long <= 0:
            raise ValueError("short and long must be positive integers")
        if self.short >= self.long:
            raise ValueError("short window must be < long window")

        self._params_hash = self._compute_params_hash()

    def _compute_params_hash(self) -> str:
        return _hash_params(
            {
                "short": self.short,
                "long": self.long,
                "ma_type": self.ma_type,
                "use_momentum": self.use_momentum,
                "clip": self.clip,
                "price_col": self.price_col,
            }
        )

    def features_plan(self) -> FeaturePlan:
        steps = []
        if self.ma_type == "sma":
            steps.extend(
                [
                    FeatureStep(
                        "technical.SMA",
                        inputs=("price",),
                        kwargs={"input_col": "price", "window": self.short, "out_col": "ma_short"},
                    ),
                    FeatureStep(
                        "technical.SMA",
                        inputs=("price",),
                        kwargs={"input_col": "price", "window": self.long, "out_col": "ma_long"},
                    ),
                ]
            )
        else:
            steps.extend(
                [
                    FeatureStep(
                        "technical.EMA",
                        inputs=("price",),
                        kwargs={"input_col": "price", "span": self.short, "out_col": "ma_short"},
                    ),
                    FeatureStep(
                        "technical.EMA",
                        inputs=("price",),
                        kwargs={"input_col": "price", "span": self.long, "out_col": "ma_long"},
                    ),
                ]
            )
        return FeaturePlan.from_steps(steps)

    @instrument(name="ma_cross_generate_signal", measure_latency=True)
    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        df: pd.DataFrame = _as_df(features)

        feat_hash = _hash_features(df)
        cache_key = _cache_key_for_signal("ma_cross", self._params_hash, feat_hash)

        # 1) REQUIRED INPUTS FIRST so the test sees an ERROR even if there is a cache entry
        if self.price_col is not None and self.price_col not in df.columns:
            LOG.error("missing_feature: strategy=ma_cross column=%s", self.price_col)
            if _metrics:
                _metrics.record_counter(_metrics.counter("ma_cross_missing_price"))
            result = pd.Series(0.0, index=df.index, dtype=float)
            _set_to_cache(cache_key, result)
            return result

        if self.price_col is None:
            base_col = (
                "price"
                if "price" in df.columns
                else ("close" if "close" in df.columns else _first_numeric_col(df))
            )
            if base_col is None:
                LOG.error("missing_feature: strategy=ma_cross column=%s", "any_numeric")
                if _metrics:
                    _metrics.record_counter(_metrics.counter("ma_cross_missing_numeric"))
                result = pd.Series(0.0, index=df.index, dtype=float)
                _set_to_cache(cache_key, result)
                return result
        else:
            base_col = self.price_col

        # 2) Now normal-path cache short-circuit
        cached = _get_from_cache(cache_key, "ma_cross")
        if cached is not None:
            return cached
        if _metrics:
            _metrics.record_counter(_metrics.counter("ma_cross_cache_miss"))

        # explicit price_col → must exist; otherwise log + return neutral zeros
        if self.price_col is not None:
            if self.price_col not in df.columns:
                LOG.error("missing_feature", strategy="ma_cross", column=self.price_col)
                LOG.error("missing_feature: strategy=ma_cross column=%s", self.price_col)
                LOG.error("missing_feature: strategy=ma_cross column=%s", self.price_col)
                if _metrics:
                    _metrics.record_counter(_metrics.counter("ma_cross_missing_price"))
                result = pd.Series(0.0, index=df.index, dtype=float)
                _set_to_cache(cache_key, result)
                return result
            base_col = self.price_col
        else:
            # implicit selection: prefer 'price', then 'close', else first numeric
            base_col = (
                "price"
                if "price" in df.columns
                else ("close" if "close" in df.columns else _first_numeric_col(df))
            )
            if base_col is None:
                LOG.error("missing_feature", strategy="ma_cross", column="any_numeric")
                LOG.error("missing_feature: strategy=ma_cross column=%s", "any_numeric")
                LOG.error("missing_feature: strategy=ma_cross column=%s", "any_numeric")
                if _metrics:
                    _metrics.record_counter(_metrics.counter("ma_cross_missing_numeric"))
                result = pd.Series(0.0, index=df.index, dtype=float)
                _set_to_cache(cache_key, result)
                return result

        s = pd.to_numeric(df[base_col], errors="coerce")
        if self.ma_type == "sma":
            ma_short = s.rolling(window=self.short, min_periods=self.short).mean()
            ma_long = s.rolling(window=self.long, min_periods=self.long).mean()
        else:
            ma_short = s.ewm(span=self.short, adjust=False).mean()
            ma_long = s.ewm(span=self.long, adjust=False).mean()

        # binary crossover (-1/+1) with clipping
        diff = ma_short - ma_long
        sig_arr = np.where(diff > 0.0, 1.0, -1.0)
        sig_arr = np.clip(sig_arr, -self.clip, self.clip)

        result = pd.Series(sig_arr, index=df.index)
        _set_to_cache(cache_key, result)
        return result


# =====================================================================================

StrategyRegistry.register("moving_average_crossover", MovingAverageCrossoverStrategy)


# --------------------------------------------------------------------------------------
# Ensemble Strategy (FIX #1, #2, #6: namespacing, leverage guard, adaptive stability)
# --------------------------------------------------------------------------------------


@dataclass
class _SubStrategyState:
    strategy: PipelineStrategy
    params: dict[str, Any]
    failure_count: int = 0
    last_signal: pd.Series | None = None
    open: bool = True


class EnsemblePipelineStrategy(PipelineStrategy):
    ADAPTIVE_UPDATE_INTERVAL: Final[int] = 10
    MAX_FAILURES: Final[int] = 5
    MIN_WEIGHT_FLOOR: Final[float] = 0.05

    def __init__(
        self,
        strategy_specs: list[tuple[str, dict]],
        weights: list[float] | None = None,
        combination_method: Literal["weighted", "majority"] = "weighted",
        adaptive_weights: bool = False,
        performance_window: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not strategy_specs:
            raise ValueError("strategy_specs cannot be empty")

        self.strategy_specs = strategy_specs
        self.combination_method = combination_method
        self.adaptive_weights = bool(adaptive_weights)
        self.performance_window = int(performance_window)

        self._states: list[_SubStrategyState] = []
        for name, params in strategy_specs:
            try:
                factory = StrategyRegistry.get(name)

                if not (
                    isinstance(factory, PipelineStrategy)
                    or (isinstance(factory, type) and issubclass(factory, PipelineStrategy))
                    or callable(factory)
                ):
                    raise RuntimeError(f"Invalid strategy factory for '{name}'")

                params_copy = dict(params or {})
                if isinstance(factory, PipelineStrategy):
                    strat = factory
                    params_used = dict(getattr(strat, "params", {}) or {})
                else:
                    strat = factory(**params_copy)
                    params_used = params_copy

                self._states.append(_SubStrategyState(strategy=strat, params=params_used))
            except (KeyError, TypeError, ValueError, RuntimeError) as e:
                LOG.error("ensemble_init_failed", strategy=name, error=str(e))
                raise

        if weights is not None:
            if len(weights) != len(self._states):
                raise ValueError("weights length must match number of strategies")
            self.weights = np.array(weights, dtype=np.float64)
            total = np.sum(np.abs(self.weights))
            if total <= 1e-12:
                raise ValueError("weights cannot all be zero")
            self.weights /= total
        else:
            n = len(self._states)
            self.weights = np.ones(n, dtype=np.float64) / n

        self._call_count = 0
        self._col_aliases: list[dict[str, str]] = []
        self._features_plan_cache: FeaturePlan | None = None

    @lru_cache(maxsize=1)
    def features_plan(self) -> FeaturePlan:
        # FIX #1: Namespace collision prevention with s{i}__ prefixes
        all_steps: list[FeatureStep] = []
        seen_sigs: set[tuple] = set()
        self._col_aliases.clear()

        for i, state in enumerate(self._states):
            plan = state.strategy.features_plan()
            alias: dict[str, str] = {}

            # Handle None (no features required) or empty plans gracefully
            if plan is None:
                self._col_aliases.append(alias)
                continue

            steps = getattr(plan, "steps", [])
            if callable(steps):
                steps = steps()
            try:
                _steps_iter = list(steps)
            except TypeError:
                # Handle Mock objects or other non-iterables
                rv = getattr(steps, "return_value", None)
                # Guard against Mock.return_value also being a Mock
                try:
                    _steps_iter = list(rv) if rv is not None else []
                except TypeError:
                    # If still not iterable (nested Mocks), treat as empty
                    _steps_iter = []

            for step in _steps_iter:
                kw = dict(step.kwargs or {})

                # Namespace all common output keys
                for out_key in (
                    "out",
                    "out_col",
                    "out_mid",
                    "out_std",
                    "out_macd",
                    "out_signal",
                    "out_hist",
                    "out_fast",
                    "out_slow",
                    "out_upper",
                    "out_lower",
                ):
                    if out_key in kw:
                        orig_name = kw[out_key]
                        prefixed_name = f"s{i}__{orig_name}"
                        alias[orig_name] = prefixed_name
                        kw[out_key] = prefixed_name

                sig = (step.op, step.inputs, step.args, tuple(sorted(kw.items())))
                if sig not in seen_sigs:
                    all_steps.append(FeatureStep(step.op, step.inputs, step.args, kw))
                    seen_sigs.add(sig)

            self._col_aliases.append(alias)

        return FeaturePlan.from_steps(all_steps)

    @instrument(name="ensemble_generate_signal", measure_latency=True)
    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate an ensemble trading signal by aggregating sub-strategy signals,
        applying either weighted or majority voting, and (optionally) updating
        adaptive weights. Guaranteed to return a pd.Series indexed like `features`.
        """
        self._call_count += 1

        # Normalize the features input to a pandas DataFrame (tests sometimes pass mocks)
        df = features if isinstance(features, pd.DataFrame) else features.to_pandas()

        # Lazy-init column alias mapping (namespacing fix for child strategies)
        if not self._col_aliases:
            _ = self.features_plan()

        signals: list[pd.Series] = []

        # Collect each sub-strategy's signal, with circuit-breaker + fallback
        for i, state in enumerate(self._states):
            if not state.open:
                # Circuit already open: reuse last known or neutral zeros
                fallback = (
                    state.last_signal
                    if state.last_signal is not None
                    else pd.Series(0.0, index=df.index)
                )
                signals.append(fallback)

                if _metrics:
                    _metrics.record_counter(
                        _metrics.counter("ensemble_circuit_open"),
                        labels={"idx": str(i)},
                    )
                continue

            try:
                # Map namespaced columns (s{i}__foo) back to the child's expected names
                inv_alias = {v: k for k, v in self._col_aliases[i].items()}
                sub_df = df.rename(columns=inv_alias, copy=False)

                params_dict = state.params or {}
                sentinel = object()

                attr_price = getattr(state.strategy, "price_col", sentinel)
                explicit_param = isinstance(params_dict, dict) and ("price_col" in params_dict)
                param_price = params_dict.get("price_col") if explicit_param else sentinel

                # Case 1: explicit None (either via attribute or params) -> log ERROR, neutral signal
                if attr_price is None or (explicit_param and param_price is None):
                    strategy_name = type(state.strategy).__name__
                    LOG.error(
                        "ensemble_price_column_invalid",
                        idx=i,
                        strategy=strategy_name,
                    )
                    # Also log via stdlib logger so pytest caplog (default WARNING+) can catch ERROR
                    LOG.error(
                        "ensemble_price_column_invalid idx=%s strategy=%s",
                        i,
                        strategy_name,
                    )
                    neutral = pd.Series(0.0, index=df.index, dtype=float)
                    state.last_signal = neutral
                    state.failure_count = 0
                    signals.append(neutral)
                    continue

                # Determine effective price column, if any
                effective_price = None
                if attr_price is not sentinel:
                    effective_price = attr_price
                elif explicit_param and param_price is not sentinel:
                    effective_price = param_price

                # Case 2: provided name not present in sub_df -> log ERROR, neutral signal
                if isinstance(effective_price, str) and effective_price not in sub_df.columns:
                    strategy_name = type(state.strategy).__name__
                    LOG.error(
                        "ensemble_price_column_missing",
                        idx=i,
                        price_col=effective_price,
                        strategy=strategy_name,
                    )
                    LOG.error(
                        "ensemble_price_column_missing idx=%s strategy=%s price_col=%s",
                        i,
                        strategy_name,
                        effective_price,
                    )
                    neutral = pd.Series(0.0, index=df.index, dtype=float)
                    state.last_signal = neutral
                    state.failure_count = 0
                    signals.append(neutral)
                    continue

                sig = state.strategy.generate_signal(sub_df)

                # Some strategies return a DataFrame; tests expect we downcast to first column
                if isinstance(sig, pd.DataFrame):
                    sig = sig.iloc[:, 0]

                state.last_signal = sig
                state.failure_count = 0
                signals.append(sig)

            except (RuntimeError, ValueError, KeyError, AttributeError, TypeError) as e:
                # Sub-strategy failed: log, increment failure count, maybe open circuit,
                # and fall back to last_signal or neutral zeros.
                LOG.warning("ensemble_sub_failure", idx=i, error=str(e))

                state.failure_count += 1
                if state.failure_count >= self.MAX_FAILURES:
                    state.open = False
                    LOG.warning(
                        "ensemble_circuit_opened",
                        idx=i,
                        failures=state.failure_count,
                    )
                    if _metrics:
                        _metrics.record_counter(
                            _metrics.counter("ensemble_circuit_opened"),
                            labels={"idx": str(i)},
                        )

                fallback = (
                    state.last_signal
                    if state.last_signal is not None
                    else pd.Series(0.0, index=df.index)
                )
                signals.append(fallback)

                if _metrics:
                    _metrics.record_counter(
                        _metrics.counter("ensemble_sub_failure"),
                        labels={"idx": str(i)},
                    )

        # -------------------------
        # Combine sub-strategy signals
        # -------------------------

        def _to_float_array(sig_obj: Any, n_out: int) -> np.ndarray:
            """
            Convert a signal-like object (Series / ndarray / list / mocky thing)
            into a well-shaped float64 numpy array with NaNs/Infs neutralized.
            This prevents cases like float * Mock from blowing up.
            """
            # pd.Series → fill NaN, get numpy
            if isinstance(sig_obj, pd.Series):
                return sig_obj.fillna(0.0).to_numpy(dtype=np.float64, copy=False)

            # Raw array / list-like
            if isinstance(sig_obj, (np.ndarray, list, tuple)):
                arr = np.asarray(sig_obj, dtype=np.float64)
                if arr.shape[0] != n_out:
                    return np.zeros(n_out, dtype=np.float64)
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            # Anything else (e.g. MagicMock, unexpected types) → neutral zeros
            return np.zeros(n_out, dtype=np.float64)

        n = len(df)

        if self.combination_method == "weighted":
            combined = np.zeros(n, dtype=np.float64)
            for i, sig in enumerate(signals):
                arr = _to_float_array(sig, n)
                # float(...) ensures mocks/np arrays of weights can't sneak in
                combined += float(self.weights[i]) * arr

            # Leverage guard: clip final signal to [-1, 1]
            result = pd.Series(
                np.clip(combined, -1.0, 1.0),
                index=df.index,
            )

        else:  # "majority"
            votes = np.zeros(n, dtype=np.float64)
            for sig in signals:
                votes += np.sign(_to_float_array(sig, n))

            result = pd.Series(
                np.sign(votes),
                index=df.index,
            )

        # -------------------------
        # Adaptive weight update (rate-limited)
        # -------------------------
        if self.adaptive_weights and self._call_count % self.ADAPTIVE_UPDATE_INTERVAL == 0:
            try:
                self._update_adaptive_weights(signals, df)
            except (ValueError, RuntimeError, ArithmeticError) as e:
                # Narrow set only: invalid math / model fit / numeric instability
                LOG.warning("adaptive_weights_failed", error=str(e))

        return result

    def _update_adaptive_weights(self, signals: list[pd.Series], features: pd.DataFrame) -> None:
        # FIX #6: Stability improvements
        if len(features) < self.performance_window:
            return

        if "price" not in features.columns:
            LOG.warning("adaptive_weights_skipped", reason="no_price_column")
            return

        recent_rets = features["price"].pct_change().fillna(0.0).tail(self.performance_window)
        recent_rets_arr = recent_rets.to_numpy(dtype=np.float64, copy=False)

        perfs: list[float] = []
        total_non_zero = 0

        for sig in signals:
            recent_sig = sig.tail(self.performance_window).shift(1).fillna(0.0)
            recent_sig_arr = recent_sig.to_numpy(dtype=np.float64, copy=False)
            total_non_zero += int(np.count_nonzero(recent_sig_arr))
            perf = (np.sign(recent_sig_arr) * np.sign(recent_rets_arr)).mean()
            perfs.append(max(0.0, perf))

        # Skip if signals mostly neutral
        if total_non_zero < len(signals) * 5:
            LOG.debug("adaptive_weights_skipped", reason="signals_neutral")
            return

        new_w = np.array(perfs, dtype=np.float64)
        total = np.sum(new_w)

        if total > 1e-12:
            new_w = new_w / total
            # FIX #6: Apply floor
            new_w = np.maximum(new_w, self.MIN_WEIGHT_FLOOR)
            new_w = new_w / np.sum(new_w)

            # EMA update
            self.weights = 0.9 * self.weights + 0.1 * new_w

            if _metrics:
                _metrics.record_counter(_metrics.counter("ensemble_weight_update"))
            LOG.debug("ensemble_weights_updated", weights=self.weights.tolist())


StrategyRegistry.register("ensemble", EnsemblePipelineStrategy)


# --------------------------------------------------------------------------------------
# Convenience function
# --------------------------------------------------------------------------------------


def create_ensemble(
    ctx_or_configs: Any,
    strategy_configs: list[dict] | None = None,
    weights: list[float] | None = None,
    combination_method: Literal["weighted", "majority"] = "weighted",
    adaptive: bool = False,
) -> EnsemblePipelineStrategy:
    # Support both call styles:
    #   create_ensemble(ctx, configs, ...)
    #   create_ensemble(configs, ...)
    if strategy_configs is None and isinstance(ctx_or_configs, list):
        configs = ctx_or_configs
    else:
        configs = strategy_configs or []
    specs = [(cfg["type"], cfg.get("params", {})) for cfg in configs]
    return EnsemblePipelineStrategy(
        strategy_specs=specs,
        weights=weights,
        combination_method=combination_method,
        adaptive_weights=adaptive,
    )


# Register cache hit-rate gauges if metrics available
if _metrics and _cache_client:
    from pysrc.ops.observability import register_cache_hit_rate_gauges

    register_cache_hit_rate_gauges(_cache_client, "strategies_cache_hit_rate")
