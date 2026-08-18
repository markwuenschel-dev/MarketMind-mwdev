"""Pipeline strategy IR, feature materialization, and strategy registry.

Stages (Programming Guidelines §3.6): specification → ``FeaturePlan`` IR →
materialization (``materialize_features``) → signal / trade intent → downstream
artifacts. Point-in-time and provenance enter through ``StrategyContext.pit_provenance``
(§4.3). The imperative shell coordinates I/O and caching; deterministic transforms
live in the functional core paths behind ``feature_op`` / graph ops (§4.1).

See ``docs/Programming Guidelines.md`` v3.0 for the normative engineering model.
"""

from __future__ import annotations

import concurrent.futures as _fut
import dataclasses
import hashlib
import importlib
import inspect
import itertools
import json
import pickle
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Final,
    Protocol,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
import pandas as pd

from pysrc.backtesting.contracts.types import PitMeta

# Optional deps must never break import-time
if TYPE_CHECKING:
    from polars import DataFrame as PolarsDataFrame
    from polars import Series as PolarsSeries

    # bridge: AlphaIR migrates to contracts/ at Phase II (AQ-07)
    from pysrc.strategies.momentum.alpha_ir import AlphaIR
else:
    PolarsDataFrame = Any
    PolarsSeries = Any

OptunaTrial = Any

try:
    optuna: Any | None = importlib.import_module("optuna")
except (ImportError, ModuleNotFoundError):
    optuna = None

try:
    pl: Any | None = importlib.import_module("polars")
except (ImportError, ModuleNotFoundError):
    pl = None

import contextlib

from pysrc.ops.mm_logkit import get_logger

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
LOG = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Errors / Result types
# --------------------------------------------------------------------------------------
class PipelineError(Exception):
    pass


class ValidationError(PipelineError):
    pass


class MaterializationError(PipelineError):
    pass


@dataclass
class TradeIntent:
    # weights: index aligned with prices (Series for single-asset, DataFrame for multi)
    weights: pd.Series | pd.DataFrame
    raw: Mapping[str, object] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)


SignalOutput = Union[pd.Series, pd.DataFrame]


class SignalEnvelope(Protocol):
    @property
    def signal(self) -> SignalOutput: ...


StrategySignal = Union[SignalOutput, SignalEnvelope, "AlphaIR"]


# --------------------------------------------------------------------------------------
# Backend types and small infra
# --------------------------------------------------------------------------------------
@dataclass
class StrategyContext:
    prices: pd.Series | pd.DataFrame
    features: pd.DataFrame | PolarsDataFrame | None = None
    timestamps: pd.Index | None = None
    asset_names: list[str] | None = None
    backend: str = "pandas"  # "pandas" | "polars"
    cache_dir: str | Path = ".cache_pipeline"
    random_state: int = 1337
    pit_provenance: PitMeta | None = None

    def validate(self) -> StrategyContext:
        if not isinstance(self.prices, (pd.Series, pd.DataFrame)):
            raise ValidationError("prices must be a pandas Series or DataFrame")

        if self.timestamps is not None and not isinstance(self.timestamps, pd.Index):
            raise ValidationError("timestamps must be a pandas Index")
        elif self.timestamps is None:
            idx = self.prices.index if isinstance(self.prices, (pd.Series, pd.DataFrame)) else None
            if not isinstance(idx, pd.Index):
                raise ValidationError("timestamps must be a pandas Index")

        if self.backend not in ("pandas", "polars"):
            raise ValidationError("backend must be 'pandas' or 'polars'")
        if self.backend == "polars" and globals().get("pl", None) is None:
            raise ValidationError("polars is not installed")

        if isinstance(self.prices, pd.DataFrame):
            self.asset_names = list(self.prices.columns)
        else:
            self.asset_names = ["asset"]
        return self

    def _cache_dir_path(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


# --------------------------------------------------------------------------------------
# Protocols for plug-ins
# --------------------------------------------------------------------------------------
@runtime_checkable
class RegimeDetector(Protocol):
    def gate(
        self, features: pd.DataFrame | PolarsDataFrame
    ) -> pd.Series | PolarsSeries | float | int: ...  # returns multiplier or 0/1 mask


@runtime_checkable
class RiskManager(Protocol):
    def clamp(
        self, weights: pd.Series | pd.DataFrame, prices: pd.Series | pd.DataFrame, **kwargs: Any
    ) -> pd.Series | pd.DataFrame: ...


@runtime_checkable
class PositionSizer(Protocol):
    def size(self, signal: pd.Series | pd.DataFrame, **kwargs: Any) -> pd.Series | pd.DataFrame: ...


# --------------------------------------------------------------------------------------
# Feature plan definition and registry
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureStep:
    op: str
    inputs: tuple[str, ...]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    out: str = ""


# Bump when built-in ``feature_op`` table semantics change (cache keys; §5.2).
FEATURE_OP_REGISTRY_VERSION: Final[str] = "1.0.0"


def feature_op_registry_version() -> str:
    """Semantic version string for the legacy ``_FEATURE_OPS`` registry (§3.2, §5.2)."""
    return FEATURE_OP_REGISTRY_VERSION


@dataclass(frozen=True)
class FeaturePlan:
    """Immutable ordered feature IR (Programming Guidelines §4.5)."""

    steps: tuple[FeatureStep, ...]

    @classmethod
    def from_steps(cls, steps: Sequence[FeatureStep]) -> FeaturePlan:
        """Build a plan from any ordered sequence of steps."""
        return cls(tuple(steps))

    def signature(self) -> str:
        """Non-cryptographic plan fingerprint for caches and debug (§7.2)."""
        blob = json.dumps([dataclasses.asdict(s) for s in self.steps], sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()


# Registry for feature operations
_FEATURE_OPS: dict[str, Callable[..., Any]] = {}


def feature_op(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _FEATURE_OPS[name] = fn
        return fn

    return deco


# --- Minimal, efficient ops for pandas (and polars when available) --------------------
@feature_op("PCT_CHANGE")
def op_pct_change(
    df: pd.DataFrame | PolarsDataFrame,
    col: str,
    periods: int = 1,
    out: str | None = None,
) -> pd.DataFrame | PolarsDataFrame:
    if isinstance(df, pd.DataFrame):
        pct_change = df[col].pct_change(periods)
        df[out or f"{col}_ret{periods}"] = pct_change
        return df
    if pl and isinstance(df, pl.DataFrame):
        pct_change_expr = df[col].pct_change(n=periods)
        return cast(
            PolarsDataFrame, df.with_columns(pct_change_expr.alias(out or f"{col}_ret{periods}"))
        )
    raise MaterializationError("Unsupported backend for PCT_CHANGE")


@feature_op("ROLL_MEAN")
def op_roll_mean(
    df: pd.DataFrame | PolarsDataFrame,
    col: str,
    window: int,
    minp: int | None = None,
    out: str | None = None,
) -> pd.DataFrame | PolarsDataFrame:
    # Ensure rolling statistics are defined for variance checks (ddof=1 needs n>=2)
    minp_eff = 2 if minp is None else max(1, int(minp))
    if minp_eff > window:
        minp_eff = window

    if isinstance(df, pd.DataFrame):
        s = df[col].rolling(window, min_periods=minp_eff).mean()
        df[out or f"{col}_sma{window}"] = s
        return df
    if pl and isinstance(df, pl.DataFrame):
        # polars>=1.21 renamed `min_periods`→`min_samples`; prefer new arg but fall back for older versions
        expr = cast(Any, pl.col(col)).rolling_mean(window_size=window)
        try:
            expr = cast(Any, expr).rolling_mean(window_size=window, min_samples=minp_eff)
        except TypeError:
            expr = cast(Any, pl.col(col)).rolling_mean(window_size=window, min_periods=minp_eff)
        return cast(PolarsDataFrame, df.with_columns(expr.alias(out or f"{col}_sma{window}")))
    raise MaterializationError("Unsupported backend for ROLL_MEAN")


@feature_op("ROLL_STD")
def op_roll_std(
    df: pd.DataFrame | PolarsDataFrame,
    col: str,
    window: int,
    minp: int | None = None,
    out: str | None = None,
) -> pd.DataFrame | PolarsDataFrame:
    # default to full-window to avoid early zero-variance z-scores; allow override
    minp_eff = window if minp is None else max(1, int(minp))
    if isinstance(df, pd.DataFrame):
        s = df[col].rolling(window, min_periods=minp_eff).std(ddof=0)
        df[out or f"{col}_std{window}"] = s
        return df
    if pl and isinstance(df, pl.DataFrame):
        # prefer `min_samples`; fall back to `min_periods` for older Polars
        try:
            expr = cast(Any, pl.col(col)).rolling_std(window_size=window, min_samples=minp_eff)
        except TypeError:
            expr = cast(Any, pl.col(col)).rolling_std(window_size=window, min_periods=minp_eff)
        return cast(PolarsDataFrame, df.with_columns(expr.alias(out or f"{col}_std{window}")))
    raise MaterializationError("Unsupported backend for ROLL_STD")


@feature_op("EMA")
def op_ema(
    df: pd.DataFrame | PolarsDataFrame,
    col: str,
    span: int,
    adjust: bool = False,
    out: str | None = None,
) -> pd.DataFrame | PolarsDataFrame:
    if isinstance(df, pd.DataFrame):
        s = df[col].ewm(span=span, adjust=adjust).mean()
        df[out or f"{col}_ema{span}"] = s
        return df
    if pl and isinstance(df, pl.DataFrame):
        # Approximate EMA via exponential smoothing (no built-in exact equivalent)
        vals = df[col].to_numpy()
        alpha = 2.0 / (span + 1)
        # Build a nullable result: all-None when no finite seeds exist (so polars .is_null() is True)
        if not np.isfinite(vals).any():
            ser = pl.Series(out or f"{col}_ema{span}", [None] * len(vals))
            return cast(PolarsDataFrame, df.with_columns(ser))
        ema = np.empty_like(vals, dtype=float)
        ema[:] = np.nan
        m = np.isfinite(vals)
        first = int(np.argmax(m))
        acc = float(vals[first])
        ema[first] = acc
        for i in range(first + 1, len(vals)):
            v = vals[i]
            if np.isnan(v):
                # forward-fill gaps; keeps column non-null for intermittent NaNs
                ema[i] = ema[i - 1]
            else:
                acc = alpha * v + (1.0 - alpha) * acc
                ema[i] = acc
        # Convert NaN sentinels to nulls only if the entire column is NaN-free case is false;
        # here we only had intermittent gaps, so keep as numeric with no nulls.
        return cast(PolarsDataFrame, df.with_columns(pl.Series(out or f"{col}_ema{span}", ema)))
    raise MaterializationError("Unsupported backend for EMA")


@feature_op("Z_SCORE")
def op_zscore(
    df: pd.DataFrame | PolarsDataFrame,
    col: str,
    window: int,
    minp: int | None = None,
    out: str | None = None,
) -> pd.DataFrame | PolarsDataFrame:
    minp = minp or window
    if isinstance(df, pd.DataFrame):
        r = df[col]
        mu = r.rolling(window, min_periods=minp).mean()
        sd = r.rolling(window, min_periods=minp).std(ddof=0)
        z = (r - mu) / sd.replace(0.0, np.nan)
        df[out or f"{col}_z{window}"] = z
        return df
    if pl and isinstance(df, pl.DataFrame):
        # prefer `min_samples`; fall back to `min_periods` for older Polars
        try:
            mu = cast(Any, pl.col(col)).rolling_mean(window_size=window, min_samples=minp)
            sd = cast(Any, pl.col(col)).rolling_std(window_size=window, min_samples=minp)
        except TypeError:
            mu = cast(Any, pl.col(col)).rolling_mean(window_size=window, min_periods=minp)
            sd = cast(Any, pl.col(col)).rolling_std(window_size=window, min_periods=minp)
        z = (pl.col(col) - mu) / sd
        return cast(PolarsDataFrame, df.with_columns(z.alias(out or f"{col}_z{window}")))
    raise MaterializationError("Unsupported backend for Z_SCORE")


# --- TA feature ops usable by any strategy ---------------------------------
@feature_op("RSI")
def _op_rsi(df: pd.DataFrame, col: str, window: int = 14, out: str | None = None) -> pd.DataFrame:
    price = df[col].astype(float)
    delta = price.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    alpha = 1.0 / max(1, window)
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df[out or f"{col}_rsi{window}"] = rsi
    return df


@feature_op("MACD")
def _op_macd(
    df: pd.DataFrame,
    col: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    out_fast: str | None = None,
    out_slow: str | None = None,
    out_macd: str | None = None,
    out_signal: str | None = None,
    out_hist: str | None = None,
) -> pd.DataFrame:
    price = df[col].astype(float)
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    df[out_fast or f"{col}_ema{fast}"] = ema_fast
    df[out_slow or f"{col}_ema{slow}"] = ema_slow
    df[out_macd or "macd"] = macd
    df[out_signal or "macd_signal"] = sig
    df[out_hist or "macd_hist"] = hist
    return df


@feature_op("ADX")
def _op_adx(
    df: pd.DataFrame,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    window: int = 14,
    out: str | None = None,
) -> pd.DataFrame:
    cols = set(df.columns)
    if not {high, low, close}.issubset(cols):
        df[out or f"ADX{window}"] = 25.0
        return df
    H, L, C = df[high].astype(float), df[low].astype(float), df[close].astype(float)
    plus_dm = (H.diff()).clip(lower=0.0)
    minus_dm = (-L.diff()).clip(upper=0.0).abs()
    minus_dm = (-L.diff()).clip(lower=0.0)
    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm < plus_dm] = 0.0
    tr = pd.concat([(H - L), (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    alpha = 1.0 / max(1, window)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-12))
    minus_di = 100.0 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-12))
    dx = (100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)).fillna(0.0)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    df[out or f"ADX{window}"] = adx
    return df


# --------------------------------------------------------------------------------------
# Materializer with caching and code-hash invalidation
# --------------------------------------------------------------------------------------
class _Cache:
    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        with self._lock:
            if not p.exists():
                return None
            try:
                with p.open("rb") as fh:
                    return pickle.load(fh)
            except (
                pickle.PickleError,
                EOFError,
                OSError,
                ImportError,
                AttributeError,
                ModuleNotFoundError,
            ):
                with contextlib.suppress(OSError):
                    p.unlink(missing_ok=True)
                return None

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        with self._lock, p.open("wb") as fh:
            sink = cast(BinaryIO, fh)  # precise runtime-safe type; no _typeshed
            try:
                pickle.dump(value, sink, protocol=pickle.HIGHEST_PROTOCOL)
            except (pickle.PicklingError, TypeError) as e:
                # Some backends (notably certain pandas builds) attach local
                # functions to indexes/blocks that are not picklable in all
                # environments. Cache failures must never break materialization
                # itself, so we treat this as a best-effort and skip caching.
                LOG.warning(
                    "pipeline_cache_pickle_failed",
                    extra={
                        "error": str(e),
                        "value_type": type(value).__name__,
                        "cache_key": key,
                    },
                )
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass


def _code_hash(obj: Any) -> str:
    try:
        src = inspect.getsource(obj)  # best-effort
    except (OSError, TypeError):
        src = repr(obj)
    return hashlib.sha1(src.encode()).hexdigest()


def _code_hash_type(cls: Any) -> str:
    ver = getattr(cls, "OP_VERSION", None)
    if ver is not None:
        return f"ver:{ver}"
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        src = repr(cls)
    return hashlib.sha1(src.encode()).hexdigest()


def _registry_state() -> tuple[Callable[[str], str], Mapping[str, Any], Mapping[str, str]]:
    try:
        factory = importlib.import_module("pysrc.preprocessor.graph.factory")
    except ImportError:
        registry: Mapping[str, Any] = {}
        alias_map: Mapping[str, str] = {}

        def resolve_name(name: str) -> str:
            return name

        return resolve_name, registry, alias_map

    resolve_name_fn = cast(Callable[[str], str], factory.resolve_name)
    registry = cast(Mapping[str, Any], factory._OP_REGISTRY)
    alias_map = cast(Mapping[str, str], factory._ALIAS_MAP)
    return resolve_name_fn, registry, alias_map


_LEGACY_CANONICAL_NAMES: dict[str, str] = {
    "ROLL_MEAN": "technical.SMA",
    "ROLL_STD": "stats.rolling_std",
    "Z_SCORE": "scaling.zscore_roll",
    "EMA": "technical.EMA",
    "RSI": "technical.RSI",
    "MACD": "technical.MACD_line_signal",
}

_CANONICAL_GRAPH_OPS: set[str] = {
    *set(_LEGACY_CANONICAL_NAMES.values()),
    "technical.Bollinger",
    "technical.ATR",
    "technical.OBV",
    "technical.VWAP",
}


@dataclass(frozen=True)
class _LoweredGraphStep:
    op: str
    params: Mapping[str, Any]


def _is_governed_context(ctx: StrategyContext) -> bool:
    return ctx.pit_provenance is not None


def _resolve_feature_op(step: FeatureStep) -> tuple[str | None, str | None]:
    # ADR-001 two-stage lookup is unchanged: Stage 1 checks _OP_REGISTRY by resolved name,
    # Stage 2 checks _FEATURE_OPS by the raw authoring token.
    resolve_name, op_registry, _ = _registry_state()
    resolved = resolve_name(step.op)
    if resolved in op_registry:
        return ("op_registry", resolved)
    fn = _FEATURE_OPS.get(step.op)
    if fn is not None:
        return ("feature_ops", step.op)
    return (None, None)


def _unknown_op_message(op: str) -> str:
    resolve_name, op_registry, _ = _registry_state()
    resolved = resolve_name(op)
    graph_keys = sorted(op_registry.keys()) if op_registry else []
    legacy_keys = sorted(_FEATURE_OPS.keys()) if _FEATURE_OPS else []
    suggestions: list[str] = []
    for key in graph_keys[:5]:
        suggestions.append(key)
    for key in legacy_keys[:5]:
        if key not in suggestions:
            suggestions.append(key)
    suffix = f" Checked _OP_REGISTRY (resolved name '{resolved}'), _FEATURE_OPS."
    if suggestions:
        suffix += f" Known ops (sample): {', '.join(suggestions)}."
    return f"Unknown feature op: '{op}'." + suffix


def _coalesce_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _coalesce_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int):
            return value
    return None


def _lower_feature_step(step: FeatureStep, resolved_key: str) -> _LoweredGraphStep:
    kw = dict(step.kwargs)
    if getattr(step, "out", ""):
        kw.setdefault("out", step.out)

    def require_str(value: str | None, label: str) -> str:
        if value is None:
            raise MaterializationError(f"{resolved_key} requires {label}")
        return value

    def require_int(value: int | None, label: str) -> int:
        if value is None:
            raise MaterializationError(f"{resolved_key} requires {label}")
        return value

    def primary_input() -> str:
        return require_str(
            _coalesce_str(
                kw.get("input_col"),
                kw.get("col"),
                kw.get("column"),
                *(step.inputs[:1] or (None,)),
            ),
            "an input column",
        )

    if resolved_key == "technical.SMA":
        input_col = primary_input()
        window = require_int(_coalesce_int(kw.get("window"), *(step.args[:1] or (None,))), "window")
        min_samples = kw.get("min_samples")
        if min_samples is None:
            minp = kw.get("minp")
            min_samples = int(minp) if isinstance(minp, int) else min(2, window)
        return _LoweredGraphStep(
            op=resolved_key,
            params={
                "input_col": input_col,
                "window": window,
                "min_samples": int(min_samples),
                "out_col": _coalesce_str(kw.get("out_col"), kw.get("out"))
                or f"{input_col}_sma{window}",
            },
        )

    if resolved_key == "stats.rolling_std":
        input_col = primary_input()
        window = require_int(_coalesce_int(kw.get("window"), *(step.args[:1] or (None,))), "window")
        min_samples = kw.get("min_samples")
        if min_samples is None:
            minp = kw.get("minp")
            min_samples = window if minp is None else int(minp)
        return _LoweredGraphStep(
            op=resolved_key,
            params={
                "col": input_col,
                "window": window,
                "min_samples": int(min_samples),
                "out_col": _coalesce_str(kw.get("out_col"), kw.get("out"))
                or f"{input_col}_std{window}",
            },
        )

    if resolved_key == "scaling.zscore_roll":
        input_col = primary_input()
        window = require_int(_coalesce_int(kw.get("window"), *(step.args[:1] or (None,))), "window")
        min_samples = kw.get("min_samples")
        if min_samples is None:
            minp = kw.get("minp")
            min_samples = window if minp is None else int(minp)
        return _LoweredGraphStep(
            op=resolved_key,
            params={
                "col": input_col,
                "window": window,
                "min_samples": int(min_samples),
                "out_col": _coalesce_str(kw.get("out_col"), kw.get("out"))
                or f"{input_col}_z{window}",
            },
        )

    if resolved_key == "technical.EMA":
        input_col = primary_input()
        span = require_int(_coalesce_int(kw.get("span"), *(step.args[:1] or (None,))), "span")
        return _LoweredGraphStep(
            op=resolved_key,
            params={
                "input_col": input_col,
                "span": span,
                "adjust": bool(kw.get("adjust", False)),
                "out_col": _coalesce_str(kw.get("out_col"), kw.get("out"))
                or f"{input_col}_ema{span}",
            },
        )

    if resolved_key == "technical.RSI":
        input_col = primary_input()
        window = _coalesce_int(kw.get("window"), *(step.args[:1] or (None,))) or 14
        return _LoweredGraphStep(
            op=resolved_key,
            params={
                "input_col": input_col,
                "window": window,
                "out_col": _coalesce_str(kw.get("out_col"), kw.get("out")) or "rsi",
            },
        )

    if resolved_key == "technical.MACD_line_signal":
        input_col = primary_input()
        fast = _coalesce_int(kw.get("fast"), *(step.args[0:1] or (None,))) or 12
        slow = _coalesce_int(kw.get("slow"), *(step.args[1:2] or (None,))) or 26
        signal = _coalesce_int(kw.get("signal"), *(step.args[2:3] or (None,))) or 9
        return _LoweredGraphStep(
            op=resolved_key,
            params={
                "input_col": input_col,
                "fast": fast,
                "slow": slow,
                "signal": signal,
                "out_fast": _coalesce_str(kw.get("out_fast")) or f"{input_col}_ema{fast}",
                "out_slow": _coalesce_str(kw.get("out_slow")) or f"{input_col}_ema{slow}",
                "out_macd": _coalesce_str(kw.get("out_macd")) or "macd",
                "out_signal": _coalesce_str(kw.get("out_signal")) or "macd_signal",
                "out_hist": _coalesce_str(kw.get("out_hist")) or "macd_hist",
            },
        )

    if resolved_key == "technical.Bollinger":
        input_col = primary_input()
        window = _coalesce_int(kw.get("window"), *(step.args[0:1] or (None,))) or 20
        num_std = float(kw.get("num_std", 2.0))
        return _LoweredGraphStep(
            op=resolved_key,
            params={
                "input_col": input_col,
                "window": window,
                "num_std": num_std,
                "out_mid": _coalesce_str(kw.get("out_mid"), kw.get("out"))
                or f"{input_col}_sma{window}",
                "out_std": _coalesce_str(kw.get("out_std")) or f"{input_col}_std{window}",
                "out_upper": _coalesce_str(kw.get("out_upper")) or f"{input_col}_bb_upper{window}",
                "out_lower": _coalesce_str(kw.get("out_lower")) or f"{input_col}_bb_lower{window}",
            },
        )

    if resolved_key == "technical.ATR":
        high_col = _coalesce_str(kw.get("high_col"), "high")
        low_col = _coalesce_str(kw.get("low_col"), "low")
        close_col = _coalesce_str(kw.get("close_col"), "close")
        window = _coalesce_int(kw.get("window"), *(step.args[:1] or (None,))) or 14
        return _LoweredGraphStep(
            op=resolved_key,
            params={
                "high_col": high_col,
                "low_col": low_col,
                "close_col": close_col,
                "window": window,
                "out_col": _coalesce_str(kw.get("out_col"), kw.get("out")) or f"atr_{window}",
            },
        )

    if resolved_key == "technical.OBV":
        input_col = primary_input()
        return _LoweredGraphStep(
            op=resolved_key,
            params={
                "input_col": input_col,
                "volume_col": _coalesce_str(kw.get("volume_col"), "volume") or "volume",
                "out_col": _coalesce_str(kw.get("out_col"), kw.get("out")) or "obv",
            },
        )

    if resolved_key == "technical.VWAP":
        return _LoweredGraphStep(
            op=resolved_key,
            params={
                "price_col": _coalesce_str(
                    kw.get("price_col"), kw.get("input_col"), *(step.inputs[:1] or (None,))
                )
                or "close",
                "volume_col": _coalesce_str(kw.get("volume_col"), "volume") or "volume",
                "session_col": _coalesce_str(kw.get("session_col")),
                "timestamp_col": _coalesce_str(kw.get("timestamp_col")),
                "out_col": _coalesce_str(kw.get("out_col"), kw.get("out")) or "vwap",
            },
        )

    return _LoweredGraphStep(op=resolved_key, params=kw)


def _step_cache_descriptor(step: FeatureStep) -> str | None:
    resolve_name, op_registry, _ = _registry_state()
    resolved = resolve_name(step.op)
    if resolved in op_registry:
        op_cls = op_registry[resolved]
        lowered = _lower_feature_step(step, resolved)
        op_identity = f"{getattr(op_cls, '__module__', '')}.{getattr(op_cls, '__name__', type(op_cls).__name__)}"
        version_tag = _code_hash_type(op_cls)
        payload = json.dumps(dict(lowered.params), sort_keys=True, default=str)
        return f"{lowered.op}:{op_identity}:{version_tag}:{payload}"
    if step.op in _FEATURE_OPS:
        payload = json.dumps(dict(step.kwargs), sort_keys=True, default=str)
        return f"legacy:{step.op}:{_code_hash(_FEATURE_OPS[step.op])}:{payload}"
    return None


def _execute_graph_step(
    step: FeatureStep, resolved_key: str, feats: PolarsDataFrame
) -> PolarsDataFrame:
    factory = importlib.import_module("pysrc.preprocessor.graph.factory")
    executor_module = importlib.import_module("pysrc.preprocessor.graph.executor")
    planner_module = importlib.import_module("pysrc.preprocessor.graph.planner")
    build_graph = cast(Callable[..., Any], factory.build_graph)
    executor_factory = executor_module.ExecutorFactory
    planner_cls = planner_module.Planner

    lowered = _lower_feature_step(step, resolved_key)
    graph = build_graph([lowered.op], {lowered.op: [dict(lowered.params)]})
    if not graph.nodes:
        raise MaterializationError(f"Graph built empty for op: {lowered.op}")

    try:
        planner = cast(Any, planner_cls)()
        plan_ir = planner.plan(graph, [])
    except (AttributeError, TypeError):

        def _node_op_name(node: Any) -> str:
            op = getattr(node, "op", None)
            if op is None:
                return ""
            return getattr(op, "name", None) or getattr(op, "NAME", "") or ""

        node = next((node for node in graph.nodes if _node_op_name(node) == resolved_key), None)
        if node is None:
            raise MaterializationError(
                f"Graph has no node for resolved op '{resolved_key}'; nodes: {[_node_op_name(node) for node in graph.nodes]}"
            )
        plan_ir = [node.to_ir()]

    try:
        executor = executor_factory.create("polars")
        return cast(PolarsDataFrame, executor.execute(plan_ir, feats, []))
    except ValueError as exc:
        raise MaterializationError(str(exc)) from exc


def _execute_legacy_feature_step(
    feats: pd.DataFrame | PolarsDataFrame,
    step: FeatureStep,
    fn: Callable[..., Any],
) -> pd.DataFrame | PolarsDataFrame:
    kw = dict(step.kwargs)
    if getattr(step, "out", ""):
        kw.setdefault("out", step.out)
    allow_extra = bool(kw.pop("allow_extra_outputs", False))
    declared_out = _coalesce_str(kw.get("out"), getattr(step, "out", "")) or ""
    existing_cols = list(feats.columns) if hasattr(feats, "columns") else []
    if declared_out and declared_out in existing_cols:
        raise MaterializationError(
            f"output column '{declared_out}' already present; possible collision"
        )
    before_cols = set(existing_cols)
    out = fn(feats, *step.inputs, *step.args, **kw)
    after_cols = set(out.columns) if hasattr(out, "columns") else set()
    if not after_cols.issuperset(before_cols):
        dropped = sorted(before_cols - after_cols)
        raise MaterializationError(f"legacy op dropped columns {dropped}; possible corruption")
    if declared_out and declared_out not in after_cols:
        raise MaterializationError(
            f"legacy op did not produce declared output column '{declared_out}'"
        )
    extra = after_cols - before_cols
    allowed_extra: set[str] = set()
    if declared_out and declared_out in extra:
        allowed_extra.add(declared_out)
    if not allow_extra:
        unexpected = extra - allowed_extra
        if unexpected:
            raise MaterializationError(
                f"legacy op produced unexpected output columns {sorted(unexpected)}; "
                "set allow_extra_outputs=True to permit this"
            )
    return cast(pd.DataFrame | PolarsDataFrame, out)


def materialize_features(
    ctx: StrategyContext,
    plan: FeaturePlan,
    price_col: str | None = None,
) -> pd.DataFrame | PolarsDataFrame:
    ctx.validate()
    cache = _Cache(ctx.cache_dir)
    governed = _is_governed_context(ctx)
    _, op_registry, alias_map = _registry_state()

    price_sig = (
        hashlib.sha1(
            np.asarray(pd.util.hash_pandas_object(ctx.prices.fillna(0)).values).tobytes()
        ).hexdigest()
        if isinstance(ctx.prices, pd.Series)
        else hashlib.sha1(
            np.asarray(
                pd.util.hash_pandas_object(ctx.prices.fillna(0), index=True).values
            ).tobytes()
        ).hexdigest()
    )
    op_hash_parts = [
        descriptor
        for descriptor in (_step_cache_descriptor(step) for step in plan.steps)
        if descriptor
    ]
    graph_keys = sorted(op_registry.keys()) if op_registry else []
    alias_items = sorted(alias_map.items()) if alias_map else []
    legacy_keys = sorted(_FEATURE_OPS.keys()) if _FEATURE_OPS else []
    registry_fingerprint = hashlib.sha1(
        json.dumps(
            {"graph": graph_keys, "aliases": alias_items, "legacy": legacy_keys}, sort_keys=True
        ).encode()
    ).hexdigest()
    op_hash = hashlib.sha1("".join(op_hash_parts).encode()).hexdigest()
    key = hashlib.sha1(
        json.dumps(
            {
                "plan": plan.signature(),
                "prices": price_sig,
                "backend": ctx.backend,
                "op_hash": op_hash,
                "registry_fingerprint": registry_fingerprint,
                "feature_op_registry_version": FEATURE_OP_REGISTRY_VERSION,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    cached = cache.get(key)
    if cached is not None:
        return cast(pd.DataFrame | PolarsDataFrame, cached)

    if isinstance(ctx.prices, pd.Series):
        df = ctx.prices.to_frame(name=price_col or "price")
    else:
        df = ctx.prices.copy()
        if price_col and price_col not in df.columns:
            raise ValidationError(f"price_col '{price_col}' not found in prices columns")

    if governed and hasattr(df, "attrs"):
        df.attrs.clear()

    feats_pd: pd.DataFrame | None = df.copy()
    feats_pl: PolarsDataFrame | None = None

    def ensure_polars() -> PolarsDataFrame:
        nonlocal feats_pd, feats_pl
        if pl is None:
            raise MaterializationError("Canonical graph execution requires polars to be installed")
        if feats_pl is None:
            assert feats_pd is not None
            feats_pl = pl.from_pandas(feats_pd)
        return feats_pl

    def ensure_pandas() -> pd.DataFrame:
        nonlocal feats_pd, feats_pl
        if feats_pd is None:
            assert feats_pl is not None
            feats_pd = feats_pl.to_pandas()
        return feats_pd

    # Hot path (§5.1): one pass over the plan; inner ops should stay vectorized.
    for step in plan.steps:
        kind, resolved_key = _resolve_feature_op(step)
        if kind == "op_registry":
            legacy_fn = _FEATURE_OPS.get(step.op)
            try:
                feats_pl = _execute_graph_step(step, cast(str, resolved_key), ensure_polars())
                feats_pd = None
                continue
            except (AttributeError, TypeError):
                if governed or legacy_fn is None or cast(str, resolved_key) in _CANONICAL_GRAPH_OPS:
                    raise
                if ctx.backend == "pandas":
                    feats_pd = cast(
                        pd.DataFrame, _execute_legacy_feature_step(ensure_pandas(), step, legacy_fn)
                    )
                    feats_pl = None
                else:
                    feats_pl = cast(
                        PolarsDataFrame,
                        _execute_legacy_feature_step(ensure_polars(), step, legacy_fn),
                    )
                    feats_pd = None
                continue

        if kind == "feature_ops":
            if governed:
                raise MaterializationError(
                    f"Governed feature materialization cannot execute '{step.op}' through _FEATURE_OPS; canonical graph lowering is required"
                )
            fn = _FEATURE_OPS[cast(str, resolved_key)]
            if ctx.backend == "pandas":
                feats_pd = cast(
                    pd.DataFrame, _execute_legacy_feature_step(ensure_pandas(), step, fn)
                )
                feats_pl = None
            else:
                feats_pl = cast(
                    PolarsDataFrame, _execute_legacy_feature_step(ensure_polars(), step, fn)
                )
                feats_pd = None
            continue

        raise MaterializationError(_unknown_op_message(step.op))

    if ctx.backend == "pandas":
        result: pd.DataFrame | PolarsDataFrame = ensure_pandas()
    else:
        result = ensure_polars()

    cache.set(key, result)
    return result


# --------------------------------------------------------------------------------------
# Strategy base and registry
# --------------------------------------------------------------------------------------
@dataclass
class StrategySpec:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


class StrategyRegistry:
    _REGISTRY: dict[str, type[PipelineStrategy]] = {}

    @classmethod
    def register(cls, name: str, strat_cls: type[PipelineStrategy]) -> None:
        # register canonical key -> class; class values allow late binding to params
        cls._REGISTRY[name] = strat_cls
        # NEW: also register by class-name alias (e.g., "RSIStrategy")
        cls._REGISTRY.setdefault(strat_cls.__name__, strat_cls)

    @classmethod
    def get(cls, name: str) -> type[PipelineStrategy]:
        # fast path: direct, canonical key
        if name in cls._REGISTRY:
            return cls._REGISTRY[name]
        # Try dynamic import if not found
        module_map = {
            "momentum": "pysrc.strategies.momentum",
            "momentum_tsmom": "pysrc.strategies.momentum",
            "momentum_dual": "pysrc.strategies.momentum",
            "momentum_industry": "pysrc.strategies.momentum",
            "momentum_residual": "pysrc.strategies.momentum",
            "momentum_kalman": "pysrc.strategies.momentum",
            "momentum_ensemble": "pysrc.strategies.momentum",
            "momentum_ml": "pysrc.strategies.momentum",
            "stat_arb_pairs": "pysrc.strategies.stat_arb.pairs",
        }

        if name in module_map:
            import importlib

            try:
                imported_module = importlib.import_module(module_map[name])
            except (ModuleNotFoundError, ImportError) as e:
                # Don’t leak import errors from optional modules; proceed to alias lookup.
                raise KeyError(
                    f"Strategy '{name}' not registered: failed to import module '{module_map[name]}': {e}"
                ) from e
            if module_map[name] == "pysrc.strategies.momentum":
                momentum_cls = getattr(imported_module, "MomentumStrategy", None)
                if isinstance(momentum_cls, type):
                    for alias in (
                        "momentum",
                        "momentum_tsmom",
                        "momentum_dual",
                        "momentum_industry",
                        "momentum_residual",
                        "momentum_kalman",
                        "momentum_ensemble",
                        "momentum_ml",
                    ):
                        cls._REGISTRY.setdefault(alias, momentum_cls)
            if name in cls._REGISTRY:
                return cls._REGISTRY[name]
            raise KeyError(
                f"Strategy '{name}' not registered after importing module '{module_map[name]}'"
            )
        # alias path: accept class-name lookups (e.g., "RSIStrategy") without duplicating registry keys
        for _, strat_cls in cls._REGISTRY.items():
            # tolerate mixed registration; only class values are eligible aliases
            if isinstance(strat_cls, type) and strat_cls.__name__ == name:
                return strat_cls
        # precise error type; no broad exception patterns
        raise KeyError(f"Strategy '{name}' not registered")

    @classmethod
    def clear_for_test(cls) -> None:
        # test-only hook to avoid global leakage between scenarios
        cls._REGISTRY.clear()


class PipelineStrategy:
    # Plug-ins (can be swapped at runtime)
    regime: RegimeDetector | None = None
    risk: RiskManager | None = None
    sizer: PositionSizer | None = None

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.random_state = params.get("random_state", 1337)

    # --- Abstracts ------------------------------------------------------------------
    def features_plan(self) -> FeaturePlan:
        raise NotImplementedError

    def generate_signal(self, features: pd.DataFrame | PolarsDataFrame) -> StrategySignal:
        raise NotImplementedError

    # --- Default implementations ----------------------------------------------------
    def generate_trade_intent(self, ctx: StrategyContext) -> TradeIntent:
        t0 = time.perf_counter()
        feats = materialize_features(ctx, self.features_plan())
        sig = _unwrap_signal_output(self.generate_signal(feats))

        # Regime gating
        if self.regime is not None:
            try:
                gate = self.regime.gate(feats)
                if isinstance(sig, pd.DataFrame):
                    sig = sig.mul(_to_series_like(sig, gate), axis=0)
                else:
                    sig = sig * _to_series_like(sig, gate)
            except (
                AttributeError,
                TypeError,
                ValidationError,
                MaterializationError,
                ValueError,
            ) as e:
                raise PipelineError(f"regime gating failed: {e}")

        # Sizing
        if self.sizer is not None:
            try:
                sig = self.sizer.size(sig)
            except (
                AttributeError,
                TypeError,
                ValidationError,
                MaterializationError,
                ValueError,
            ) as e:
                raise PipelineError(f"position sizing failed: {e}")

        weights = sig

        # Risk clamp
        if self.risk is not None:
            try:
                weights = self.risk.clamp(weights=weights, prices=ctx.prices)
            except (
                AttributeError,
                TypeError,
                ValidationError,
                MaterializationError,
                ValueError,
            ) as e:
                raise PipelineError(f"risk clamp failed: {e}")

        dt = time.perf_counter() - t0
        return TradeIntent(
            weights=weights,
            raw={"signal": sig, "features": feats},
            diagnostics={"latency_s": dt},
        )


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _to_series_like(
    ref: pd.Series | pd.DataFrame,
    x: pd.Series | PolarsSeries | float | int,
) -> pd.Series | float:
    if isinstance(ref, pd.DataFrame):
        if isinstance(x, (int, float)):
            return pd.Series(x, index=ref.index)
        if isinstance(x, pd.Series):
            aligned = x.reindex(ref.index)
            return aligned.ffill().fillna(0.0)
    else:  # ref is Series
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, pd.Series):
            aligned = x.reindex(ref.index)
            return aligned.ffill().fillna(0.0)
        raise ValidationError("Unsupported gate type for Series signal")
    raise ValidationError("Unsupported gate type for DataFrame signal")


def _unwrap_signal_output(signal: StrategySignal) -> SignalOutput:
    if isinstance(signal, (pd.Series, pd.DataFrame)):
        return signal
    envelope_signal = getattr(signal, "signal", None)
    if isinstance(envelope_signal, (pd.Series, pd.DataFrame)):
        raise ValidationError(
            "Strategies returning signal envelopes must override generate_trade_intent() "
            "and unwrap .signal explicitly."
        )
    raise ValidationError(
        "generate_signal() must return a pandas Series/DataFrame or a signal envelope "
        "with a Series/DataFrame .signal attribute."
    )


# --------------------------------------------------------------------------------------
# Portfolio backtest (multi-asset, costs, constraints)
# --------------------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    cost_per_unit_turnover: float = 0.0  # commission + slippage per 1.0 turnover
    leverage_cap: float = 1.0  # L1 norm cap per timestamp
    initial_nav: float = 1.0


def backtest_portfolio(
    prices: pd.DataFrame, weights: pd.DataFrame, cfg: BacktestConfig
) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or not isinstance(weights, pd.DataFrame):
        raise ValidationError("backtest_portfolio expects DataFrame prices & weights")

    prices, weights = prices.align(weights, join="inner", axis=0)

    # Compute returns
    rets = prices.pct_change().fillna(0.0)

    # Enforce leverage cap (L1) with zero-safe scaling
    l1 = weights.abs().sum(axis=1)
    safe = l1.replace(0.0, np.nan)
    scale = (cfg.leverage_cap / safe).clip(upper=1.0).fillna(1.0)
    weights_capped = weights.mul(scale, axis=0)

    # Turnover & costs
    w_prev = weights_capped.shift(1).fillna(0.0)
    turnover = (weights_capped - w_prev).abs().sum(axis=1)
    costs = turnover * cfg.cost_per_unit_turnover

    # Do not penalize the initial timestamp; first trade happens *from* cash into weights.
    if not costs.empty:
        costs.iloc[0] = 0.0

    pnl = (weights_capped.shift(1).fillna(0.0) * rets).sum(axis=1) - costs
    nav = (1.0 + pnl).cumprod() * cfg.initial_nav

    return pd.DataFrame(
        {
            "pnl": pnl,
            "nav": nav,
            "turnover": turnover,
            "leverage": l1,
            "costs": costs,
        }
    )


# --------------------------------------------------------------------------------------
# Parallel runner & Tuning (grid + optuna if available)
# --------------------------------------------------------------------------------------
@dataclass
class SweepResult:
    params: Mapping[str, Any]
    score: float
    details: Mapping[str, Any] = field(default_factory=dict)


def _score_nav(nav: pd.Series) -> float:
    # Simple Sharpe-like score (annualization omitted for brevity)
    r = nav.pct_change().dropna()
    return float(r.mean() / (r.std(ddof=0) + 1e-12))


# --- top-level, picklable worker for process pools -------------------------------------
def _sweep_worker_entry(
    strategy_cls: type[PipelineStrategy],
    params: Mapping[str, Any],
    ctx: StrategyContext,
    prices: pd.DataFrame,
    backtest_cfg: BacktestConfig,
) -> SweepResult:
    strat = strategy_cls(**dict(params))
    try:
        intent = strat.generate_trade_intent(ctx)
        w = intent.weights
        if isinstance(w, pd.Series):
            w = w.to_frame(name=prices.columns[0])
        bt = backtest_portfolio(prices=prices, weights=w, cfg=backtest_cfg)
        score = _score_nav(bt["nav"])
        return SweepResult(
            params=dict(params), score=score, details={"last_nav": float(bt["nav"].iloc[-1])}
        )
    except (ValueError, PipelineError, ValidationError, MaterializationError) as e:
        # Return a structured failure rather than crashing the sweep
        return SweepResult(
            params=dict(params),
            score=float("-inf"),
            details={"error_type": type(e).__name__, "error": str(e)},
        )


def parameter_sweep(
    strategy_cls: type[PipelineStrategy],
    param_grid: Mapping[str, Sequence[Any]],
    ctx: StrategyContext,
    *,
    prices: pd.DataFrame | None = None,
    backtest_cfg: BacktestConfig | None = None,
    n_jobs: int = 1,
) -> list[SweepResult]:
    prices = (
        prices
        if prices is not None
        else (ctx.prices if isinstance(ctx.prices, pd.DataFrame) else ctx.prices.to_frame())
    )
    backtest_cfg = backtest_cfg or BacktestConfig()

    keys = list(param_grid.keys())
    combos = [
        dict(zip(keys, vals, strict=False))
        for vals in itertools.product(*(param_grid[k] for k in keys))
    ]

    # Single-worker path: deterministic, no fork/spawn, no pickling of locals
    if n_jobs <= 1:
        results: list[SweepResult] = []
        for p in combos:
            results.append(_sweep_worker_entry(strategy_cls, p, ctx, prices, backtest_cfg))
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # Multi-worker path: true process-parallel execution
    parallel_results: list[SweepResult] = []
    with _fut.ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [
            ex.submit(_sweep_worker_entry, strategy_cls, p, ctx, prices, backtest_cfg)
            for p in combos
        ]
        for fu in _fut.as_completed(futs):
            # Worker already returns a structured result (including errors)
            parallel_results.append(fu.result())
    parallel_results.sort(key=lambda r: r.score, reverse=True)
    return parallel_results


def optuna_tune(
    strategy_cls: type[PipelineStrategy],
    sampler_spec: Mapping[str, tuple[float, float]],
    ctx: StrategyContext,
    *,
    prices: pd.DataFrame | None = None,
    backtest_cfg: BacktestConfig | None = None,
    n_trials: int = 50,
) -> list[SweepResult]:
    if optuna is None:
        # tests expect *exact* message text
        raise RuntimeError("optuna is not installed")
    assert optuna is not None

    prices = (
        prices
        if prices is not None
        else (ctx.prices if isinstance(ctx.prices, pd.DataFrame) else ctx.prices.to_frame())
    )
    backtest_cfg = backtest_cfg or BacktestConfig()

    def objective(trial: OptunaTrial) -> float:
        params = {k: trial.suggest_float(k, low, high) for k, (low, high) in sampler_spec.items()}
        strat = strategy_cls(**params)
        w = strat.generate_trade_intent(ctx).weights
        if isinstance(w, pd.Series):
            w = w.to_frame(name=prices.columns[0])
        bt = backtest_portfolio(prices=prices, weights=w, cfg=backtest_cfg)
        return _score_nav(bt["nav"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    trials = sorted(study.trials, key=lambda t: t.value or -np.inf, reverse=True)
    out: list[SweepResult] = []
    for t in trials:
        out.append(SweepResult(params=t.params, score=float(t.value or float("nan"))))
    return out


# --------------------------------------------------------------------------------------
# Drift monitor & Champion/Challenger
# --------------------------------------------------------------------------------------
@dataclass
class DriftState:
    ref_mean: float
    ref_std: float


def detect_drift(
    series: pd.Series,
    st: DriftState | None,
    threshold: float = 3.0,
    sensitivity: float = 1.0,
) -> tuple[DriftState, bool]:
    m = float(series.mean())
    s = float(series.std(ddof=0) + 1e-12)
    if st is None:
        return DriftState(ref_mean=m, ref_std=s), False
    eff_thresh = threshold / max(sensitivity, 1e-6)
    z = abs(m - st.ref_mean) / (st.ref_std + 1e-12)
    return DriftState(ref_mean=m, ref_std=s), bool(z >= eff_thresh)


@dataclass
class ChampionChallenger:
    strategy_cls: type[PipelineStrategy]
    ctx: StrategyContext
    prices: pd.DataFrame
    backtest_cfg: BacktestConfig = field(default_factory=BacktestConfig)

    # Store a concrete, mutable dict here (not Mapping)
    champion_params: dict[str, Any] = field(default_factory=dict)

    # Keep history flexible: we record immutable snapshots (dict copies)
    history: list[tuple[str, dict[str, Any], float]] = field(default_factory=list)

    def evaluate(self, params: Mapping[str, Any]) -> float:
        # Accept Mapping for callers, but expand to dict for **kwargs
        strat = self.strategy_cls(**dict(params))
        w = strat.generate_trade_intent(self.ctx).weights
        if isinstance(w, pd.Series):
            w = w.to_frame(name=self.prices.columns[0])
        bt = backtest_portfolio(self.prices, w, self.backtest_cfg)
        return float(_score_nav(bt["nav"]))

    def step(
        self, challenger_params: Mapping[str, Any], improvement: float = 0.01
    ) -> dict[str, Any]:
        # Normalize to dict to satisfy type hints and avoid aliasing
        challenger_params = dict(challenger_params)

        if not self.champion_params:
            self.champion_params = challenger_params
            self.history.append(("init", dict(challenger_params), float("nan")))
            return self.champion_params

        champ = self.evaluate(self.champion_params)
        chall = self.evaluate(challenger_params)
        self.history.append(("compare", dict(challenger_params), chall - champ))

        if chall >= champ * (1.0 + improvement):
            self.champion_params = challenger_params

        return self.champion_params


# --------------------------------------------------------------------------------------
# Simple, safe default plug-ins
# --------------------------------------------------------------------------------------
class NullRegime:
    def gate(self, features: pd.DataFrame | PolarsDataFrame) -> int:
        return 1


class LinearSizer:
    def __init__(self, scale: float = 1.0, clip: float = 1.0):
        self.scale = float(scale)
        self.clip = float(clip)

    def size(self, signal: pd.Series | pd.DataFrame, **_: Any) -> pd.Series | pd.DataFrame:
        if isinstance(signal, pd.DataFrame):
            return signal.clip(-self.clip, self.clip) * self.scale
        return signal.clip(-self.clip, self.clip) * self.scale


class TurnoverLimiterRisk:
    def __init__(self, max_turnover: float = 0.5):
        self.max_turnover = float(max_turnover)

    def clamp(
        self,
        weights: pd.Series | pd.DataFrame,
        prices: pd.Series | pd.DataFrame,
        **_: Any,
    ) -> pd.Series | pd.DataFrame:
        if isinstance(weights, pd.Series):
            vals = pd.Series(weights, copy=False).astype(float).to_numpy()
            out = np.empty_like(vals, dtype=float)

            prev = 0.0  # assume flat at t=-1
            for i in range(len(vals)):
                target = float(vals[i]) if np.isfinite(vals[i]) else 0.0
                delta = target - prev
                # clamp turnover per step
                if delta > self.max_turnover:
                    delta = self.max_turnover
                elif delta < -self.max_turnover:
                    delta = -self.max_turnover
                prev = prev + delta
                # hard clamp to allowable leverage bounds
                if prev > 1.0:
                    prev = 1.0
                elif prev < -1.0:
                    prev = -1.0
                out[i] = prev

            return pd.Series(out, index=weights.index, name=getattr(weights, "name", None))

        # DataFrame (multi-asset) path: vectorized across columns, sequential across time
        arr = pd.DataFrame(weights, copy=False).astype(float).to_numpy()
        out = np.empty_like(arr, dtype=float)

        prev_vec = np.zeros(arr.shape[1], dtype=float)
        for i in range(arr.shape[0]):
            row = arr[i]
            row = np.where(np.isfinite(row), row, 0.0)
            delta_vec = row - prev_vec
            delta_vec = np.clip(delta_vec, -self.max_turnover, self.max_turnover)
            prev_vec = prev_vec + delta_vec
            prev_vec = np.clip(prev_vec, -1.0, 1.0)
            out[i] = prev_vec

        return pd.DataFrame(out, index=weights.index, columns=weights.columns)


# --------------------------------------------------------------------------------------
# Blending utilities (combinatoric composition)
# --------------------------------------------------------------------------------------
@dataclass
class BlendSpec:
    parts: list[tuple[float, PipelineStrategy]]  # (weight, strategy)

    def normalize(self) -> None:
        s = sum(abs(w) for w, _ in self.parts) or 1.0
        self.parts = [(w / s, srt) for w, srt in self.parts]


def blend(ctx: StrategyContext, spec: BlendSpec) -> TradeIntent:
    # Shape-aware combinatorial blend
    spec.normalize()

    is_single = isinstance(ctx.prices, pd.Series) or (
        isinstance(ctx.prices, pd.DataFrame) and ctx.prices.shape[1] == 1
    )

    acc_s: pd.Series | None = None
    acc_df: pd.DataFrame | None = None

    for wgt, strat in spec.parts:
        ti = strat.generate_trade_intent(ctx)
        w = ti.weights

        if is_single:
            if isinstance(w, pd.DataFrame):
                w = w.iloc[:, 0] if w.shape[1] == 1 else w.sum(axis=1)
            w = w.reindex(ctx.prices.index).astype(float) * wgt
            acc_s = w if acc_s is None else acc_s.add(w, fill_value=0.0)
        else:
            if isinstance(w, pd.Series):
                w = w.to_frame(name=ctx.prices.columns[0])
            w = (
                w.reindex(
                    index=ctx.prices.index, columns=ctx.prices.columns, fill_value=0.0
                ).astype(float)
            ) * wgt
            acc_df = w if acc_df is None else acc_df.add(w, fill_value=0.0)

    if is_single:
        assert acc_s is not None
        l1 = acc_s.abs().replace(0.0, np.nan)
        scale = (1.0 / l1).clip(upper=1.0).fillna(1.0)
        acc_s = acc_s * scale
        return TradeIntent(weights=acc_s, raw={}, diagnostics={"blend_parts": len(spec.parts)})
    else:
        assert acc_df is not None
        l1 = acc_df.abs().sum(axis=1).replace(0.0, np.nan)
        scale = (1.0 / l1).clip(upper=1.0).fillna(1.0)
        acc_df = acc_df.mul(scale, axis=0)
        return TradeIntent(weights=acc_df, raw={}, diagnostics={"blend_parts": len(spec.parts)})


# --------------------------------------------------------------------------------------
# Legacy adapter (optional): allows gradual migration from baseStrategies-style classes
# --------------------------------------------------------------------------------------
class LegacyBaseStrategy(PipelineStrategy):
    def __init__(self, legacy_impl: Any, **params: Any) -> None:
        super().__init__(**params)
        self.legacy = legacy_impl

    def features_plan(self) -> FeaturePlan:
        # Assume legacy computes inside generate_signal; minimal plan
        return FeaturePlan.from_steps(
            [FeatureStep("PCT_CHANGE", inputs=("price",), args=(1,), kwargs={"out": "ret1"})]
        )

    def generate_signal(self, features: pd.DataFrame | PolarsDataFrame) -> pd.Series | pd.DataFrame:
        # Delegate to legacy class that expects pandas DataFrame
        if not isinstance(features, pd.DataFrame) and pl is not None:
            features = features.to_pandas()
        features_pd = cast(pd.DataFrame, features)
        return pd.Series(np.tanh(features_pd["ret1"].fillna(0.0)))


# --------------------------------------------------------------------------------------
# Example usage (commented)
# --------------------------------------------------------------------------------------
# ctx = StrategyContext(prices=price_df, backend="pandas")
# strat = MomentumStrategy(some_param=1.0)
# strat.regime = NullRegime()
# strat.sizer = LinearSizer(scale=1.0, clip=1.0)
# strat.risk = TurnoverLimiterRisk(max_turnover=0.3)
# intent = strat.generate_trade_intent(ctx)
# bt = backtest_portfolio(prices=price_df, weights=intent.weights.to_frame("w"), cfg=BacktestConfig(cost_per_unit_turnover=0.0005))
# results = parameter_sweep(MomentumStrategy, {"some_param": [0.5, 1.0, 1.5]}, ctx, prices=price_df, n_jobs=4)
