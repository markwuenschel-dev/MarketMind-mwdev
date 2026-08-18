from __future__ import annotations

# === IMPORTS (stdlib → project) ===
import asyncio
import inspect
import os
import threading
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from typing import Any, Final

from pysrc.core.errors import (
    ConfigValidationError,
)
from pysrc.core.runtime.optional_imports import optional_import

# --- MarketMind Ops & Utils Contract ---
from pysrc.ops.mm_logkit import (
    BoundLogger,
    get_logger,
)
from pysrc.ops.observability import get_metrics, get_tracing, instrument

# DataValidationError is required by invariants; enforce contract presence explicitly.
try:
    from pysrc.core.errors import DataValidationError
except ImportError as _e:
    raise RuntimeError("Missing ops/utils contract: DataValidationError") from _e

# === OPS HANDLES (module-level singletons) ===
logger: BoundLogger = get_logger(__name__)
metrics = get_metrics()
tracing = get_tracing()


# === METRICS COMPAT LAYER ===
# MetricsManager shape may vary (some versions expose .increment / .histogram,
# others expose .counter(name).inc()). We adapt at runtime to avoid AttributeError.
def _metrics_inc(name: str, **kwargs: Any) -> None:
    m = metrics
    if m is None:
        return
    # Preferred path: increment(name, tags=..., value=...)
    inc_fn = getattr(m, "increment", None)
    if callable(inc_fn):
        with suppress(Exception):
            inc_fn(name, **kwargs)
        return
    # Fallback path: counter(name).inc()
    counter_fn = getattr(m, "counter", None)
    if callable(counter_fn):
        with suppress(Exception):
            c = counter_fn(name)
            if hasattr(c, "inc") and callable(c.inc):
                c.inc()
    # If no recognized API, noop


def _metrics_hist(name: str, value: float | int) -> None:
    m = metrics
    if m is None:
        return
    hist_fn = getattr(m, "histogram", None)
    if callable(hist_fn):
        with suppress(Exception):
            hist_fn(name, value)
        return
    dist_fn = getattr(m, "distribution", None)
    if callable(dist_fn):
        with suppress(Exception):
            dist_fn(name, value)
    # else noop


# === OPTIONAL DEPENDENCIES (capability-driven singletons) ===
# We resolve optional backends once through the runtime optional-import surface.
pl = optional_import("polars")
pd = optional_import("pandas")
_HAS_POLARS = pl is not None
_HAS_PANDAS = pd is not None

# === ADAPTIVE KNOBS (12/10 loop inputs; all pulled from env) ===
ASYNC_RESOLVE_TIMEOUT: Final[float] = float(os.getenv("ASYNC_RESOLVE_TIMEOUT", "30.0"))
MAX_CONCAT_FRAMES: Final[int] = int(os.getenv("MAX_CONCAT_FRAMES", "1000"))
DATETIME_PARSE_STRICT: Final[bool] = os.getenv("DATETIME_PARSE_STRICT", "false").lower() == "true"
NORMALIZE_MAX_WORKERS: Final[int] = max(1, int(os.getenv("NORMALIZE_MAX_WORKERS", "4")))

# to_polars fallback policy knob. "allow" → best-effort pl.DataFrame(df); anything else forbids loose conversion.
_TO_POLARS_POLICY = os.getenv("TO_POLARS_CONVERSION_POLICY", "allow").lower()
TO_POLARS_ALLOW_LOOSE_CONSTRUCT: Final[bool] = _TO_POLARS_POLICY in ("allow", "1", "true", "yes")

# ticker inference priority knob (comma-separated list like "ticker,symbol,asset")
_TICKER_PRIORITY_RAW = os.getenv("TICKER_PRIORITY", "ticker,symbol,asset")
_TICKER_PRIORITY: Sequence[str] = tuple(
    c.strip() for c in _TICKER_PRIORITY_RAW.split(",") if c.strip()
)

__all__ = ["to_polars", "ensure_datetime_col", "infer_ticker_col", "normalize_fetched"]


def _resolve_awaitable_sync(awaitable: Any, timeout: float = ASYNC_RESOLVE_TIMEOUT) -> Any:
    # Resolve an awaitable in a dedicated event loop on a background daemon thread.
    # Fatal control-path exceptions (KeyboardInterrupt, SystemExit, asyncio.CancelledError)
    # are propagated verbatim. Timeout / generic failures become DataValidationError.
    # No 'except BaseException:' is used anywhere in this helper.
    result_box: dict[str, Any] = {}
    error_box: dict[str, Exception] = {}
    fatal_box: dict[str, BaseException] = {}

    def _runner() -> None:
        async def _await_with_timeout(a: Any) -> Any:
            return await asyncio.wait_for(a, timeout=timeout)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box["v"] = loop.run_until_complete(_await_with_timeout(awaitable))
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError) as fatal_exc:
            fatal_box["fatal"] = fatal_exc
        except TimeoutError:
            error_box["e"] = DataValidationError(f"Async resolution timeout after {timeout}s")
        except Exception as e:
            error_box["e"] = DataValidationError(
                f"Async resolution failed: {type(e).__name__}: {str(e)}"
            )
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout + 1.0)

    # If the helper thread never finished, treat as timeout escalation.
    if t.is_alive():
        raise DataValidationError(f"Async resolution thread hung after {timeout + 1.0}s")

    # Re-raise fatal exceptions immediately (Zero-Tolerance Contract).
    if "fatal" in fatal_box:
        raise fatal_box["fatal"]

    # If coroutine failed or timed out, raise the mapped DataValidationError.
    if "e" in error_box:
        raise error_box["e"]

    # Success path.
    return result_box["v"]


@instrument(record_exceptions=True, measure_latency=True)
def to_polars(df: Any) -> Any:
    # Convert arbitrary inputs to a clean Polars DataFrame.
    # Adaptive knob: TO_POLARS_ALLOW_LOOSE_CONSTRUCT.
    _metrics_inc("dataframe_helpers.to_polars.count")

    # Relaxed dependency check: if Polars was not found up front, try a direct import
    # before treating it as a hard configuration error. This keeps tests working
    # when polars is installed but capability flags are not wired correctly.
    global pl
    if not _HAS_POLARS or pl is None:
        try:
            import polars as _polars  # type: ignore

            pl = _polars
        except Exception:
            raise ConfigValidationError("Polars is required but not installed or enabled.")

    # Already polars → fast path
    if isinstance(df, pl.DataFrame):
        _metrics_inc("dataframe_helpers.to_polars.already_polars")
        return df

    # pandas → polars, stripping pandas index to prevent schema pollution
    if _HAS_PANDAS and pd is not None and isinstance(df, pd.DataFrame):
        _metrics_inc("dataframe_helpers.to_polars.from_pandas")
        try:
            return pl.from_pandas(df, include_index=False)
        except TypeError:
            df_clean = df.reset_index(drop=True)
            return pl.from_pandas(df_clean)

    # dict-like → polars
    if isinstance(df, Mapping):
        _metrics_inc("dataframe_helpers.to_polars.from_dict")
        try:
            return pl.from_dicts([df])
        except Exception:
            return pl.DataFrame(df)

    # list / tuple → polars
    if isinstance(df, (list, tuple)):
        _metrics_inc("dataframe_helpers.to_polars.from_sequence")
        if df and isinstance(df[0], Mapping):
            return pl.from_dicts(df)
        try:
            return pl.DataFrame(df)
        except Exception as e:
            raise DataValidationError(
                f"Cannot convert sequence to Polars DataFrame: {type(e).__name__}: {str(e)}"
            ) from e

    # final fallback if policy allows arbitrary pl.DataFrame(df)
    if not TO_POLARS_ALLOW_LOOSE_CONSTRUCT:
        raise DataValidationError("Polars conversion disabled by policy")

    try:
        _metrics_inc("dataframe_helpers.to_polars.fallback")
        return pl.DataFrame(df)
    except Exception as e:
        raise DataValidationError(
            f"Cannot convert {type(df).__name__} to Polars DataFrame: {type(e).__name__}: {str(e)}"
        ) from e


@instrument(record_exceptions=True, measure_latency=True)
def ensure_datetime_col(df: Any, date_col: str = "date") -> Any:
    # Enforce that df[date_col] is present and usable as a datetime column.
    # Adaptive knob: DATETIME_PARSE_STRICT (applies to Polars string→Datetime parsing).
    # Invariant: MUST raise DataValidationError if pandas coercion yields any NaT.
    _metrics_inc("dataframe_helpers.ensure_datetime_col.count")

    # Polars branch
    if _HAS_POLARS and pl is not None and isinstance(df, pl.DataFrame):
        _metrics_inc("dataframe_helpers.ensure_datetime_col.polars")

        if date_col not in df.columns:
            raise DataValidationError(f"Missing '{date_col}' column in Polars DataFrame.")

        dtype = df[date_col].dtype

        # Date -> Datetime cast
        if dtype == pl.Date:
            _metrics_inc("dataframe_helpers.ensure_datetime_col.polars_date_cast")
            return df.with_columns(pl.col(date_col).cast(pl.Datetime))

        # String-like or other non-Datetime -> parse to Datetime using knob strictness
        if dtype != pl.Datetime:
            _metrics_inc("dataframe_helpers.ensure_datetime_col.polars_str_parse")
            try:
                return df.with_columns(
                    pl.col(date_col).str.strptime(pl.Datetime, strict=DATETIME_PARSE_STRICT)
                )
            except Exception as e:
                raise DataValidationError(
                    f"Failed to parse datetime in Polars column '{date_col}': {type(e).__name__}: {str(e)}"
                ) from e

        # Already Datetime
        return df

    # pandas branch
    if _HAS_PANDAS and pd is not None and isinstance(df, pd.DataFrame):
        _metrics_inc("dataframe_helpers.ensure_datetime_col.pandas")

        if date_col not in df.columns:
            raise DataValidationError(f"Missing '{date_col}' column in pandas DataFrame.")

        # Coerce to timezone-aware datetime if not already datetime64-any
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            _metrics_inc("dataframe_helpers.ensure_datetime_col.pandas_coerce")
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)

        # Unconditional invariant: NaT after coercion is an error, not a warning
        null_count = int(df[date_col].isna().sum())
        if null_count > 0:
            raise DataValidationError(
                f"Failed to parse datetime in pandas column '{date_col}'; {null_count} null after parse."
            )

        return df

    # Unsupported df type
    raise DataValidationError("Unsupported dataframe type in ensure_datetime_col")


@instrument(record_exceptions=True, measure_latency=True)
def infer_ticker_col(df: Any) -> str | None:
    # Heuristic inference of a ticker/symbol column.
    # Adaptive knob: _TICKER_PRIORITY defines priority order dynamically via env.
    _metrics_inc("dataframe_helpers.infer_ticker_col.count")

    for cand in _TICKER_PRIORITY:
        if _HAS_POLARS and pl is not None and isinstance(df, pl.DataFrame):
            if cand in df.columns:
                _metrics_inc(
                    "dataframe_helpers.infer_ticker_col.found",
                    tags={"column": cand, "backend": "polars"},
                )
                return cand

        if _HAS_PANDAS and pd is not None and isinstance(df, pd.DataFrame):
            if cand in df.columns:
                _metrics_inc(
                    "dataframe_helpers.infer_ticker_col.found",
                    tags={"column": cand, "backend": "pandas"},
                )
                return cand

    _metrics_inc("dataframe_helpers.infer_ticker_col.not_found")
    return None


@instrument(record_exceptions=True, measure_latency=True)
def normalize_fetched(obj: Any) -> Any:
    # Consolidate heterogeneous fetch results (awaitables, LazyFrames, dict[ticker->df])
    # into a single concrete pandas or polars DataFrame.
    #
    # Adaptive knobs:
    #   ASYNC_RESOLVE_TIMEOUT: max seconds to await async sources
    #   MAX_CONCAT_FRAMES: warn threshold for dict fan-in size
    #   NORMALIZE_MAX_WORKERS: per-ticker parallelism budget
    #
    # Invariants:
    #   - Resolve/collect all awaitables before returning (no pending coroutine leaks)
    #   - Collect all LazyFrames (no lazy computation leakage)
    #   - Inject 'ticker' col where missing when merging dict[ticker->df]
    #   - Skip dict entries whose value is an Exception
    #   - Raise DataValidationError if the dict yields zero usable frames
    #   - Return a concrete DataFrame (pandas or polars), never LazyFrame
    _metrics_inc("dataframe_helpers.normalize_fetched.count")

    # Step 1: Resolve awaitables / futures early
    if inspect.iscoroutine(obj) or isinstance(obj, asyncio.Future):
        _metrics_inc("dataframe_helpers.normalize_fetched.awaitable_resolve")
        try:
            asyncio.get_running_loop()
            obj = _resolve_awaitable_sync(obj, timeout=ASYNC_RESOLVE_TIMEOUT)
        except RuntimeError:
            obj = asyncio.run(obj)

    # Step 2: If we got a single Polars LazyFrame, collect it
    if _HAS_POLARS and pl is not None and isinstance(obj, pl.LazyFrame):
        _metrics_inc("dataframe_helpers.normalize_fetched.lazyframe_collect")
        obj = obj.collect()

    # Step 3: dict[ticker -> frame/lazyframe/Exception] consolidation
    if isinstance(obj, dict):
        _metrics_inc("dataframe_helpers.normalize_fetched.dict_concat")

        dict_size = len(obj)
        if dict_size > MAX_CONCAT_FRAMES:
            logger.warning(
                "normalize_fetched_large_dict",
                dict_size=dict_size,
                max_allowed=MAX_CONCAT_FRAMES,
            )

        lazy_parts: list[Any] = []
        concrete_parts: list[tuple[str, Any]] = []

        # Classify items from dict
        for tkr, part in obj.items():
            if isinstance(part, Exception):
                _metrics_inc("dataframe_helpers.normalize_fetched.skipped_exception")
                continue

            # LazyFrame branch: add ticker col before parallel collect
            if _HAS_POLARS and pl is not None and isinstance(part, pl.LazyFrame):
                if "ticker" not in part.columns:
                    part = part.with_columns(pl.lit(tkr).alias("ticker"))
                lazy_parts.append(part)
                continue

            # Concrete polars / pandas frames
            if _HAS_POLARS and pl is not None and isinstance(part, pl.DataFrame):
                concrete_parts.append((tkr, part))
                continue

            if _HAS_PANDAS and pd is not None and isinstance(part, pd.DataFrame):
                concrete_parts.append((tkr, part))
                continue

            # Fallback: unknown entry type → skip, not fatal
            logger.warning(
                "normalize_fetched.skip_unknown_type",
                ticker=tkr,
                type=str(type(part)),
            )

        collected_parts: list[Any] = []

        # 3a. Parallel collect LazyFrames with polars-native executor
        if lazy_parts:
            try:
                lazy_collected = pl.collect_all(lazy_parts)
            except Exception as e:
                raise DataValidationError("Failed to collect lazy frames in parallel") from e
            collected_parts.extend(lazy_collected)
            _metrics_inc(
                "dataframe_helpers.normalize_fetched.lazy_collected",
                tags={"count": len(lazy_parts)},
            )

        # 3b. Inject ticker for concrete frames, potentially in parallel
        def _process_concrete(tkr: str, frame: Any) -> Any:
            if _HAS_POLARS and pl is not None and isinstance(frame, pl.DataFrame):
                if "ticker" not in frame.columns:
                    return frame.with_columns(pl.lit(tkr).alias("ticker"))
                return frame
            if _HAS_PANDAS and pd is not None and isinstance(frame, pd.DataFrame):
                if "ticker" not in frame.columns:
                    return frame.assign(ticker=tkr)
                return frame
            return None

        processed_concrete: list[Any] = []
        max_workers = min(NORMALIZE_MAX_WORKERS, len(concrete_parts)) if concrete_parts else 0

        if max_workers > 1 and len(concrete_parts) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_process_concrete, tkr, frame): tkr
                    for (tkr, frame) in concrete_parts
                }
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception:
                        continue
                    if res is not None:
                        processed_concrete.append(res)
        else:
            for tkr, frame in concrete_parts:
                try:
                    res = _process_concrete(tkr, frame)
                except Exception:
                    continue
                if res is not None:
                    processed_concrete.append(res)

        collected_parts.extend(processed_concrete)
        _metrics_inc(
            "dataframe_helpers.normalize_fetched.concrete_processed",
            tags={"count": len(processed_concrete)},
        )

        # 3c. Concat results
        if not collected_parts:
            raise DataValidationError("Fetched dict had no usable DataFrames or LazyFrames.")

        _metrics_hist(
            "dataframe_helpers.normalize_fetched.concat_size",
            len(collected_parts),
        )

        # Homogeneous polars
        if (
            _HAS_POLARS
            and pl is not None
            and all(isinstance(p, pl.DataFrame) for p in collected_parts)
        ):
            obj = pl.concat(collected_parts, how="vertical_relaxed")

        # Homogeneous pandas
        elif (
            _HAS_PANDAS
            and pd is not None
            and all(isinstance(p, pd.DataFrame) for p in collected_parts)
        ):
            obj = pd.concat(collected_parts, ignore_index=True)

        # Mixed → convert to polars
        else:
            if not (_HAS_POLARS and pl is not None):
                raise ConfigValidationError(
                    "normalize_fetched requires Polars for mixed-type consolidation."
                )
            obj = pl.concat(
                [to_polars(p) for p in collected_parts],
                how="vertical_relaxed",
            )

    # Step 4: must return a concrete DataFrame (never LazyFrame / awaitable)
    if _HAS_POLARS and pl is not None and isinstance(obj, pl.DataFrame):
        _metrics_inc("dataframe_helpers.normalize_fetched.result_polars")
        return obj

    if _HAS_PANDAS and pd is not None and isinstance(obj, pd.DataFrame):
        _metrics_inc("dataframe_helpers.normalize_fetched.result_pandas")
        return obj

    _metrics_inc("dataframe_helpers.normalize_fetched.fallback_to_polars")
    return to_polars(obj)


# --- Back-compat shims for older call sites ---
def _to_polars(*args, **kwargs):
    warnings.warn(
        "`_to_polars` is deprecated; use `to_polars`",
        DeprecationWarning,
        stacklevel=2,
    )
    return to_polars(*args, **kwargs)


def _ensure_datetime_col(*args, **kwargs):
    warnings.warn(
        "`_ensure_datetime_col` is deprecated; use `ensure_datetime_col`",
        DeprecationWarning,
        stacklevel=2,
    )
    return ensure_datetime_col(*args, **kwargs)


def _infer_ticker_col(*args, **kwargs):
    warnings.warn(
        "`_infer_ticker_col` is deprecated; use `infer_ticker_col`",
        DeprecationWarning,
        stacklevel=2,
    )
    return infer_ticker_col(*args, **kwargs)


def _normalize_fetched(*args, **kwargs):
    warnings.warn(
        "`_normalize_fetched` is deprecated; use `normalize_fetched`",
        DeprecationWarning,
        stacklevel=2,
    )
    return normalize_fetched(*args, **kwargs)
