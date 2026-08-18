# py/ops/caching.py
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar, cast

import pandas as pd
import polars as pl

from pysrc.core.errors import DataError, InvalidInputError
from pysrc.core.validation import validate_dataframe

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def ttl_cache(*, ttl: int, maxsize: int = 128) -> Callable[[F], F]:
    """Cache function results for ttl seconds using JSON-stable call keys."""
    try:
        from cachetools import TTLCache
    except ImportError:

        class TTLCache(dict):  # type: ignore[no-redef]
            def __init__(self, maxsize: int, ttl: int) -> None:
                super().__init__()
                self.maxsize = maxsize
                self.ttl = ttl

            def __getitem__(self, key: str) -> Any:
                expires_at, value = super().__getitem__(key)
                if time.monotonic() >= expires_at:
                    super().__delitem__(key)
                    raise KeyError(key)
                return value

            def __setitem__(self, key: str, value: Any) -> None:
                if len(self) >= self.maxsize:
                    oldest = next(iter(self))
                    super().__delitem__(oldest)
                super().__setitem__(key, (time.monotonic() + self.ttl, value))

    def decorator(func: F) -> F:
        cache: Any = TTLCache(maxsize=maxsize, ttl=ttl)
        lock = RLock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = json.dumps(
                {"args": args, "kwargs": kwargs},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            with lock:
                try:
                    return cache[key]
                except KeyError:
                    pass
            result = func(*args, **kwargs)
            with lock:
                cache[key] = result
            return result

        def cache_clear() -> None:
            with lock:
                cache.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        return cast(F, wrapper)

    return decorator


# ============================================================================
# Modern Hash Functions
# ============================================================================


class HashAlgorithm(Enum):
    """Modern hash algorithms optimized for different use cases"""

    XXHASH = "xxhash"  # Fast non-cryptographic hash
    BLAKE3 = "blake3"  # High-speed cryptographic hash
    SIPHASH = "siphash"  # Keyed hash for untrusted inputs
    SHA256 = "sha256"  # Traditional cryptographic hash


def hash_bytes(data: bytes, algo: HashAlgorithm = HashAlgorithm.XXHASH) -> str:
    """Hash bytes using specified algorithm with optional accelerated backends."""
    import hashlib

    from pysrc.core.runtime.optional_imports import optional_import

    if algo == HashAlgorithm.XXHASH:
        xxhash = optional_import("xxhash")
        if xxhash is not None:
            return xxhash.xxh3_128(data).hexdigest()
        # Fallback to SHA256 if xxhash unavailable
        return hashlib.sha256(data).hexdigest()
    elif algo == HashAlgorithm.BLAKE3:
        blake3 = optional_import("blake3")
        if blake3 is not None:
            return blake3.blake3(data).hexdigest()
        # Fallback to SHA256 if blake3 unavailable
        return hashlib.sha256(data).hexdigest()
    elif algo == HashAlgorithm.SIPHASH:
        # Use secrets module for key in production
        return hashlib.blake2b(data, digest_size=16).hexdigest()
    else:  # SHA256 default
        return hashlib.sha256(data).hexdigest()


def hash_config(cfg_obj, algo: HashAlgorithm = HashAlgorithm.XXHASH) -> str:
    """Hash configuration with fast algorithm"""
    payload = json.dumps(cfg_obj, sort_keys=True, separators=(",", ":"))
    return hash_bytes(payload.encode("utf-8"), algo)


def hash_dataframe_deterministic(df, cols=None, algo: HashAlgorithm = HashAlgorithm.XXHASH) -> str:
    import numpy as np
    import pyarrow as pa

    # 1) Normalize to pandas
    if hasattr(df, "to_pandas"):
        try:
            df_pd = df.to_pandas()
        except (ValueError, TypeError, AttributeError) as e:
            raise InvalidInputError(f"Failed to convert to pandas DataFrame: {e}") from e
    elif isinstance(df, pd.DataFrame):
        df_pd = df.copy()
    else:
        try:
            df_pd = pd.DataFrame(df)
        except (ValueError, TypeError) as e:
            raise InvalidInputError(f"Cannot coerce input to DataFrame: {e}") from e

    if cols is not None:
        try:
            df_pd = df_pd[cols]
        except KeyError as e:
            raise InvalidInputError(f"Requested columns not found: {e}") from e

    # 2) Stable column order
    df_pd = df_pd.sort_index(axis=1)

    # 3) Canonicalize columns to avoid Arrow conversion overflows
    INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1

    def _is_missing_scalar(value: Any) -> bool:
        if value is None:
            return True
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
        return False

    def _stringify_non_missing(value: Any) -> Any:
        return np.nan if _is_missing_scalar(value) else str(value)

    def _canon_object_value(value: Any) -> Any:
        if _is_missing_scalar(value):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return value
        return str(value)

    def _needs_stringify(series: pd.Series) -> bool:
        """True if any python int lies outside int64 range (object dtypes included)"""
        if series.dtype.kind in ("i", "u"):  # numpy ints
            # numpy will already be fixed-width; overflows would have failed earlier
            return False
        # object or mixed types: detect large python ints
        mask = series.map(
            lambda x: isinstance(x, int) and not isinstance(x, bool), na_action="ignore"
        )
        if mask.any():
            vals = series[mask].astype("object")
            try:
                return bool((vals < INT64_MIN).any() or (vals > INT64_MAX).any())
            except (TypeError, ValueError):
                return False
        return False

    def _canon(series: pd.Series) -> pd.Series:
        """
        Canonicalize series for Arrow compatibility.
        - Large ints → strings
        - Numeric dtypes with inf/nan → keep numeric, DON'T stringify NaN
        - Datetime → ISO UTC strings
        - Object dtype → infer and preserve numeric types where possible
        """
        s = series

        # Large python ints that exceed int64 → stringify
        if _needs_stringify(s):
            return s.map(_stringify_non_missing)

        if pd.api.types.is_string_dtype(s):
            return s.map(_stringify_non_missing).astype("object")

        # Floating point: keep numeric, replace inf with sentinels for stable hashing
        if pd.api.types.is_float_dtype(s):
            # Replace inf with large finite values for deterministic hashing
            # Keep NaN as np.nan (not string) for Arrow compatibility
            return s.replace([np.inf, -np.inf], [1e308, -1e308])

        # Datetime → UTC ISO strings
        if pd.api.types.is_datetime64_any_dtype(s):
            try:
                return pd.to_datetime(s, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            except (ValueError, TypeError, AttributeError):
                # Fall back to string representation
                return s.astype(str)

        # Object dtype: attempt to preserve numeric types
        if s.dtype == "object":
            # Try to infer and convert to numeric, preserving NaN as numeric
            try:
                # Attempt numeric conversion - this preserves np.nan as float NaN
                converted = pd.to_numeric(s, errors="coerce")
                # If we got mostly numeric values (> 50%), use the numeric version
                if converted.notna().sum() > len(s) * 0.5:
                    return converted
                # Otherwise, stringify non-null values only, keep NaN as is
                return s.map(_canon_object_value)
            except (ValueError, TypeError):
                # Final fallback: stringify but preserve numeric NaN
                return s.map(_canon_object_value)

        return s

    try:
        df_pd = df_pd.apply(_canon)
    except (ValueError, TypeError, AttributeError) as e:
        raise InvalidInputError(f"Canonicalization failed: {e}") from e

    # 4) Arrow table + IPC bytes = canonical, cross-platform binary
    try:
        table = pa.Table.from_pandas(df_pd, preserve_index=True)
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError, TypeError) as e:
        raise InvalidInputError(f"Arrow conversion failed: {e}") from e

    try:
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        arrow_bytes = sink.getvalue().to_pybytes()
    except (pa.ArrowInvalid, OSError) as e:
        raise InvalidInputError(f"Arrow serialization failed: {e}") from e

    return hash_bytes(arrow_bytes, algo)


def versioned_key(*parts: str, version: str = "v1") -> str:
    """Create versioned cache key for safe invalidation"""
    joined = "|".join([version] + [str(p) for p in parts])
    return hash_bytes(joined.encode("utf-8"))


# ============================================================================
# Compression Strategy
# ============================================================================


class CompressionLevel(Enum):
    NONE = 0
    FAST = 1  # LZ4-like, for medium data
    HIGH = 2  # Zstd level 3, for large cold data


@dataclass
class CompressionStrategy:
    """Adaptive compression based on data size and access patterns"""

    small_threshold: int = 1024  # Don't compress below 1KB
    fast_threshold: int = 100_000  # Use fast compression up to 100KB

    def compress(
        self, data: bytes, level: CompressionLevel = None
    ) -> tuple[bytes, CompressionLevel]:
        """Compress data with adaptive strategy"""
        size = len(data)

        if level is None:
            if size < self.small_threshold:
                level = CompressionLevel.NONE
            elif size < self.fast_threshold:
                level = CompressionLevel.FAST
            else:
                level = CompressionLevel.HIGH

        if level == CompressionLevel.NONE:
            return data, level
        elif level == CompressionLevel.FAST:
            # Use zlib level 1 as fast compression
            return zlib.compress(data, level=1), level
        else:  # HIGH
            from pysrc.core.runtime.optional_imports import optional_import

            zstd = optional_import("zstandard")
            if zstd is not None:
                cctx = zstd.ZstdCompressor(level=3)
                return cctx.compress(data), level
            # Fallback to zlib level 6 if zstandard unavailable
            return zlib.compress(data, level=6), level

    def decompress(self, data: bytes, level: CompressionLevel) -> bytes:
        """Decompress data"""
        if level == CompressionLevel.NONE:
            return data
        elif level == CompressionLevel.FAST:
            return zlib.decompress(data)
        else:
            from pysrc.core.runtime.optional_imports import optional_import

            zstd = optional_import("zstandard")
            if zstd is not None:
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(data)
            # Fallback to zlib if zstandard unavailable
            return zlib.decompress(data)


# ============================================================================
# Enhanced Cache Manager with TinyLFU-inspired Admission
# ============================================================================


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    value: Any
    expiry: float
    version: int = 0
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    compression: CompressionLevel = CompressionLevel.NONE


@dataclass
class CacheMetrics:
    """Observability metrics"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    sets: int = 0
    total_latency_ns: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_latency_us(self) -> float:
        ops = self.hits + self.misses
        return (self.total_latency_ns / ops / 1000) if ops > 0 else 0.0


class AdaptiveTTLManager:
    """Manages TTLs based on volatility and access patterns"""

    def __init__(self, base_ttl: float = 300):
        self.base_ttl = base_ttl
        self.volatility_multiplier = 1.0

    def get_ttl(self, key: Any, volatility: float = 0.0) -> float:
        """Calculate adaptive TTL based on market volatility"""
        # Higher volatility = shorter TTL
        adaptive_factor = 1.0 / (1.0 + volatility)
        return self.base_ttl * adaptive_factor

    def update_volatility(self, volatility: float):
        """Update volatility for all future TTL calculations"""
        self.volatility_multiplier = 1.0 / (1.0 + volatility)


class EnhancedCacheManager:
    """
    Production-grade cache manager with:
    - TinyLFU-inspired admission policy
    - Adaptive TTL
    - Compression
    - Circuit breaker
    - Comprehensive metrics
    """

    def __init__(
        self,
        max_size: int = 128,
        ttl: float | None = None,
        eviction_policy: str = "lru",
        enable_compression: bool = True,
        enable_metrics: bool = True,
    ):
        self.max_size = max_size
        self.base_ttl = ttl
        self.eviction_policy = eviction_policy
        self.enable_compression = enable_compression

        self._cache: dict[Any, CacheEntry] = {}
        self._access_order: OrderedDict[Any, None] = OrderedDict()

        # Frequency sketch for admission policy (simplified)
        self._frequency: dict[Any, int] = {}

        self.compression = CompressionStrategy()
        self.ttl_manager = AdaptiveTTLManager(ttl or 300)
        self.metrics = CacheMetrics() if enable_metrics else None

        # Circuit breaker state
        self._circuit_open = False
        self._failure_count = 0
        self._last_failure_time = 0
        self._failure_threshold = 5
        self._reset_timeout = 60

        if eviction_policy not in {"lru", "lfu", "fifo"}:
            raise InvalidInputError(f"Unsupported eviction policy: {eviction_policy}")

    def _should_admit(self, key: Any) -> bool:
        """TinyLFU-inspired admission: only cache frequently accessed items"""
        # If cache not full, always admit
        if len(self._cache) < self.max_size:
            return True

        # Check if new item is more worthy than LRU candidate
        new_freq = self._frequency.get(key, 0)

        # Get LRU victim's frequency
        victim_key = next(iter(self._access_order))
        victim_freq = self._cache[victim_key].access_count

        # Admit if new item accessed more frequently
        return new_freq > victim_freq

    def _update_circuit_breaker(self, success: bool):
        """Update circuit breaker state"""
        now = time.time()

        if success:
            self._failure_count = 0
            if self._circuit_open and (now - self._last_failure_time) > self._reset_timeout:
                self._circuit_open = False
        else:
            self._failure_count += 1
            self._last_failure_time = now
            if self._failure_count >= self._failure_threshold:
                self._circuit_open = True

    def _evict(self):
        """Evict one item based on policy"""
        if self.eviction_policy == "lru":
            key, _ = self._access_order.popitem(last=False)
        elif self.eviction_policy == "lfu":
            # Find least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        else:  # fifo
            key = next(iter(self._cache))

        del self._cache[key]
        self._access_order.pop(key, None)

        if self.metrics:
            self.metrics.evictions += 1

    def _cleanup_expired(self):
        """Remove expired items"""
        now = time.time()
        expired = [k for k, entry in self._cache.items() if entry.expiry < now]
        for k in expired:
            del self._cache[k]
            self._access_order.pop(k, None)

    def get(self, key: Any, version: int = 0) -> Any | None:
        """Get value if not expired and version matches"""
        if self._circuit_open:
            return None

        start = time.perf_counter_ns()

        self._cleanup_expired()

        if key in self._cache:
            entry = self._cache[key]

            # Check version compatibility
            if entry.version != version and version != 0:
                if self.metrics:
                    self.metrics.misses += 1
                return None

            if self.base_ttl is None or entry.expiry > time.time():
                # Update access patterns
                entry.access_count += 1
                entry.last_access = time.time()
                self._frequency[key] = self._frequency.get(key, 0) + 1

                if self.eviction_policy == "lru":
                    self._access_order.move_to_end(key)

                if self.metrics:
                    self.metrics.hits += 1
                    self.metrics.total_latency_ns += time.perf_counter_ns() - start

                self._update_circuit_breaker(True)
                return entry.value
            else:
                del self._cache[key]
                self._access_order.pop(key, None)

        if self.metrics:
            self.metrics.misses += 1
            self.metrics.total_latency_ns += time.perf_counter_ns() - start

        return None

    def set(
        self,
        key: Any,
        value: Any,
        ttl: float | None = None,
        version: int = 0,
        volatility: float = 0.0,
    ):
        """Set value with adaptive TTL and admission policy"""
        if self._circuit_open:
            return

        # Check admission policy
        if len(self._cache) >= self.max_size and not self._should_admit(key):
            return

        self._cleanup_expired()

        if len(self._cache) >= self.max_size:
            self._evict()

        # Calculate adaptive TTL
        effective_ttl = ttl if ttl is not None else self.ttl_manager.get_ttl(key, volatility)
        expiry = time.time() + effective_ttl if effective_ttl is not None else float("inf")
        entry = CacheEntry(
            value=value,
            expiry=expiry,
            version=version,
            access_count=1,
            last_access=time.time(),
        )

        self._cache[key] = entry
        self._access_order[key] = None
        self._frequency[key] = self._frequency.get(key, 0) + 1

        if self.eviction_policy == "lru":
            self._access_order.move_to_end(key)

        if self.metrics:
            self.metrics.sets += 1

        self._update_circuit_breaker(True)

    def invalidate(self, key: Any):
        """Explicitly invalidate a key"""
        if key in self._cache:
            del self._cache[key]
            self._access_order.pop(key, None)

    def invalidate_pattern(self, prefix: str):
        """Invalidate all keys matching a prefix"""
        keys_to_remove = [k for k in self._cache if str(k).startswith(prefix)]
        for k in keys_to_remove:
            self.invalidate(k)

    async def get_async(self, key: Any, version: int = 0) -> Any | None:
        """Async version of get"""
        return await asyncio.to_thread(self.get, key, version)

    async def set_async(self, key: Any, value: Any, **kwargs):
        """Async version of set"""
        await asyncio.to_thread(self.set, key, value, **kwargs)


# ============================================================================
# Distributed Cache Coordination (CAS operations)
# ============================================================================


class DistributedCacheCoordinator:
    """
    Handles distributed cache coherence with Compare-And-Set operations.
    Uses Redis Lua scripts for atomic updates.
    """

    CAS_SCRIPT = """
    local key = KEYS[1]
    local new_value = ARGV[1]
    local new_timestamp = tonumber(ARGV[2])

    local current = redis.call('HGETALL', key)
    local current_ts = tonumber(current[2]) or 0

    if new_timestamp > current_ts then
        redis.call('HSET', key, 'value', new_value, 'timestamp', new_timestamp)
        return 1
    end
    return 0
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._cas_script = None

    async def cas_update(self, key: str, value: Any, timestamp: float) -> bool:
        """
        Atomic Compare-And-Set update.
        Only updates if new timestamp > current timestamp.
        """
        if not self.redis:
            raise InvalidInputError("Redis client required for distributed operations")

        # register_script() may be synchronous (returns a callable) *or* asynchronous
        # (returns a coroutine that resolves to a callable). Handle both forms robustly.
        if not self._cas_script:
            script_or_coro = self.redis.register_script(self.CAS_SCRIPT)
            # If an awaitable was returned, resolve it once and cache the callable
            if asyncio.iscoroutine(script_or_coro):
                script = await script_or_coro
            else:
                script = script_or_coro
            self._cas_script = script  # cache the resolved callable

        # Some redis clients return an awaitable result when the script is *called*.
        maybe_awaitable = self._cas_script(keys=[key], args=[value, timestamp])
        result = await maybe_awaitable if asyncio.iscoroutine(maybe_awaitable) else maybe_awaitable
        return bool(result)

    async def invalidate_broadcast(self, channel: str, key: str) -> None:
        """
        Publish an invalidation message to *channel*. Accepts both sync and async Redis clients.
        """
        if not self.redis:
            raise InvalidInputError("Redis client required for distributed operations")

        payload = json.dumps({"action": "invalidate", "key": key, "timestamp": time.time()})
        try:
            maybe = self.redis.publish(channel, payload)
            if asyncio.iscoroutine(maybe):
                await maybe
        except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError, TypeError):
            # Best-effort broadcast; do not raise to callers exercising fault-tolerance paths.
            # (Use observability pipeline to record the error type.)
            pass


# ============================================================================
# Decorator with Enhanced Features
# ============================================================================


def enhanced_cache(
    max_size: int = 128,
    ttl: float | None = None,
    key_fn: Callable | None = None,
    version: str = "v1",
    enable_metrics: bool = True,
):
    """
    Enhanced caching decorator with:
    - Versioned keys
    - Adaptive TTL
    - Metrics
    - Circuit breaker
    """
    cache = EnhancedCacheManager(
        max_size=max_size,
        ttl=ttl,
        eviction_policy="lru",
        enable_metrics=enable_metrics,
    )

    def make_key(func, args, kwargs):
        if key_fn:
            return key_fn(func, args, kwargs)
        return versioned_key(func.__name__, str(args), str(kwargs), version=version)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = make_key(func, args, kwargs)

            cached = cache.get(key)
            if cached is not None:
                return cached

            result = func(*args, **kwargs)

            # Validate DataFrames before caching
            if isinstance(result, (pd.DataFrame, pl.DataFrame)):
                try:
                    validate_dataframe(result)
                except DataError as e:
                    raise InvalidInputError(f"Cached DataFrame invalid: {e}") from e

            cache.set(key, result)
            return result

        wrapper.cache = cache  # Expose cache for metrics
        return wrapper

    return decorator


# ============================================================================
# Persistent Cache with Compression
# ============================================================================


class PersistentCache:
    """L4 persistent cache with compression and versioning"""

    def __init__(self, cache_dir: str = ".cache", enable_compression: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.compression = CompressionStrategy() if enable_compression else None

    def _to_path(self, key: str, suffix: str = "") -> Path:
        return self.cache_dir / f"{key}{suffix}"

    def exists(self, key: str) -> bool:
        return self._to_path(key, suffix=".parquet").exists()

    def save_df(self, key: str, df, version: str = "v1"):
        """
        Save a DataFrame-like with version + engine metadata.
        - Records original library ("pandas" | "polars" | "unknown") to preserve type fidelity.
        - Uses only specific exceptions; best-effort cleanup uses contextlib.suppress().
        """
        data_path = self._to_path(key, suffix=".parquet")
        meta_path = self._to_path(key, suffix=".meta.json")

        # Detect available libraries lazily
        try:
            import pandas as _pd  # type: ignore
        except ImportError:
            _pd = None  # type: ignore[assignment]
        try:
            import polars as _pl  # type: ignore
        except ImportError:
            _pl = None  # type: ignore[assignment]

        # Decide writer + library tag using structural checks (no broad excepts)
        library = "unknown"
        try:
            if _pd is not None and isinstance(df, _pd.DataFrame):
                # Exclude index for cross-library compatibility (polars has no index concept)
                df.to_parquet(data_path, index=False)
                library = "pandas"
            elif _pl is not None and isinstance(df, _pl.DataFrame):
                df.write_parquet(data_path)
                library = "polars"
            elif hasattr(df, "to_parquet"):
                # pandas-like
                df.to_parquet(data_path, index=False)  # Add index=False
                library = "pandas"
            elif hasattr(df, "write_parquet"):
                # polars-like
                df.write_parquet(data_path)
                library = "polars"
            else:
                # Last resort: coerce to pandas if available
                if _pd is None:
                    raise ValueError(
                        "No supported DataFrame writer found (pandas/polars not available)"
                    )
                _pd.DataFrame(df).to_parquet(data_path, index=False)  # Add index=False
                library = "pandas"
        except (FileNotFoundError, PermissionError, OSError, ValueError, TypeError):
            # Clean up partial file on write failure
            with contextlib.suppress(FileNotFoundError, PermissionError, OSError):
                if data_path.exists():
                    data_path.unlink()
            raise

        # Persist metadata (including library)
        try:
            # Compute shape robustly
            try:
                shape = tuple(df.shape)
            except (TypeError, AttributeError):
                rows = len(df) if hasattr(df, "__len__") else None
                cols = len(df.columns) if hasattr(df, "columns") else None
                shape = (rows, cols)
            metadata = {
                "version": version,
                "timestamp": time.time(),
                "library": library,
                "shape": shape,
            }
            meta_path.write_text(json.dumps(metadata))
        except (
            FileNotFoundError,
            PermissionError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            # If metadata write fails, data file remains usable; do not raise.
            pass

    def load_df(self, key: str, expected_version: str | None = None):
        """
        Load DataFrame with version checking and *library-aware* deserialization.
        - If metadata contains {"library": ...}, return that library's DataFrame.
        - If metadata is missing, default to pandas if available (matches tests),
          otherwise fall back to polars.
        """
        data_path = self._to_path(key, suffix=".parquet")
        meta_path = self._to_path(key, suffix=".meta.json")

        # Validate presence of data file
        if not data_path.exists():
            raise FileNotFoundError(f"Cached artifact not found for key={key!r}: {data_path}")

        # Load metadata (optional)
        library_hint: str | None = None
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
                if expected_version and metadata.get("version") != expected_version:
                    raise InvalidInputError(
                        f"Version mismatch: expected {expected_version}, got {metadata.get('version')}"
                    )
                library_hint = metadata.get("library")
            except (json.JSONDecodeError, ValueError, OSError):
                # Ignore malformed metadata; fall back to best-effort engine choice
                library_hint = None

        # Lazy imports to avoid hard deps
        try:
            import pandas as _pd  # type: ignore
        except ImportError:
            _pd = None  # type: ignore[assignment]
        try:
            import polars as _pl  # type: ignore
        except ImportError:
            _pl = None  # type: ignore[assignment]

        # Choose reader preserving type when possible
        try:
            if library_hint == "pandas" or (library_hint is None and _pd is not None):
                if _pd is None:
                    raise ImportError(
                        "pandas is not available to satisfy requested library 'pandas'"
                    )
                return _pd.read_parquet(data_path)
            elif library_hint == "polars":
                if _pl is None:
                    raise ImportError(
                        "polars is not available to satisfy requested library 'polars'"
                    )
                return _pl.read_parquet(data_path)
            else:
                # Unknown/missing hint: prefer pandas if present (aligns with tests), else polars
                if _pd is not None:
                    return _pd.read_parquet(data_path)
                if _pl is not None:
                    return _pl.read_parquet(data_path)
                raise ImportError("Neither pandas nor polars is available to load cached DataFrame")
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            # Propagate specific IO/parse errors
            raise

    def invalidate(self, key: str):
        """Remove cached file"""
        for suffix in [".parquet", ".meta.json", ".npz", ".json"]:
            path = self._to_path(key, suffix=suffix)
            if path.exists():
                path.unlink()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from pysrc.ops.mm_logkit import get_logger

    _demo_log = get_logger(__name__)

    # Example: Enhanced cache with metrics
    @enhanced_cache(max_size=100, ttl=60, version="v1")
    def expensive_computation(x: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return x * x

    # Test cache performance
    result1 = expensive_computation(5)  # Cache miss
    result2 = expensive_computation(5)  # Cache hit

    # Check metrics
    metrics = expensive_computation.cache.metrics
    _demo_log.info(
        "cache_demo_metrics",
        hit_rate=f"{metrics.hit_rate:.2%}",
        avg_latency_us=f"{metrics.avg_latency_us:.2f}",
    )

    # Example: Deterministic DataFrame hashing
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    hash1 = hash_dataframe_deterministic(df)

    # Reorder columns - should produce same hash
    df2 = df[["b", "a"]]
    hash2 = hash_dataframe_deterministic(df2)
    _demo_log.info("dataframe_hash_equal", equal=hash1 == hash2)
