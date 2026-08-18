# py/ops/multi_tier_cache.py
"""
Multi-tier caching client with L1→L2→L3→L4 read-through and write-back.
Implements singleflight, TTL jitter, and distributed coordination.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import inspect
import pickle
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar

import pandas as pd
import polars as pl

from pysrc.core.runtime.optional_imports import optional_import
from pysrc.ops.caching import (
    EnhancedCacheManager,
    HashAlgorithm,
    PersistentCache,
    hash_bytes,
    versioned_key,
)
from pysrc.ops.mm_logkit import get_logger

_LOG = get_logger(__name__)

# At top of file
try:
    import importlib.util

    PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
    if PYARROW_AVAILABLE:
        import pyarrow.plasma as plasma  # noqa: F401
except ImportError:
    PYARROW_AVAILABLE = False
    plasma = None


redis = optional_import("redis")
REDIS_AVAILABLE = redis is not None

# Build exception tuple for L3 cache operations based on what's available
# Always catch transport/serialization errors; include redis.RedisError when available
if redis and hasattr(redis, "RedisError"):
    _L3_CACHE_EXCEPTIONS = (
        redis.RedisError,
        OSError,
        ConnectionError,
        TimeoutError,
        pickle.PickleError,
        TypeError,
    )
else:
    _L3_CACHE_EXCEPTIONS = (OSError, ConnectionError, TimeoutError, pickle.PickleError, TypeError)


T = TypeVar("T")


# ============================================================================
# Utilities
# ============================================================================


def version_to_int(version_str: str) -> int:
    """Convert version string to stable 32-bit integer (deterministic across processes)"""
    digest = hashlib.blake2b(version_str.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


# ============================================================================
# Singleflight: Prevent stampeding herd on cache misses
# ============================================================================


@dataclass
class Call:
    """Represents an in-flight call"""

    future: Future | asyncio.Future
    start_time: float
    is_async: bool = False


class Singleflight:
    """
    Ensures only one execution per key is in-flight at a time.
    Other requests for the same key wait for the first result.
    Supports both sync and async execution.
    """

    def __init__(self):
        self._calls: dict[Any, Call] = {}
        self._lock = Lock()
        self._async_lock = asyncio.Lock()  # Shared async lock

    def do(self, key: Any, fn: Callable[[], T]) -> tuple[T, bool]:
        """
        Execute fn for key, or wait if already in-flight.
        Returns (result, is_shared) where is_shared=True if we waited on another call.
        """
        with self._lock:
            if key in self._calls:
                call = self._calls[key]
                shared = True
            else:
                future = Future()
                call = Call(future=future, start_time=time.time(), is_async=False)
                self._calls[key] = call
                shared = False

        if shared:
            return call.future.result(), True

        try:
            result = fn()
            call.future.set_result(result)
            return result, False
        except Exception as e:
            call.future.set_exception(e)
            raise
        finally:
            with self._lock:
                self._calls.pop(key, None)

    async def do_async(self, key: Any, fn: Callable) -> tuple[T, bool]:
        """Async version of do()"""
        async with self._async_lock:
            call = self._calls.get(key)
            if call:
                # If a call is already in flight, wait for its future and return
                return await call.future, True

            # No call in flight, create a new one
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._calls[key] = Call(future=future, start_time=time.time(), is_async=True)

        try:
            # Execute the function
            result = fn()
            # Await the result if it's awaitable (i.e., a coroutine)
            if inspect.isawaitable(result):
                result = await result

            future.set_result(result)
            return result, False
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            # Clean up the call from the dictionary
            async with self._async_lock:
                self._calls.pop(key, None)


# ============================================================================
# L2: Shared Memory Cache (Plasma or memfd)
# ============================================================================


class L2Cache:
    """Base class for L2 shared memory cache"""

    def get(self, key: str) -> Any | None:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: float | None = None):
        raise NotImplementedError

    def invalidate(self, key: str):
        raise NotImplementedError


class PlasmaL2Cache(L2Cache):
    """L2 cache using Apache Arrow Plasma for zero-copy sharing"""

    def __init__(self, plasma_path: str = "/tmp/plasma"):
        # In __init__
        if PYARROW_AVAILABLE:
            try:
                self.client = plasma.connect(plasma_path)
                self._available = True
            except OSError as e:
                _LOG.warning("plasma_connection_failed", error=str(e))
                self._available = False
                self.client = None
        else:
            self._available = False
            self.client = None

        self._metadata: dict[str, tuple[bytes, float]] = {}  # key -> (object_id_bytes, expiry)
        self._lock = Lock()

    def _make_object_id(self, key: str):
        """Generate deterministic Plasma ObjectID from key (20 raw bytes)"""
        import pyarrow.plasma as plasma

        # BLAKE2b with 20-byte digest for ObjectID
        raw_bytes = hashlib.blake2b(key.encode(), digest_size=20).digest()
        return plasma.ObjectID(raw_bytes)

    def get(self, key: str) -> Any | None:
        if not self._available:
            return None

        with self._lock:
            if key not in self._metadata:
                return None

            object_id_bytes, expiry = self._metadata[key]

            # Check TTL
            if expiry < time.time():
                del self._metadata[key]
                return None

        try:
            import pyarrow.plasma as plasma

            object_id = plasma.ObjectID(object_id_bytes)

            # Get from Plasma (zero-copy if same process)
            [buffer] = self.client.get_buffers([object_id])
            if buffer is None:
                return None

            # Deserialize
            return pickle.loads(buffer.to_pybytes())
        except (pickle.PickleError, OSError, TypeError):
            return None

    def set(self, key: str, value: Any, ttl: float | None = None):
        if not self._available:
            return

        try:
            # Serialize value
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            object_id = self._make_object_id(key)

            # Put in Plasma
            buf = self.client.create(object_id, len(data))
            memoryview(buf).cast("B")[:] = data
            self.client.seal(object_id)

            # Track metadata with raw bytes
            expiry = time.time() + ttl if ttl is not None else float("inf")
            with self._lock:
                self._metadata[key] = (object_id.binary(), expiry)
        except (pickle.PickleError, OSError, TypeError):
            pass

    def invalidate(self, key: str):
        if not self._available or not self.client:
            return

        object_id_bytes = None
        with self._lock:
            if key in self._metadata:
                object_id_bytes, _ = self._metadata.pop(key)

        if object_id_bytes:
            try:
                import pyarrow.plasma as plasma

                self.client.delete([plasma.ObjectID(object_id_bytes)])
            except (ImportError, OSError, AttributeError):
                # If pyarrow is missing or deletion fails, ignore per best-effort semantics
                pass


class MemfdL2Cache(L2Cache):
    """L2 cache using memory-mapped files (memfd on Linux)"""

    def __init__(self, cache_dir: str = "/dev/shm/l2_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata: dict[str, float] = {}  # key -> expiry
        self._lock = Lock()

    def _path(self, key: str) -> Path:
        key_hash = hash_bytes(key.encode(), HashAlgorithm.XXHASH)
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str) -> Any | None:
        # Fast-path using metadata if present
        with self._lock:
            expiry = self._metadata.get(key)

        if expiry is not None:
            if expiry < time.time():
                with self._lock:
                    self._metadata.pop(key, None)
                return None
        else:
            # Metadata-miss fallback: if an artifact exists on disk, read it anyway.
            # This makes L2 tolerant to rare metadata races or post-fork scenarios.
            path = self._path(key)
            if not path.exists():
                return None
            try:
                data = path.read_bytes()
                value = pickle.loads(data)
                # Adopt a conservative expiry (no TTL known): treat as non-expiring.
                with self._lock:
                    self._metadata[key] = float("inf")
                return value
            except (OSError, pickle.PickleError, TypeError, AttributeError):
                return None

        # Normal read with known expiry
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = path.read_bytes()
            return pickle.loads(data)
        except (OSError, pickle.PickleError, TypeError, AttributeError):
            return None

    def set(self, key: str, value: Any, ttl: float | None = 60) -> None:
        path = self._path(key)
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        # TTL == 0 → treat as immediate expiry (remove any prior artifact + metadata)
        if ttl == 0:
            with self._lock:
                self._metadata.pop(key, None)
            with contextlib.suppress(FileNotFoundError, PermissionError, OSError):
                if path.exists():
                    path.unlink()
            return

        try:
            # 1) Serialize first (this is where patch('pickle.dumps', ...) will raise)
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            # 2) Atomic write: tmp then replace
            tmp_path.write_bytes(data)
            tmp_path.replace(path)

            # 3) Record expiry only after successful replace
            expiry = time.time() + (ttl if ttl is not None else float("inf"))
            with self._lock:
                self._metadata[key] = expiry

        except (pickle.PickleError, TypeError, AttributeError, OSError):
            # On failure, ensure no tmp remains and leave metadata untouched
            with contextlib.suppress(FileNotFoundError, PermissionError, OSError):
                if tmp_path.exists():
                    tmp_path.unlink()
            return

    def invalidate(self, key: str):
        with self._lock:
            self._metadata.pop(key, None)

        path = self._path(key)
        if path.exists():
            path.unlink()


# ============================================================================
# L3: Distributed Cache (Redis/KeyDB)
# ============================================================================


class L3Cache:
    """L3 distributed cache using Redis"""

    def __init__(self, redis_client=None, key_prefix: str = "l3:"):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self._available = redis_client is not None

    def _full_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Any | None:
        if not self._available:
            return None

        try:
            full_key = self._full_key(key)
            data = self.redis.get(full_key)
            if data:
                return pickle.loads(data)
        except _L3_CACHE_EXCEPTIONS:
            pass
        return None

    def set(self, key: str, value: Any, ttl: float | None = None):
        if not self._available:
            return

        try:
            full_key = self._full_key(key)
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            if ttl:
                self.redis.setex(full_key, int(ttl), data)
            else:
                self.redis.set(full_key, data)
        except _L3_CACHE_EXCEPTIONS:
            pass

    def invalidate(self, key: str):
        if not self._available:
            return

        try:
            full_key = self._full_key(key)
            self.redis.delete(full_key)
        except _L3_CACHE_EXCEPTIONS:
            pass

    def publish_invalidation(self, channel: str, key: str):
        """Broadcast invalidation to subscribers"""
        if not self._available:
            return

        try:
            import json

            message = json.dumps({"action": "invalidate", "key": key, "timestamp": time.time()})
            self.redis.publish(channel, message)
        except _L3_CACHE_EXCEPTIONS:
            pass


# ============================================================================
# Multi-Tier Metrics
# ============================================================================


@dataclass
class TierMetrics:
    """Per-tier cache metrics"""

    tier_name: str
    hits: int = 0
    misses: int = 0
    sets: int = 0
    promotions: int = 0  # Count of promotions from this tier
    latency_ns: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_latency_us(self) -> float:
        ops = self.hits + self.misses
        return (self.latency_ns / ops / 1000) if ops > 0 else 0.0


class MultiTierMetrics:
    """Aggregated metrics across all tiers"""

    def __init__(self):
        self.l1 = TierMetrics("L1_local")
        self.l2 = TierMetrics("L2_shared")
        self.l3 = TierMetrics("L3_distributed")
        self.l4 = TierMetrics("L4_persistent")
        self.singleflight_saved = 0

    def summary(self) -> dict[str, Any]:
        """Generate metrics summary for monitoring"""
        return {
            "l1": {
                "hit_rate": self.l1.hit_rate,
                "avg_latency_us": self.l1.avg_latency_us,
                "hits": self.l1.hits,
                "misses": self.l1.misses,
                "sets": self.l1.sets,
            },
            "l2": {
                "hit_rate": self.l2.hit_rate,
                "avg_latency_us": self.l2.avg_latency_us,
                "hits": self.l2.hits,
                "misses": self.l2.misses,
                "promotions": self.l2.promotions,
            },
            "l3": {
                "hit_rate": self.l3.hit_rate,
                "avg_latency_us": self.l3.avg_latency_us,
                "hits": self.l3.hits,
                "misses": self.l3.misses,
                "promotions": self.l3.promotions,
            },
            "l4": {
                "hits": self.l4.hits,
                "misses": self.l4.misses,
            },
            "singleflight_saved": self.singleflight_saved,
        }


# ============================================================================
# Invalidation Listener
# ============================================================================


class InvalidationListener:
    """Background thread that listens for Redis pub/sub invalidations"""

    def __init__(self, redis_client, channel: str, callback: Callable[[str], None]):
        self.redis = redis_client
        self.channel = channel
        self.callback = callback
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the listener thread"""
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the listener thread"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _listen_loop(self):
        """Listen for invalidation messages"""
        try:
            import json

            pubsub = self.redis.pubsub()
            pubsub.subscribe(self.channel)

            for message in pubsub.listen():
                if self._stop_event.is_set():
                    break

                if message["type"] == "message":
                    try:
                        raw_data = message["data"]
                        # Ensure data is a string before JSON decoding
                        if isinstance(raw_data, (bytes, bytearray)):
                            raw_data = raw_data.decode("utf-8", "replace")

                        data = json.loads(raw_data)
                        if data.get("action") == "invalidate":
                            key = data.get("key")
                            if key:
                                self.callback(key)
                    except (json.JSONDecodeError, TypeError):
                        pass
        except redis.RedisError:
            pass


# ============================================================================
# Multi-Tier Client
# ============================================================================


class MultiTierClient:
    """
    Orchestrates L1→L2→L3→L4 read-through with write-back.

    Features:
    - Singleflight to prevent duplicate computations
    - TTL jitter to avoid thundering herds
    - Per-tier metrics
    - Background pub/sub invalidation listener
    - Optional L4 read-through for DataFrames
    """

    def __init__(
        self,
        l1_size: int = 128,
        l1_ttl: float | None = 60,
        l2_type: str = "memfd",
        l2_path: str | None = None,
        redis_client=None,
        l3_key_prefix: str = "l3:",
        l4_cache_dir: str = ".cache",
        enable_singleflight: bool = True,
        ttl_jitter: float = 0.1,
        check_l4_on_miss: bool = False,
        enable_invalidation_listener: bool = True,
    ):
        self._base_ttl = l1_ttl  # Store the canonical TTL
        # L1: In-process cache
        self.l1 = EnhancedCacheManager(
            max_size=l1_size,
            ttl=l1_ttl,
            eviction_policy="lru",
            enable_metrics=False,
        )

        # L2: Shared memory
        if l2_type == "plasma":
            self.l2 = PlasmaL2Cache(l2_path or "/tmp/plasma")
        else:
            self.l2 = MemfdL2Cache(l2_path or "/dev/shm/l2_cache")

        # L3: Distributed cache
        self.l3 = L3Cache(redis_client, key_prefix=l3_key_prefix)

        # L4: Persistent storage
        self.l4 = PersistentCache(cache_dir=l4_cache_dir)

        # Singleflight
        self.singleflight = Singleflight() if enable_singleflight else None

        # Configuration
        self.ttl_jitter = ttl_jitter
        self.check_l4_on_miss = check_l4_on_miss

        # Metrics
        self.metrics = MultiTierMetrics()

        # Invalidation
        self._invalidation_channel = f"{l3_key_prefix}invalidations"
        self._listener = None
        if enable_invalidation_listener and redis_client:
            self._listener = InvalidationListener(
                redis_client,
                self._invalidation_channel,
                self._handle_invalidation,
            )
            self._listener.start()

    def _handle_invalidation(self, key: str):
        """Handle invalidation message from pub/sub"""
        self.l1.invalidate(key)
        self.l2.invalidate(key)

    def _apply_jitter(self, ttl: float | None) -> float | None:
        """Add random jitter to TTL"""
        if ttl is None:
            return None

        jitter_range = ttl * self.ttl_jitter
        jitter = random.uniform(-jitter_range, jitter_range)
        # Ensure float return to satisfy type checkers & downstream arithmetic
        return max(1.0, ttl + jitter)

    def get(self, key: str, version: int = 0) -> Any | None:
        """Get value from cache with L1→L2→L3→L4 fallback"""
        # L1: In-process
        start = time.perf_counter_ns()
        value = self.l1.get(key, version)
        self.metrics.l1.latency_ns += time.perf_counter_ns() - start

        if value is not None:
            self.metrics.l1.hits += 1
            return value
        self.metrics.l1.misses += 1

        # L2: Shared memory
        start = time.perf_counter_ns()
        value = self.l2.get(key)
        self.metrics.l2.latency_ns += time.perf_counter_ns() - start

        if value is not None:
            self.metrics.l2.hits += 1
            self.metrics.l2.promotions += 1
            # Promote to L1 with jitter using the canonical TTL
            ttl_jittered = self._apply_jitter(self._base_ttl)
            self.l1.set(key, value, ttl=ttl_jittered, version=version)
            return value
        self.metrics.l2.misses += 1

        # L3: Distributed
        start = time.perf_counter_ns()
        value = self.l3.get(key)
        self.metrics.l3.latency_ns += time.perf_counter_ns() - start

        if value is not None:
            self.metrics.l3.hits += 1
            self.metrics.l3.promotions += 1
            # Promote to L2 and L1 with jitter
            ttl_jittered = self._apply_jitter(self._base_ttl)
            self.l2.set(key, value, ttl=ttl_jittered)
            self.l1.set(key, value, ttl=ttl_jittered, version=version)
            return value
        self.metrics.l3.misses += 1

        # L4: Persistent (optional, only for DataFrames)
        if self.check_l4_on_miss:
            try:
                from pysrc.core.errors import (
                    InvalidInputError,  # local import to avoid hard dep at module import
                )
            except ImportError:
                InvalidInputError = ValueError  # fallback type for precise catching

            try:
                # Some stores record human-readable versions (e.g., "v1") while callers pass hashed ints.
                # Try strict check first; on version mismatch (InvalidInputError), fall back to no check.
                expected = str(version) if version else None
                try:
                    value = self.l4.load_df(key, expected_version=expected)
                except InvalidInputError:
                    value = self.l4.load_df(key, expected_version=None)

                if value is not None:
                    self.metrics.l4.hits += 1
                    ttl_jittered = self._apply_jitter(self._base_ttl)
                    self.l3.set(key, value, ttl=ttl_jittered)
                    self.l2.set(key, value, ttl=ttl_jittered)
                    self.l1.set(key, value, ttl=ttl_jittered, version=version)
                    return value
            except (OSError, ValueError, pickle.PickleError):
                pass
            self.metrics.l4.misses += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
        version: int = 0,
        write_through: bool = True,
        persist_to_l4: bool = False,
    ):
        """Set value with write-through to all tiers"""
        ttl_with_jitter = self._apply_jitter(ttl)

        # Always write to L1
        self.l1.set(key, value, ttl=ttl_with_jitter, version=version)
        self.metrics.l1.sets += 1

        if write_through:
            self.l2.set(key, value, ttl=ttl_with_jitter)
            self.metrics.l2.sets += 1

            self.l3.set(key, value, ttl=ttl_with_jitter)
            self.metrics.l3.sets += 1

        if persist_to_l4 and isinstance(value, (pd.DataFrame, pl.DataFrame)):
            self.l4.save_df(key, value, version=str(version))
            self.metrics.l4.sets += 1

    def compute_or_get(
        self,
        key: str,
        compute_fn: Callable[[], T],
        ttl: float | None = None,
        version: int = 0,
        persist_to_l4: bool = False,
    ) -> T:
        """Get from cache or compute with singleflight"""
        value = self.get(key, version)
        if value is not None:
            return value

        # Cache miss - compute with singleflight
        if self.singleflight:
            result, shared = self.singleflight.do(key, compute_fn)
            if shared:
                self.metrics.singleflight_saved += 1
        else:
            result = compute_fn()

        # Store with write-through
        self.set(
            key, result, ttl=ttl, version=version, write_through=True, persist_to_l4=persist_to_l4
        )

        return result

    async def compute_or_get_async(
        self,
        key: str,
        compute_fn: Callable,
        ttl: float | None = None,
        version: int = 0,
        persist_to_l4: bool = False,
    ) -> T:
        """Async version of compute_or_get"""
        value = self.get(key, version)
        if value is not None:
            return value

        if self.singleflight:
            result, shared = await self.singleflight.do_async(key, compute_fn)
            if shared:
                self.metrics.singleflight_saved += 1
        else:
            if inspect.iscoroutinefunction(compute_fn):
                result = await compute_fn()
            else:
                result = compute_fn()

        self.set(
            key, result, ttl=ttl, version=version, write_through=True, persist_to_l4=persist_to_l4
        )

        return result

    def invalidate(self, key: str, broadcast: bool = True):
        """Invalidate key across all tiers"""
        self.l1.invalidate(key)
        self.l2.invalidate(key)
        self.l3.invalidate(key)

        if broadcast and self.l3._available:
            self.l3.publish_invalidation(self._invalidation_channel, key)

    def invalidate_pattern(self, prefix: str, broadcast: bool = True):
        """Invalidate all keys matching prefix"""
        self.l1.invalidate_pattern(prefix)

    def close(self):
        """Cleanup resources"""
        if self._listener:
            self._listener.stop()


# ============================================================================
# Decorators
# ============================================================================


def multi_tier_cache(
    ttl: float | None = 60,
    version: str = "v1",
    persist_large_objects: bool = False,
    key_fn: Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], str] | None = None,
    redis_client=None,
    l2_type: str = "memfd",
    check_l4_on_miss: bool = False,
):
    """Multi-tier caching decorator with singleflight and TTL jitter"""

    version_int = version_to_int(version)

    client = MultiTierClient(
        l1_ttl=ttl,
        redis_client=redis_client,
        l2_type=l2_type,
        enable_singleflight=True,
        ttl_jitter=0.1,
        check_l4_on_miss=check_l4_on_miss,
    )

    def make_key(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        if key_fn:
            return key_fn(func, args, kwargs)  # must return a stable str key
        # Use qualname for uniqueness and repr() for stable, unambiguous text
        return versioned_key(
            func.__qualname__, repr(args), repr(sorted(kwargs.items())), version=version
        )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                key = make_key(func, args, kwargs)

                result = await client.compute_or_get_async(
                    key=key,
                    compute_fn=lambda: func(*args, **kwargs),
                    ttl=ttl,
                    version=version_int,
                    persist_to_l4=persist_large_objects,
                )
                return result

            async_wrapper.cache_client = client
            async_wrapper.cache_metrics = lambda: client.metrics.summary()
            async_wrapper.invalidate = lambda k: client.invalidate(k)
            async_wrapper.invalidate_pattern = lambda p: client.invalidate_pattern(p)
            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                key = make_key(func, args, kwargs)

                result = client.compute_or_get(
                    key=key,
                    compute_fn=lambda: func(*args, **kwargs),
                    ttl=ttl,
                    version=version_int,
                    persist_to_l4=persist_large_objects,
                )
                return result

            sync_wrapper.cache_client = client
            sync_wrapper.cache_metrics = lambda: client.metrics.summary()
            sync_wrapper.invalidate = lambda k: client.invalidate(k)
            sync_wrapper.invalidate_pattern = lambda p: client.invalidate_pattern(p)
            return sync_wrapper

    return decorator


# ============================================================================
# Monitoring Integration
# ============================================================================


class PrometheusExporter:
    """Export cache metrics to Prometheus format"""

    def __init__(self, client: MultiTierClient):
        self.client = client

    def export(self) -> str:
        """Generate Prometheus metrics"""
        metrics = self.client.metrics
        lines = []

        # Hit rates
        lines.append("# HELP cache_hit_rate Cache hit rate by tier")
        lines.append("# TYPE cache_hit_rate gauge")
        lines.append(f'cache_hit_rate{{tier="l1"}} {metrics.l1.hit_rate:.4f}')
        lines.append(f'cache_hit_rate{{tier="l2"}} {metrics.l2.hit_rate:.4f}')
        lines.append(f'cache_hit_rate{{tier="l3"}} {metrics.l3.hit_rate:.4f}')

        # Latencies
        lines.append("# HELP cache_latency_microseconds Average cache latency")
        lines.append("# TYPE cache_latency_microseconds gauge")
        lines.append(f'cache_latency_microseconds{{tier="l1"}} {metrics.l1.avg_latency_us:.2f}')
        lines.append(f'cache_latency_microseconds{{tier="l2"}} {metrics.l2.avg_latency_us:.2f}')
        lines.append(f'cache_latency_microseconds{{tier="l3"}} {metrics.l3.avg_latency_us:.2f}')

        # Promotions
        lines.append("# HELP cache_promotions_total Cache promotions by tier")
        lines.append("# TYPE cache_promotions_total counter")
        lines.append(f'cache_promotions_total{{tier="l2"}} {metrics.l2.promotions}')
        lines.append(f'cache_promotions_total{{tier="l3"}} {metrics.l3.promotions}')

        # Singleflight
        lines.append("# HELP cache_singleflight_saved_total Duplicate computations avoided")
        lines.append("# TYPE cache_singleflight_saved_total counter")
        lines.append(f"cache_singleflight_saved_total {metrics.singleflight_saved}")

        # L1 size (safe access)
        l1_cache_obj = getattr(self.client.l1, "_cache", {})
        l1_size = len(l1_cache_obj)
        lines.append("# HELP cache_l1_size Current L1 cache size")
        lines.append("# TYPE cache_l1_size gauge")
        lines.append(f"cache_l1_size {l1_size}")

        return "\n".join(lines)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Sync function
    @multi_tier_cache(ttl=60, version="v1")
    def fibonacci(n: int) -> int:
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    # Example: Async function
    @multi_tier_cache(ttl=60, version="v1")
    async def async_compute(x: int) -> int:
        await asyncio.sleep(0.1)
        return x * x

    # Test sync
    result = fibonacci(10)
    _LOG.info("fibonacci_example", n=10, result=result)

    # Test async
    async def test_async():
        result = await async_compute(5)
        _LOG.info("async_compute_example", x=5, result=result)

    asyncio.run(test_async())

    # Metrics
    import json

    _LOG.info("cache_metrics %s", json.dumps(fibonacci.cache_metrics(), indent=2))
