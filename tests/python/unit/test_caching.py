import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path as _Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from pysrc.core.errors import InvalidInputError
from pysrc.ops.caching import (
    AdaptiveTTLManager,
    CacheMetrics,
    CompressionLevel,
    CompressionStrategy,
    DistributedCacheCoordinator,
    EnhancedCacheManager,
    HashAlgorithm,
    PersistentCache,
    enhanced_cache,
    hash_bytes,
    hash_config,
    hash_dataframe_deterministic,
    versioned_key,
)

sys.path.insert(0, str(_Path(__file__).parent.parent / "infra"))
from tests.python.infra.compat_layer import compat
from tests.python.infra.matrix import matrix

# ============================================================================
# Environment Detection for Available Libraries
# ============================================================================


def _detect_xxhash():
    try:
        import xxhash

        xxhash.xxh3_128(b"test")
        return True
    except (ImportError, AttributeError):
        return False


def _detect_blake3():
    try:
        import blake3

        blake3.blake3(b"test")
        return True
    except (ImportError, AttributeError):
        return False


def _detect_zstandard():
    try:
        import zstandard

        zstandard.ZstdCompressor()
        return True
    except (ImportError, AttributeError):
        return False


compat.register("has_xxhash", _detect_xxhash)
compat.register("has_blake3", _detect_blake3)
compat.register("has_zstandard", _detect_zstandard)

# Run detection once at module load
ENVIRONMENT = compat.detect()


# ============================================================================
# Hash Function Tests with Adaptive Matrix
# ============================================================================


class TestHashBytes:
    def test_hash_bytes_xxhash_available(self, monkeypatch):
        if not ENVIRONMENT.get("has_xxhash"):
            pytest.skip("xxhash not available")

        result = hash_bytes(b"test", HashAlgorithm.XXHASH)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_bytes_blake3_available(self, monkeypatch):
        if not ENVIRONMENT.get("has_blake3"):
            pytest.skip("blake3 not available")

        result = hash_bytes(b"test", HashAlgorithm.BLAKE3)
        assert isinstance(result, str)

    def test_hash_bytes_siphash_fallback(self):
        result = hash_bytes(b"test", HashAlgorithm.SIPHASH)
        assert isinstance(result, str)
        assert len(result) == 32

    def test_hash_bytes_sha256_default(self):
        result = hash_bytes(b"test", HashAlgorithm.SHA256)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_bytes_import_error_fallback(self, monkeypatch):
        # Only fail imports for the optional hashing backends; leave the rest
        # of the interpreter and pytest internals untouched.
        import builtins as _builtins

        real_import = _builtins.__import__

        def guarded_import(name, *args, **kwargs):
            # Simulate the optional-backend import failing
            if name in ("xxhash", "blake3"):
                raise ImportError("Mock import error")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(_builtins, "__import__", guarded_import, raising=True)

        # Should gracefully fall back to hashlib in pysrc.ops.caching.hash_bytes
        result = hash_bytes(b"test", HashAlgorithm.XXHASH)
        assert isinstance(result, str)

    @given(data=st.binary(min_size=0, max_size=1000))
    @seed(12345)
    @settings(deadline=None)
    def test_hash_bytes_deterministic(self, data):
        hash1 = hash_bytes(data, HashAlgorithm.SHA256)
        hash2 = hash_bytes(data, HashAlgorithm.SHA256)
        assert hash1 == hash2

    def test_hash_bytes_different_inputs_different_hashes(self):
        hash1 = hash_bytes(b"input1", HashAlgorithm.SHA256)
        hash2 = hash_bytes(b"input2", HashAlgorithm.SHA256)
        assert hash1 != hash2

    @matrix(
        algo=[
            HashAlgorithm.XXHASH,
            HashAlgorithm.BLAKE3,
            HashAlgorithm.SIPHASH,
            HashAlgorithm.SHA256,
        ],
        data_size=[10, 1000, 10000],
        learn=True,
        min_fail_skip=2,
    )
    def test_hash_algorithm_matrix(self, algo, data_size):
        # Skip if algorithm library not available
        if algo == HashAlgorithm.XXHASH and not ENVIRONMENT.get("has_xxhash"):
            pytest.skip("xxhash not available")
        if algo == HashAlgorithm.BLAKE3 and not ENVIRONMENT.get("has_blake3"):
            pytest.skip("blake3 not available")

        data = b"x" * data_size
        result = hash_bytes(data, algo)
        assert isinstance(result, str)
        assert len(result) > 0


class TestHashConfig:
    def test_hash_config_dict(self):
        cfg = {"key": "value", "num": 42}
        result = hash_config(cfg)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_config_deterministic_key_order(self):
        cfg1 = {"a": 1, "b": 2, "c": 3}
        cfg2 = {"c": 3, "b": 2, "a": 1}
        assert hash_config(cfg1) == hash_config(cfg2)

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers()))
    @seed(12345)
    @settings(deadline=None)
    def test_hash_config_property_stable(self, cfg):
        hash1 = hash_config(cfg)
        hash2 = hash_config(cfg)
        assert hash1 == hash2

    def test_hash_config_nested_structures(self):
        cfg = {"outer": {"inner": [1, 2, 3]}, "list": ["a", "b"]}
        result = hash_config(cfg)
        assert isinstance(result, str)


class TestHashDataframeDeterministic:
    def test_hash_dataframe_pandas(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = hash_dataframe_deterministic(df)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_dataframe_polars(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = hash_dataframe_deterministic(df)
        assert isinstance(result, str)

    def test_hash_dataframe_column_order_invariant(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"b": [3, 4], "a": [1, 2]})
        hash1 = hash_dataframe_deterministic(df1)
        hash2 = hash_dataframe_deterministic(df2)
        assert hash1 == hash2

    def test_hash_dataframe_subset_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        hash_all = hash_dataframe_deterministic(df)
        hash_subset = hash_dataframe_deterministic(df, cols=["a", "b"])
        assert hash_all != hash_subset

    def test_hash_dataframe_different_data_different_hash(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        assert hash_dataframe_deterministic(df1) != hash_dataframe_deterministic(df2)

    @pytest.mark.determinism("d1")
    def test_hash_dataframe_handles_array_like_object_cells(self, deterministic_seed: int):
        _ = deterministic_seed
        df = pd.DataFrame(
            {
                "array_value": [np.array([], dtype=float), np.array([1.0, 2.0])],
                "label": ["empty", "values"],
            }
        )

        first = hash_dataframe_deterministic(df)
        second = hash_dataframe_deterministic(df.copy())

        assert isinstance(first, str)
        assert first == second

    @given(st.lists(st.integers(), min_size=1, max_size=10))
    @seed(12345)
    @settings(deadline=None)
    def test_hash_dataframe_property_consistent(self, data):
        df = pd.DataFrame({"col": data})
        hash1 = hash_dataframe_deterministic(df)
        hash2 = hash_dataframe_deterministic(df)
        assert hash1 == hash2


class TestVersionedKey:
    def test_versioned_key_basic(self):
        result = versioned_key("part1", "part2", version="v1")
        assert isinstance(result, str)

    def test_versioned_key_different_versions(self):
        key_v1 = versioned_key("data", version="v1")
        key_v2 = versioned_key("data", version="v2")
        assert key_v1 != key_v2

    def test_versioned_key_multiple_parts(self):
        result = versioned_key("a", "b", "c", "d", version="v3")
        assert isinstance(result, str)

    @given(st.lists(st.text(min_size=1), min_size=1, max_size=5))
    @seed(12345)
    @settings(deadline=None)
    def test_versioned_key_property_deterministic(self, parts):
        key1 = versioned_key(*parts, version="v1")
        key2 = versioned_key(*parts, version="v1")
        assert key1 == key2


# ============================================================================
# Compression Strategy Tests with Matrix
# ============================================================================


class TestCompressionStrategy:
    def test_compression_none_small_data(self):
        strategy = CompressionStrategy(small_threshold=1024)
        data = b"x" * 500
        compressed, level = strategy.compress(data)
        assert level == CompressionLevel.NONE
        assert compressed == data

    def test_compression_fast_medium_data(self):
        strategy = CompressionStrategy(small_threshold=1024, fast_threshold=100_000)
        data = b"x" * 5000
        compressed, level = strategy.compress(data)
        assert level == CompressionLevel.FAST
        assert len(compressed) <= len(data)

    def test_compression_high_large_data(self):
        strategy = CompressionStrategy(small_threshold=1024, fast_threshold=100_000)
        data = b"x" * 200_000
        compressed, level = strategy.compress(data)
        assert level == CompressionLevel.HIGH

    def test_compression_explicit_level(self):
        strategy = CompressionStrategy()
        data = b"test data"
        compressed, level = strategy.compress(data, level=CompressionLevel.FAST)
        assert level == CompressionLevel.FAST

    def test_decompress_none(self):
        strategy = CompressionStrategy()
        data = b"uncompressed"
        decompressed = strategy.decompress(data, CompressionLevel.NONE)
        assert decompressed == data

    @matrix(
        level=[CompressionLevel.NONE, CompressionLevel.FAST, CompressionLevel.HIGH],
        data_size=[100, 1000, 10000],
        learn=True,
        min_fail_skip=2,
    )
    def test_compress_decompress_roundtrip_matrix(self, level, data_size):
        # Skip HIGH with zstandard if not available
        if level == CompressionLevel.HIGH and not ENVIRONMENT.get("has_zstandard"):
            # Will use zlib fallback, which is fine
            pass

        strategy = CompressionStrategy()
        original = b"test data" * (data_size // 9)
        compressed, comp_level = strategy.compress(original, level=level)
        decompressed = strategy.decompress(compressed, comp_level)
        assert decompressed == original

    @given(data=st.binary(min_size=10, max_size=10000))
    @seed(12345)
    @settings(deadline=None)
    def test_compression_roundtrip_property(self, data):
        strategy = CompressionStrategy()
        for level in [CompressionLevel.NONE, CompressionLevel.FAST]:
            compressed, comp_level = strategy.compress(data, level=level)
            decompressed = strategy.decompress(compressed, comp_level)
            assert decompressed == data


# ============================================================================
# Cache Entry and Metrics Tests
# ============================================================================


class TestCacheMetrics:
    def test_metrics_initial_state(self):
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        metrics = CacheMetrics(hits=7, misses=3)
        assert metrics.hit_rate == 0.7

    def test_hit_rate_no_operations(self):
        metrics = CacheMetrics()
        assert metrics.hit_rate == 0.0

    def test_avg_latency_calculation(self):
        metrics = CacheMetrics(hits=5, misses=5, total_latency_ns=10_000_000)
        assert metrics.avg_latency_us == 1000.0

    def test_avg_latency_no_operations(self):
        metrics = CacheMetrics()
        assert metrics.avg_latency_us == 0.0


class TestAdaptiveTTLManager:
    def test_ttl_manager_base_ttl(self):
        manager = AdaptiveTTLManager(base_ttl=300)
        ttl = manager.get_ttl("key", volatility=0.0)
        assert ttl == 300

    def test_ttl_manager_high_volatility_shorter_ttl(self):
        manager = AdaptiveTTLManager(base_ttl=300)
        ttl_low = manager.get_ttl("key", volatility=0.0)
        ttl_high = manager.get_ttl("key", volatility=1.0)
        assert ttl_high < ttl_low

    def test_ttl_manager_update_volatility(self):
        manager = AdaptiveTTLManager(base_ttl=300)
        manager.update_volatility(0.5)
        assert manager.volatility_multiplier < 1.0

    @given(volatility=st.floats(min_value=0.0, max_value=10.0))
    @seed(12345)
    @settings(deadline=None)
    def test_ttl_manager_property_positive_ttl(self, volatility):
        manager = AdaptiveTTLManager(base_ttl=300)
        ttl = manager.get_ttl("key", volatility=volatility)
        assert ttl > 0


# ============================================================================
# Enhanced Cache Manager Tests with Adaptive Matrix
# ============================================================================


class TestEnhancedCacheManager:
    def test_cache_init_default(self):
        cache = EnhancedCacheManager()
        assert cache.max_size == 128
        assert cache.eviction_policy == "lru"

    def test_cache_init_invalid_policy_raises(self):
        with pytest.raises(InvalidInputError, match="Unsupported eviction policy"):
            EnhancedCacheManager(eviction_policy="invalid")

    def test_cache_set_and_get(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager(max_size=10, ttl=60)

        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

    def test_cache_get_miss_returns_none(self):
        cache = EnhancedCacheManager()
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_expiry(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(time, "time", lambda: current_time)

        cache = EnhancedCacheManager(ttl=60)
        cache.set("key1", "value1", ttl=60)

        assert cache.get("key1") == "value1"

        monkeypatch.setattr(time, "time", lambda: current_time + 70)
        assert cache.get("key1") is None

    @matrix(
        eviction_policy=["lru", "lfu", "fifo"],
        max_size=[2, 5, 10],
        enable_compression=[True, False],
        enable_metrics=[True, False],
        learn=True,
        min_fail_skip=2,
    )
    def test_cache_eviction_policies_matrix(
        self, eviction_policy, max_size, enable_compression, enable_metrics, monkeypatch
    ):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager(
            max_size=max_size,
            eviction_policy=eviction_policy,
            enable_compression=enable_compression,
            enable_metrics=enable_metrics,
        )

        # Fill cache beyond capacity
        for i in range(max_size + 2):
            cache.set(f"k{i}", f"v{i}")

        # Cache should respect max_size
        assert len(cache._cache) <= max_size

    def test_cache_version_mismatch(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager()

        cache.set("key", "value", version=1)
        result = cache.get("key", version=2)
        assert result is None

    def test_cache_version_match(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager()

        cache.set("key", "value", version=1)
        result = cache.get("key", version=1)
        assert result == "value"

    def test_cache_invalidate(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager()

        cache.set("key", "value")
        cache.invalidate("key")
        assert cache.get("key") is None

    def test_cache_invalidate_pattern(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager()

        cache.set("user:1", "data1")
        cache.set("user:2", "data2")
        cache.set("post:1", "post_data")

        cache.invalidate_pattern("user:")

        assert cache.get("user:1") is None
        assert cache.get("user:2") is None
        assert cache.get("post:1") == "post_data"

    def test_cache_admission_policy(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager(max_size=2)

        cache.set("k1", "v1")
        cache.set("k2", "v2")

        for _ in range(5):
            cache.get("k1")

        cache.set("k3", "v3")
        assert cache.get("k1") is not None

    def test_cache_circuit_breaker_opens(self):
        cache = EnhancedCacheManager()

        for _ in range(5):
            cache._update_circuit_breaker(False)

        assert cache._circuit_open is True

    def test_cache_circuit_breaker_prevents_operations(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager()

        cache._circuit_open = True
        cache.set("key", "value")
        assert cache.get("key") is None

    def test_cache_circuit_breaker_resets(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(time, "time", lambda: current_time)

        cache = EnhancedCacheManager()
        cache._circuit_open = True
        cache._last_failure_time = current_time - 70

        monkeypatch.setattr(time, "time", lambda: current_time + 70)
        cache._update_circuit_breaker(True)
        assert cache._circuit_open is False

    def test_cache_metrics_tracking(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        monkeypatch.setattr(time, "perf_counter_ns", lambda: 1_000_000_000)

        cache = EnhancedCacheManager(enable_metrics=True)

        cache.set("key", "value")
        cache.get("key")
        cache.get("miss")

        assert cache.metrics.hits == 1
        assert cache.metrics.misses == 1
        assert cache.metrics.sets == 1

    def test_cache_adaptive_ttl_with_volatility(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager()

        cache.set("key", "value", volatility=0.5)
        assert cache.get("key") == "value"

    @pytest.mark.asyncio
    async def test_cache_get_async(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager()

        cache.set("key", "value")
        result = await cache.get_async("key")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_cache_set_async(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager()

        await cache.set_async("key", "value")
        assert cache.get("key") == "value"


# ============================================================================
# Concurrency Tests
# ============================================================================


class TestCacheConcurrency:
    def test_cache_concurrent_reads(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager()
        cache.set("shared", "value")

        def read_cache(n):
            return cache.get("shared")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_cache, i) for i in range(100)]
            wait(futures)
            results = [f.result() for f in futures]

        assert all(r == "value" for r in results)

    def test_cache_concurrent_writes_idempotent(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager(max_size=1000)

        def write_cache(n):
            cache.set(f"key{n}", f"value{n}")
            return cache.get(f"key{n}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_cache, i) for i in range(100)]
            wait(futures)
            results = [f.result() for f in futures]

        assert all(r is not None for r in results)

    def test_cache_concurrent_eviction_safe(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager(max_size=10)

        def churn_cache(n):
            for i in range(20):
                cache.set(f"k{n}_{i}", f"v{n}_{i}")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(churn_cache, i) for i in range(5)]
            wait(futures)

        assert len(cache._cache) <= cache.max_size


# ============================================================================
# Distributed Cache Coordinator Tests
# ============================================================================


class TestDistributedCacheCoordinator:
    def test_coordinator_init_without_redis_raises_on_cas(self):
        coordinator = DistributedCacheCoordinator()

        with pytest.raises(InvalidInputError, match="Redis client required"):
            asyncio.run(coordinator.cas_update("key", "value", 1000.0))

    @pytest.mark.asyncio
    async def test_coordinator_cas_update_success(self):
        mock_redis = AsyncMock()
        mock_script = AsyncMock(return_value=1)
        mock_redis.register_script.return_value = mock_script

        coordinator = DistributedCacheCoordinator(redis_client=mock_redis)
        result = await coordinator.cas_update("key", "value", 1000.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_coordinator_cas_update_failure(self):
        mock_redis = AsyncMock()
        mock_script = AsyncMock(return_value=0)
        mock_redis.register_script.return_value = mock_script

        coordinator = DistributedCacheCoordinator(redis_client=mock_redis)
        result = await coordinator.cas_update("key", "value", 1000.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_coordinator_invalidate_broadcast(self):
        mock_redis = AsyncMock()
        coordinator = DistributedCacheCoordinator(redis_client=mock_redis)

        await coordinator.invalidate_broadcast("channel", "key")

        mock_redis.publish.assert_called_once()


# ============================================================================
# Enhanced Cache Decorator Tests
# ============================================================================


class TestEnhancedCacheDecorator:
    def test_decorator_caches_result(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        call_count = 0

        @enhanced_cache(max_size=10)
        def expensive_fn(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_fn(5)
        result2 = expensive_fn(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1

    def test_decorator_custom_key_function(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)

        def custom_key_fn(func, args, kwargs):
            return f"custom_{args[0]}"

        @enhanced_cache(key_fn=custom_key_fn)
        def fn(x):
            return x * 2

        result = fn(5)
        assert result == 10

    def test_decorator_validates_dataframes(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)

        @enhanced_cache()
        def return_df():
            return pd.DataFrame({"a": [1, 2, 3]})

        result = return_df()
        assert isinstance(result, pd.DataFrame)

    def test_decorator_exposes_cache_metrics(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)

        @enhanced_cache(enable_metrics=True)
        def fn(x):
            return x

        fn(1)
        fn(1)

        assert hasattr(fn, "cache")
        assert fn.cache.metrics.hits >= 1

    def test_decorator_with_version(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)

        @enhanced_cache(version="v2")
        def fn(x):
            return x * 3

        result = fn(5)
        assert result == 15


# ============================================================================
# Persistent Cache Tests
# ============================================================================


class TestPersistentCache:
    def test_persistent_cache_init(self, tmp_path):
        cache = PersistentCache(cache_dir=str(tmp_path / "cache"))
        assert cache.cache_dir.exists()

    def test_persistent_cache_save_and_load_df(self, tmp_path):
        cache = PersistentCache(cache_dir=str(tmp_path / "cache"))
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        cache.save_df("test_key", df, version="v1")
        assert cache.exists("test_key")

        loaded = cache.load_df("test_key")
        assert loaded.shape == df.shape

    def test_persistent_cache_version_check(self, tmp_path):
        cache = PersistentCache(cache_dir=str(tmp_path / "cache"))
        df = pl.DataFrame({"a": [1, 2, 3]})

        cache.save_df("key", df, version="v1")

        with pytest.raises(InvalidInputError, match="Version mismatch"):
            cache.load_df("key", expected_version="v2")

    def test_persistent_cache_invalidate(self, tmp_path):
        cache = PersistentCache(cache_dir=str(tmp_path / "cache"))
        df = pl.DataFrame({"a": [1, 2, 3]})

        cache.save_df("key", df)
        assert cache.exists("key")

        cache.invalidate("key")
        assert not cache.exists("key")

    def test_persistent_cache_metadata_saved(self, tmp_path):
        cache = PersistentCache(cache_dir=str(tmp_path / "cache"))
        df = pl.DataFrame({"a": [1, 2, 3]})

        cache.save_df("key", df, version="v1")

        meta_path = cache._to_path("key", suffix=".meta.json")
        assert meta_path.exists()

        metadata = json.loads(meta_path.read_text())
        assert metadata["version"] == "v1"
        assert "timestamp" in metadata
        assert "shape" in metadata


# ============================================================================
# Property-Based Tests
# ============================================================================


class TestCacheProperties:
    @given(st.integers(min_value=1, max_value=100))
    @seed(12345)
    @settings(deadline=None)
    def test_cache_size_never_exceeds_max(self, max_size):
        # Use no TTL to make test time-independent
        cache = EnhancedCacheManager(max_size=max_size, ttl=None)

        for i in range(max_size * 2):
            cache.set(f"key{i}", f"value{i}")

        assert len(cache._cache) <= max_size

    @given(st.text(min_size=1, max_size=20), st.text(min_size=1))
    @seed(12345)
    @settings(deadline=None)
    def test_cache_get_after_set_returns_value(self, key, value):
        # Time-independent cache for property testing
        cache = EnhancedCacheManager(ttl=None)

        cache.set(key, value)
        result = cache.get(key)
        assert result == value

    @given(
        st.lists(
            st.tuples(st.text(min_size=1, max_size=10), st.integers()), min_size=1, max_size=20
        )
    )
    @seed(12345)
    @settings(deadline=None)
    def test_cache_invalidate_removes_key(self, items):
        # Time-independent cache for pure property testing
        cache = EnhancedCacheManager(max_size=100, ttl=None)

        for key, value in items:
            cache.set(key, value)

        for key, _ in items[: len(items) // 2]:
            cache.invalidate(key)
            assert cache.get(key) is None


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================


class TestEdgeCases:
    def test_cache_empty_operations(self):
        cache = EnhancedCacheManager()
        assert cache.get("nonexistent") is None
        cache.invalidate("nonexistent")

    def test_cache_zero_ttl(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(time, "time", lambda: current_time)
        cache = EnhancedCacheManager()

        cache.set("key", "value", ttl=0)

        monkeypatch.setattr(time, "time", lambda: current_time + 0.1)
        assert cache.get("key") is None

    def test_cache_negative_ttl(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager()

        cache.set("key", "value", ttl=-10)
        assert cache.get("key") is None

    def test_hash_empty_bytes(self):
        result = hash_bytes(b"", HashAlgorithm.SHA256)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_empty_config(self):
        result = hash_config({})
        assert isinstance(result, str)

    def test_compression_empty_data(self):
        strategy = CompressionStrategy()
        compressed, level = strategy.compress(b"")
        decompressed = strategy.decompress(compressed, level)
        assert decompressed == b""

    def test_persistent_cache_nonexistent_key(self, tmp_path):
        cache = PersistentCache(cache_dir=str(tmp_path / "cache"))
        assert not cache.exists("nonexistent")

    def test_versioned_key_empty_parts(self):
        result = versioned_key(version="v1")
        assert isinstance(result, str)


# ============================================================================
# Performance Tests (marked to skip by default)
# ============================================================================


@pytest.mark.perf
class TestCachePerformance:
    @pytest.mark.skip(reason="Performance test - enable manually")
    def test_cache_throughput_baseline(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        cache = EnhancedCacheManager(max_size=10000)

        for i in range(1000):
            cache.set(f"key{i}", f"value{i}")

        start = time.perf_counter()
        for i in range(10000):
            cache.get(f"key{i % 1000}")
        duration = time.perf_counter() - start

        ops_per_sec = 10000 / duration
        assert ops_per_sec > 100000

    @pytest.mark.skip(reason="Performance test - enable manually")
    def test_hash_dataframe_scaling(self):
        sizes = [100, 1000, 10000]
        times = []

        for size in sizes:
            df = pd.DataFrame({"a": range(size), "b": range(size)})
            start = time.perf_counter()
            hash_dataframe_deterministic(df)
            times.append(time.perf_counter() - start)

        assert times[1] / times[0] < 10
        assert times[2] / times[1] < 10
