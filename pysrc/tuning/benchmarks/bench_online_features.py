"""Benchmark: measure OnlineFeatureBuffer push and snapshot throughput."""

from __future__ import annotations

import time

from pysrc.tuning.live.online_features import OnlineFeatureBuffer


def bench_online_buffer(
    capacity: int = 500,
    n_pushes: int = 10_000,
) -> dict[str, float]:
    """Measure time to push n_pushes observations and take snapshots."""
    buf = OnlineFeatureBuffer(capacity=capacity)
    obs = {"close": 100.0, "volume": 1_000_000.0, "rsi": 55.0}
    start = time.perf_counter()
    for i in range(n_pushes):
        buf.push(obs)
        if i % capacity == 0:
            buf.snapshot()
    elapsed = time.perf_counter() - start
    return {
        "capacity": float(capacity),
        "n_pushes": float(n_pushes),
        "elapsed_s": elapsed,
        "pushes_per_sec": n_pushes / elapsed,
    }


__all__ = ["bench_online_buffer"]
