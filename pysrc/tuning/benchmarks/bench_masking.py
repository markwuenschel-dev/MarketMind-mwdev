"""Benchmark: measure SlotMask construction and active-index lookup throughput."""

from __future__ import annotations

import time

from pysrc.tuning.core.ir.masking import SlotMask


def bench_slot_mask(
    total_slots: int = 1000,
    k: int = 50,
    n_iterations: int = 10_000,
) -> dict[str, float]:
    """Measure time to create and query a SlotMask n_iterations times."""
    start = time.perf_counter()
    for _ in range(n_iterations):
        mask = SlotMask.from_k(total_slots, k)
        mask.active_indices()
    elapsed = time.perf_counter() - start
    return {
        "total_slots": float(total_slots),
        "k": float(k),
        "n_iterations": float(n_iterations),
        "elapsed_s": elapsed,
        "iterations_per_sec": n_iterations / elapsed,
    }


__all__ = ["bench_slot_mask"]
