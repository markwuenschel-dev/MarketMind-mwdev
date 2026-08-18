"""Evolutionary search primitives: selection, crossover, and mutation."""

from __future__ import annotations

import hashlib
import struct
from typing import Any


def tournament_select(
    population: list[tuple[dict[str, Any], float]],
    k: int,
    seed: int,
) -> dict[str, Any]:
    """Select the best individual from a random tournament of size k."""
    n = len(population)
    if n == 0:
        raise ValueError("Cannot select from an empty population")
    selected: list[int] = []
    for i in range(min(k, n)):
        raw = struct.pack(">QQ", seed & 0xFFFFFFFFFFFFFFFF, i)
        idx = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % n
        selected.append(idx)
    best_idx = max(selected, key=lambda i: population[i][1])
    return population[best_idx][0]


def uniform_crossover(
    parent_a: dict[str, Any],
    parent_b: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Combine two parents by randomly selecting each gene from either parent."""
    child: dict[str, Any] = {}
    for i, key in enumerate(parent_a):
        raw = struct.pack(">QQ", seed & 0xFFFFFFFFFFFFFFFF, i)
        bit = int.from_bytes(hashlib.sha256(raw).digest()[:1], "big") % 2
        child[key] = parent_a[key] if bit == 0 else parent_b.get(key, parent_a[key])
    return child


__all__ = ["tournament_select", "uniform_crossover"]
