"""Pure deterministic sampler: draws candidates from a SearchSpace given a seed."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

from pysrc.tuning.core.search.space import SearchSpace


def _hash_to_unit(seed: int, index: int) -> float:
    """Deterministically map (seed, index) to float in [0, 1)."""
    raw = struct.pack(">QQ", seed & 0xFFFFFFFFFFFFFFFF, index & 0xFFFFFFFFFFFFFFFF)
    h = hashlib.sha256(raw).digest()
    return float(struct.unpack(">Q", h[:8])[0]) / (2**64)


def sample_uniform(space: SearchSpace, seed: int, n: int) -> list[dict[str, Any]]:
    """Draw n candidates uniformly from the search space using a deterministic seed."""
    results: list[dict[str, Any]] = []
    for i in range(n):
        candidate: dict[str, Any] = dict(space.fixed)
        for j, dim in enumerate(space.dimensions):
            u = _hash_to_unit(seed, i * len(space.dimensions) + j)
            if dim.kind == "categorical":
                choices = list(dim.choices or [])
                if not choices:
                    raise ValueError(f"Categorical dimension {dim.name!r} has no choices")
                candidate[dim.name] = choices[int(u * len(choices))]
            elif dim.kind == "int":
                lo, hi = int(dim.low or 0), int(dim.high or 1)
                candidate[dim.name] = lo + int(u * (hi - lo + 1))
            elif dim.kind == "log_real":
                log_lo = math.log(dim.low or 1e-6)
                log_hi = math.log(dim.high or 1.0)
                candidate[dim.name] = math.exp(log_lo + u * (log_hi - log_lo))
            else:
                lo2, hi2 = dim.low or 0.0, dim.high or 1.0
                candidate[dim.name] = lo2 + u * (hi2 - lo2)
        results.append(candidate)
    return results


__all__ = ["sample_uniform"]
