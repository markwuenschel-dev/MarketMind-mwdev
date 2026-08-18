# utils/plan_costs.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, reduce
from time import perf_counter
from typing import Any

from pysrc.ops.mm_logkit import get_logger

from .cuda_runtime import capabilities
from .errors import OOMRetry
from .specs import Spec

logger = get_logger(__name__)


@dataclass
class PlanSegment:
    ops: list[Callable]
    spec: Spec | None = None
    estimated_cost: float = 0.0

    def __hash__(self) -> int:
        return hash((tuple(map(id, self.ops)), hash(self.spec) if self.spec else 0))


@lru_cache(maxsize=128)
def estimate_compute_cost(op: Callable, backend: str) -> float:
    costs = {
        "cudf": 1.0,  # Base multiplier for GPU
        "polars": 0.8,  # Assuming optimized
    }
    return costs.get(backend, 2.0) * (hash(op) % 100) / 100.0  # Heuristic stub


def score_segment(segment: PlanSegment, sample_data: Any, backend: str | None = None) -> float:
    backend = backend or (
        "cudf" if capabilities().has_cudf else "polars" if capabilities().has_polars_gpu else "cpu"
    )
    total_cost = sum(estimate_compute_cost(op, backend) for op in segment.ops)
    if segment.spec:
        total_cost *= 1.5  # Penalty for complex specs
    try:
        # Score on a tiny sample to avoid mutating/processing full frames
        tiny = sample_data.head(1024) if hasattr(sample_data, "head") else sample_data
        start = perf_counter()
        reduce(lambda df, op: op(df), segment.ops, tiny)
        runtime = perf_counter() - start
        logger.debug(f"Scored segment: runtime {runtime}s")
        return total_cost + runtime
    except MemoryError:
        raise OOMRetry("Segment OOM; reduce size")
    except Exception as e:
        logger.warning(f"Segment score failed: {e}")
        return float("inf")


class HeuristicPlanner:
    metrics: dict[str, float] = {}  # Self-evolving cache

    def select_plan(self, segments: list[PlanSegment], sample_data: Any) -> list[PlanSegment]:
        if not segments:
            return []
        scores = {}
        for i, seg in enumerate(segments):
            score = score_segment(seg, sample_data)
            key = str(hash(seg))
            if key not in self.metrics or score < self.metrics.get(key, float("inf")):
                self.metrics[key] = score
                logger.debug(f"Evolved: Better plan segment {i} score {score}")
            scores[i] = score
        best_idx = min(scores, key=scores.get)
        return [segments[best_idx]]

    def optimize(self, segments: list[PlanSegment], sample_data: Any) -> list[PlanSegment]:
        if not segments:
            return []
        # Combinatoric: Generate variants dynamically
        variants = [segments]  # Stub: could permute or subset

        def _score_plan(plan):
            keys = [str(hash(s)) for s in plan]
            return sum(self.metrics.get(k, float("inf")) for k in keys)

        return min((self.select_plan(v, sample_data) for v in variants), key=_score_plan)
