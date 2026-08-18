"""MLC-2 governed five-bucket curriculum sampler.

This module owns sampling policy over already-built :class:`MetaTask` objects:
five-bucket bootstrap batches, PER-like external-priority sampling, per-batch
crisis floor enforcement, sufficiency checks, and the single authoritative
training holdout exclusion surface for MLC-2.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.task import MetaTask
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER
from pysrc.ops.mm_logkit import get_logger

LOG = get_logger(__name__)

CurriculumPhase = Literal["bootstrap", "per_like"]
PriorityResolver = Callable[[str], float]


class CurriculumSufficiencyError(DataPreconditionError):
    """Raised when the trainable task pool cannot satisfy governed sampling gates."""


def _default_bucket_minimums() -> dict[str, int]:
    return {
        "bull": 50,
        "bear": 50,
        "sideways": 50,
        "high_vol": 30,
        "crisis": 20,
    }


def governed_v2_task_pool_bucket_minimums() -> dict[str, int]:
    """Core v2.0.0 §10.2-aligned pool floors (same defaults as :class:`CurriculumSamplerConfig`)."""
    return _default_bucket_minimums()


@dataclass(frozen=True, slots=True)
class HoldoutExclusionSurface:
    """Authoritative MLC-2 training exclusion surface."""

    task_ids: frozenset[str] = frozenset()

    def allows(self, task: MetaTask) -> bool:
        return task.task_id not in self.task_ids


@dataclass(frozen=True, slots=True)
class CurriculumSamplerConfig:
    batch_size: int = 32
    crisis_floor_fraction: float = 0.10
    bucket_minimums: Mapping[str, int] = field(default_factory=_default_bucket_minimums)
    priority_alpha: float = 0.6
    importance_beta: float = 0.4
    seed: int = 0


@dataclass(frozen=True, slots=True)
class CurriculumBatch:
    tasks: tuple[MetaTask, ...]
    phase: CurriculumPhase
    bucket_counts: Mapping[str, int]
    priority_alpha: float
    importance_beta: float
    importance_weights: tuple[float, ...] = ()


class CurriculumSampler:
    """Governed sampler over an existing MetaTask pool or TaskRegistry iterator."""

    def __init__(
        self,
        tasks: Iterable[MetaTask],
        *,
        config: CurriculumSamplerConfig,
        holdouts: HoldoutExclusionSurface | None = None,
    ) -> None:
        self._config = config
        self._holdouts = holdouts or HoldoutExclusionSurface()
        self._rng = np.random.default_rng(int(config.seed))
        self._tasks = tuple(task for task in tasks if self._holdouts.allows(task))
        self._by_bucket = self._bucket_tasks(self._tasks)
        self._validate_config()
        self._validate_bucket_minimums()

    @property
    def trainable_tasks(self) -> tuple[MetaTask, ...]:
        return self._tasks

    @property
    def bucket_governance_minimums(self) -> Mapping[str, int]:
        """v2.0 minimum-count surface used at sampler construction (trainer sufficiency gate)."""
        return self._config.bucket_minimums

    def sample_bootstrap(self) -> CurriculumBatch:
        selected = self._sample_with_bucket_floor(uniform=True, priority_resolver=None)
        return self._batch(selected, phase="bootstrap", weights=())

    def sample_per_like(self, *, priority_resolver: PriorityResolver) -> CurriculumBatch:
        selected = self._sample_with_bucket_floor(
            uniform=False, priority_resolver=priority_resolver
        )
        weights = self._importance_weights(selected, priority_resolver)
        return self._batch(selected, phase="per_like", weights=weights)

    def _validate_config(self) -> None:
        if self._config.batch_size < 1:
            raise CurriculumSufficiencyError(
                "batch_size must be positive", details={"batch_size": self._config.batch_size}
            )
        if not 0.0 <= self._config.crisis_floor_fraction <= 1.0:
            raise CurriculumSufficiencyError(
                "crisis_floor_fraction must be in [0, 1]",
                details={"crisis_floor_fraction": self._config.crisis_floor_fraction},
            )
        if self._config.priority_alpha < 0.0 or self._config.importance_beta < 0.0:
            raise CurriculumSufficiencyError(
                "priority alpha / beta must be non-negative",
                details={
                    "alpha": self._config.priority_alpha,
                    "beta": self._config.importance_beta,
                },
            )
        missing = set(REGIME_CLASS_ORDER) - set(self._config.bucket_minimums)
        if missing:
            raise CurriculumSufficiencyError(
                "bucket_minimums must define all five buckets",
                details={"missing": sorted(missing)},
            )

    @staticmethod
    def _bucket_tasks(tasks: Sequence[MetaTask]) -> dict[str, tuple[MetaTask, ...]]:
        buckets: dict[str, list[MetaTask]] = {bucket: [] for bucket in REGIME_CLASS_ORDER}
        for task in tasks:
            if task.regime_class not in buckets:
                raise CurriculumSufficiencyError(
                    "task carries unknown regime_class",
                    details={"task_id": task.task_id, "regime_class": task.regime_class},
                )
            buckets[task.regime_class].append(task)
        return {bucket: tuple(items) for bucket, items in buckets.items()}

    def _validate_bucket_minimums(self) -> None:
        counts = {bucket: len(self._by_bucket[bucket]) for bucket in REGIME_CLASS_ORDER}
        LOG.info("curriculum_bucket_counts", bucket_counts=counts)
        below = {
            bucket: {"actual": counts[bucket], "minimum": int(self._config.bucket_minimums[bucket])}
            for bucket in REGIME_CLASS_ORDER
            if counts[bucket] < int(self._config.bucket_minimums[bucket])
        }
        if below:
            raise CurriculumSufficiencyError(
                "bucket minimum not met", details={"bucket_counts": counts, "below": below}
            )

    def _crisis_floor_count(self) -> int:
        return int(ceil(float(self._config.batch_size) * float(self._config.crisis_floor_fraction)))

    def _sample_with_bucket_floor(
        self,
        *,
        uniform: bool,
        priority_resolver: PriorityResolver | None,
    ) -> tuple[MetaTask, ...]:
        selected: list[MetaTask] = []
        crisis_needed = self._crisis_floor_count()
        if crisis_needed > 0:
            selected.extend(
                self._sample_from_bucket(
                    "crisis",
                    crisis_needed,
                    uniform=uniform,
                    priority_resolver=priority_resolver,
                )
            )

        remaining = self._config.batch_size - len(selected)
        if remaining <= 0:
            return tuple(selected[: self._config.batch_size])

        non_empty = [bucket for bucket in REGIME_CLASS_ORDER if self._by_bucket[bucket]]
        for bucket in non_empty:
            if remaining <= 0:
                break
            if bucket == "crisis" and crisis_needed > 0:
                continue
            selected.extend(
                self._sample_from_bucket(
                    bucket,
                    1,
                    uniform=uniform,
                    priority_resolver=priority_resolver,
                )
            )
            remaining -= 1

        if remaining > 0:
            selected.extend(
                self._sample_from_pool(
                    self._tasks,
                    remaining,
                    uniform=uniform,
                    priority_resolver=priority_resolver,
                )
            )

        counts = Counter(task.regime_class for task in selected)
        if counts.get("crisis", 0) < crisis_needed:
            raise CurriculumSufficiencyError(
                "sampled batch violates crisis floor",
                details={"crisis_count": counts.get("crisis", 0), "required": crisis_needed},
            )
        return tuple(selected[: self._config.batch_size])

    def _sample_from_bucket(
        self,
        bucket: str,
        n: int,
        *,
        uniform: bool,
        priority_resolver: PriorityResolver | None,
    ) -> tuple[MetaTask, ...]:
        tasks = self._by_bucket[bucket]
        if not tasks and n > 0:
            raise CurriculumSufficiencyError(
                "cannot sample empty bucket", details={"bucket": bucket}
            )
        return self._sample_from_pool(
            tasks, n, uniform=uniform, priority_resolver=priority_resolver
        )

    def _sample_from_pool(
        self,
        tasks: Sequence[MetaTask],
        n: int,
        *,
        uniform: bool,
        priority_resolver: PriorityResolver | None,
    ) -> tuple[MetaTask, ...]:
        if n <= 0:
            return ()
        if not tasks:
            raise CurriculumSufficiencyError("cannot sample from empty task pool", details={})
        probs = None if uniform else self._priority_probs(tasks, priority_resolver)
        replace = len(tasks) < n
        idx = self._rng.choice(len(tasks), size=n, replace=replace, p=probs)
        return tuple(tasks[int(i)] for i in np.atleast_1d(idx))

    def _priority_probs(
        self,
        tasks: Sequence[MetaTask],
        priority_resolver: PriorityResolver | None,
    ) -> NDArray[np.float64]:
        if priority_resolver is None:
            raise CurriculumSufficiencyError(
                "PER-like sampling requires priority_resolver", details={}
            )
        raw = [max(float(priority_resolver(task.task_id)), 0.0) for task in tasks]
        if not all(math.isfinite(x) for x in raw):
            raise CurriculumSufficiencyError(
                "priority_resolver returned non-finite priority", details={}
            )
        weights = [math.pow(x + 1e-12, float(self._config.priority_alpha)) for x in raw]
        total = float(sum(weights))
        if total <= 0.0:
            weights = [1.0 for _ in weights]
            total = float(sum(weights))
        return np.asarray([x / total for x in weights], dtype=np.float64)

    def _importance_weights(
        self,
        tasks: Sequence[MetaTask],
        priority_resolver: PriorityResolver,
    ) -> tuple[float, ...]:
        probs = [float(x) for x in self._priority_probs(tasks, priority_resolver).tolist()]
        beta = float(self._config.importance_beta)
        weights = [math.pow(float(len(tasks)) * prob, -beta) for prob in probs]
        max_w = max(weights) if weights else 1.0
        normed = [weight / max_w for weight in weights] if max_w > 0.0 else weights
        return tuple(float(x) for x in normed)

    def _batch(
        self,
        tasks: Sequence[MetaTask],
        *,
        phase: CurriculumPhase,
        weights: tuple[float, ...],
    ) -> CurriculumBatch:
        counts = dict.fromkeys(REGIME_CLASS_ORDER, 0)
        counts.update(Counter(task.regime_class for task in tasks))
        LOG.info("curriculum_batch_sampled", phase=phase, bucket_counts=counts)
        return CurriculumBatch(
            tasks=tuple(tasks),
            phase=phase,
            bucket_counts=counts,
            priority_alpha=float(self._config.priority_alpha),
            importance_beta=float(self._config.importance_beta),
            importance_weights=weights,
        )


__all__ = [
    "CurriculumBatch",
    "CurriculumPhase",
    "CurriculumSampler",
    "CurriculumSamplerConfig",
    "CurriculumSufficiencyError",
    "HoldoutExclusionSurface",
    "PriorityResolver",
    "governed_v2_task_pool_bucket_minimums",
]
