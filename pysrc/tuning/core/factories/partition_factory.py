"""PartitionFactory: build PartitionPlan from a TuningJobSpec + date range."""

from __future__ import annotations

from datetime import datetime, timedelta

from pysrc.tuning.core.planning.partition_plan import PartitionPlan, TimePartition
from pysrc.tuning.core.specs.tuning_job_spec import TuningJobSpec
from pysrc.tuning.core.specs.validation_spec import ValidationSpec


def build_partition_plan(
    job_spec: TuningJobSpec,
    validation_spec: ValidationSpec,
    symbols: tuple[str, ...],
    start: datetime,
    end: datetime,
) -> PartitionPlan:
    """Build a PartitionPlan by dividing [start, end] into n_splits equal folds."""
    n = validation_spec.n_splits
    total_seconds = (end - start).total_seconds()
    fold_seconds = total_seconds / n
    partitions = tuple(
        TimePartition(
            name=f"fold_{i}",
            start=start + timedelta(seconds=i * fold_seconds),
            end=start + timedelta(seconds=(i + 1) * fold_seconds),
        )
        for i in range(n)
    )
    return PartitionPlan(
        job_id=job_spec.job_id,
        symbols=symbols,
        time_partitions=partitions,
        n_folds=n,
        n_total_tasks=n * len(symbols),
    )


__all__ = ["build_partition_plan"]
