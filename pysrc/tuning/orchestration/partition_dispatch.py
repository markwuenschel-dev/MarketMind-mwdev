"""PartitionDispatcher: fan out a PartitionPlan into individual task submissions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pysrc.tuning.core.planning.partition_plan import PartitionPlan

TaskSubmitFn = Callable[[str, dict[str, Any]], None]


class PartitionDispatcher:
    """Iterates a PartitionPlan and submits one task per (symbol, fold) pair."""

    def dispatch(
        self,
        plan: PartitionPlan,
        submit_fn: TaskSubmitFn,
        candidate_id: str,
    ) -> int:
        """Submit tasks; return total number of tasks submitted."""
        submitted = 0
        for symbol in plan.symbols:
            for partition in plan.time_partitions:
                submit_fn(
                    f"{plan.job_id}|{candidate_id}|{symbol}|{partition.name}",
                    {
                        "job_id": plan.job_id,
                        "candidate_id": candidate_id,
                        "symbol": symbol,
                        "partition": partition.name,
                    },
                )
                submitted += 1
        return submitted


__all__ = ["TaskSubmitFn", "PartitionDispatcher"]
