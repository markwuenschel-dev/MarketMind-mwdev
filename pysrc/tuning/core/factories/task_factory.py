"""task_factory: constructs TaskIR objects from primitive inputs."""

from __future__ import annotations

import time

from pysrc.tuning.core.ir.nodes import HParam, IRMetadata
from pysrc.tuning.core.ir.task_ir import FoldBoundary, TaskIR

__all__ = ["build_task_ir"]


def build_task_ir(
    task_id: str,
    job_id: str,
    candidate_id: str,
    fold: FoldBoundary,
    params: dict[str, object],
    feature_hash: str,
    spec_hash: str,
    determinism_tier: str = "d1",
) -> TaskIR:
    """Construct a TaskIR from primitive arguments.

    Converts the params dict into a sorted tuple of HParam nodes and stamps
    IRMetadata with a monotonic nanosecond timestamp.
    """
    hparams = tuple(HParam(name=k, value=v) for k, v in sorted(params.items()))
    meta = IRMetadata(
        spec_hash=spec_hash,
        created_at_ns=time.monotonic_ns(),
        determinism_tier=determinism_tier,
    )
    return TaskIR(
        task_id=task_id,
        job_id=job_id,
        candidate_id=candidate_id,
        fold=fold,
        params=hparams,
        feature_hash=feature_hash,
        meta=meta,
    )
