"""TaskBuilder: assembles TaskIR objects from search candidates and fold boundaries."""

from __future__ import annotations

from pysrc.tuning.core.factories.task_factory import build_task_ir
from pysrc.tuning.core.ir.task_ir import FoldBoundary, TaskIR

__all__ = ["TaskBuilder"]


class TaskBuilder:
    """Stateless builder that converts (candidate, fold, context) -> TaskIR."""

    def build(
        self,
        task_id: str,
        job_id: str,
        candidate_id: str,
        fold: FoldBoundary,
        params: dict[str, object],
        feature_hash: str,
        spec_hash: str,
        determinism_tier: str = "d1",
    ) -> TaskIR:
        return build_task_ir(
            task_id=task_id,
            job_id=job_id,
            candidate_id=candidate_id,
            fold=fold,
            params=params,
            feature_hash=feature_hash,
            spec_hash=spec_hash,
            determinism_tier=determinism_tier,
        )
