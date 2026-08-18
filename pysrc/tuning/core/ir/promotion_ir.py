"""PromotionIR: canonical IR for a promotion/rollout workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pysrc.tuning.core.ir.nodes import IRMetadata

__all__ = ["PromotionIR"]


@dataclass(frozen=True)
class PromotionIR:
    """Immutable promotion configuration derived from PromotionSpec."""

    job_id: str
    candidate_id: str
    mode: Literal["shadow", "capped_blend", "full"]
    shadow_duration_days: int
    rollback_policy: Literal["auto", "manual", "none"]
    approval_required: bool
    meta: IRMetadata
    blend_cap: float | None = None
