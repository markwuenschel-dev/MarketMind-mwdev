"""PromotionSpec: validated, frozen spec for a promotion/rollout strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PromotionSpec:
    """Validated, immutable promotion spec derived from PromotionConfig."""

    name: str
    version: str
    spec_hash: str
    mode: Literal["shadow", "capped_blend", "full"]
    shadow_duration_days: int
    rollback_policy: Literal["auto", "manual", "none"]
    approval_required: bool
    blend_cap: float | None = None


__all__ = ["PromotionSpec"]
