"""PromotionConfig: declarative spec for a rollout/promotion strategy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["PromotionConfig"]


class PromotionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str = "1.0.0"
    mode: Literal["shadow", "capped_blend", "full"] = "shadow"
    blend_cap: float | None = Field(default=None, ge=0.0, le=1.0)
    shadow_duration_days: int = Field(default=5, ge=1)
    rollback_policy: Literal["auto", "manual", "none"] = "auto"
    approval_required: bool = True
