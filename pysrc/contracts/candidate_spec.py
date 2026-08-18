"""Neutral model-candidate description shared by model-training consumers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandidateSpec(BaseModel):
    """Declared model input and decision semantics, independent of a router lane."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    router_target: str = Field(min_length=1)
    decision_rule: str = Field(min_length=1)
    input_surface: str = Field(min_length=1)
    feature_allowlist: list[str] | str
    forbidden_features: list[str] = Field(default_factory=list)
    comparators: list[str] = Field(default_factory=list)
    split_policy: str = Field(default="time_aware_holdout_30pct", min_length=1)
    status: str = Field(default="generated", min_length=1)
    reason: str | None = None
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    created_by: str = Field(default="model_candidate.v1", min_length=1)
    supervision_path: str | None = None
    feature_policy: str = Field(default="full_indicator_universe_v1", min_length=1)

    @model_validator(mode="before")
    @classmethod
    def canonicalize_legacy_default_override(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if values.get("decision_rule") == "default_to_best_child_override":
            values["decision_rule"] = "train_default_override"
            candidate_id = values.get("candidate_id")
            if isinstance(candidate_id, str):
                values["candidate_id"] = candidate_id.replace(
                    "__default_to_best_child_override", "__train_default_override"
                )
        return values


__all__ = ["CandidateSpec"]
