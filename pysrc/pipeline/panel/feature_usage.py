"""Panel training feature usage diagnostics."""

from __future__ import annotations


def feature_usage_report(
    *,
    eligible_feature_count: int,
    used_feature_count: int,
    used_all_eligible_features: bool,
    excluded_features: list[dict[str, object]] | None = None,
    model_id: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "feature_usage.v1",
        "model_id": model_id,
        "eligible_feature_count": eligible_feature_count,
        "used_feature_count": used_feature_count,
        "used_all_eligible_features": used_all_eligible_features,
        "excluded_features": excluded_features or [],
    }


__all__ = ["feature_usage_report"]
