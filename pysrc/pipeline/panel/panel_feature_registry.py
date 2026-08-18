"""Column discovery and eligibility classification for the P2-PANEL indicator universe."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import pandas as pd

# Conservative, auditable blocklist (substring match on lowered names).
LEAKAGE_BLOCKLIST_SUBSTRINGS: Final[tuple[str, ...]] = (
    "future",
    "forward",
    "target",
    "label",
    "oracle",
    "realized",
    "hindsight",
    "test_return",
    "net_utility",
    "gross_utility",
    "selected_after",
)

IDENTITY_COLUMN_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ticker",
        "symbol",
        "date",
        "datetime",
        "timestamp",
        "decision_timestamp",
        "instrument",
        "fold_id",
        "split",
        "surface_id",
        "bundle_id",
        "candidate_id",
        "child_policy_id",
        "run_id",
        "schema_version",
        "interval",
    }
)

SPLIT_OR_FOLD_COLUMN_NAMES: Final[frozenset[str]] = frozenset(
    {
        "fold_id",
        "split",
        "fold",
        "cv_fold",
        "walk_forward_fold",
    }
)

TARGET_SUFFIXES: Final[tuple[str, ...]] = (
    "_net_utility",
    "_forward_return",
    "_label",
)

POST_DECISION_SUFFIXES: Final[tuple[str, ...]] = (
    "_selected",
    "_allocation_weight",
    "_selected_weight",
)

DEFAULT_COVERAGE_THRESHOLD: Final[float] = 0.05


class FeatureExclusionReason(StrEnum):
    ELIGIBLE_FEATURE = "ELIGIBLE_FEATURE"
    IDENTITY_COLUMN = "IDENTITY_COLUMN"
    SPLIT_OR_FOLD_COLUMN = "SPLIT_OR_FOLD_COLUMN"
    TARGET_COLUMN = "TARGET_COLUMN"
    FUTURE_OR_LEAKAGE_COLUMN = "FUTURE_OR_LEAKAGE_COLUMN"
    ORACLE_COLUMN = "ORACLE_COLUMN"
    POST_DECISION_COLUMN = "POST_DECISION_COLUMN"
    LOW_COVERAGE_COLUMN = "LOW_COVERAGE_COLUMN"
    CONSTANT_COLUMN = "CONSTANT_COLUMN"
    NON_NUMERIC_UNSUPPORTED = "NON_NUMERIC_UNSUPPORTED"
    BLOCKED_REASON_UNKNOWN = "BLOCKED_REASON_UNKNOWN"


@dataclass(frozen=True, slots=True)
class FeatureColumnRecord:
    feature_name: str
    source: str
    feature_family: str
    interval: str
    coverage: float
    dtype: str
    exclusion_reason: FeatureExclusionReason
    used_by_default: bool

    @property
    def is_eligible(self) -> bool:
        return self.exclusion_reason == FeatureExclusionReason.ELIGIBLE_FEATURE


def is_eligible_feature(record: FeatureColumnRecord) -> bool:
    return record.is_eligible


def _normalized(name: str) -> str:
    return str(name).strip().lower()


def _blocklist_category(lowered: str) -> FeatureExclusionReason | None:
    if "hindsight" in lowered or lowered.startswith("row_oracle"):
        return FeatureExclusionReason.ORACLE_COLUMN
    if "oracle" in lowered:
        return FeatureExclusionReason.ORACLE_COLUMN
    if any(token in lowered for token in ("future", "forward", "test_return", "selected_after")):
        return FeatureExclusionReason.FUTURE_OR_LEAKAGE_COLUMN
    if any(
        token in lowered
        for token in ("target", "label", "realized", "net_utility", "gross_utility")
    ):
        if "realized_volatility" in lowered or lowered.endswith("_realized_vol"):
            return None
        return FeatureExclusionReason.TARGET_COLUMN
    return None


def _target_like(name: str, lowered: str) -> bool:
    if lowered.startswith("target__"):
        return True
    if lowered.startswith("utility_child_"):
        return True
    if lowered.startswith("child_") and any(lowered.endswith(sfx) for sfx in TARGET_SUFFIXES):
        return True
    if lowered in {"hindsight_best_child", "hindsight_best_child_utility"}:
        return True
    return bool(re.fullmatch(r"forward_return.*", lowered))


def _post_decision_like(lowered: str) -> bool:
    if lowered.startswith("child_") and lowered.endswith("_score"):
        return False
    if lowered.startswith("child_") and any(
        lowered.endswith(sfx) for sfx in POST_DECISION_SUFFIXES
    ):
        return True
    return lowered.endswith("_selected") and not lowered.endswith("_selected_rate")


def classify_column(
    name: str,
    series: pd.Series,
    *,
    source: str,
    feature_family: str = "discovered",
    interval: str = "daily",
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> FeatureColumnRecord:
    """Classify one discovered column; default posture is include unless invalid."""

    lowered = _normalized(name)
    dtype = str(series.dtype)
    numeric = pd.api.types.is_numeric_dtype(series)
    non_null = series.notna()
    coverage = float(non_null.mean()) if len(series) else 0.0

    if lowered in IDENTITY_COLUMN_NAMES:
        reason = FeatureExclusionReason.IDENTITY_COLUMN
    elif lowered in SPLIT_OR_FOLD_COLUMN_NAMES:
        reason = FeatureExclusionReason.SPLIT_OR_FOLD_COLUMN
    elif lowered in {"hindsight_best_child", "hindsight_best_child_utility"}:
        reason = FeatureExclusionReason.ORACLE_COLUMN
    elif _target_like(name, lowered):
        reason = FeatureExclusionReason.TARGET_COLUMN
    elif _post_decision_like(lowered):
        reason = FeatureExclusionReason.POST_DECISION_COLUMN
    else:
        blocked = _blocklist_category(lowered)
        if blocked is not None:
            reason = blocked
        elif not numeric:
            if feature_family == "discovered":
                feature_family = "categorical_criterion"
            reason = FeatureExclusionReason.NON_NUMERIC_UNSUPPORTED
        elif coverage < coverage_threshold:
            reason = FeatureExclusionReason.LOW_COVERAGE_COLUMN
        elif numeric and coverage >= coverage_threshold:
            sample = pd.to_numeric(series, errors="coerce").dropna()
            if len(sample) > 1 and float(sample.std(ddof=0)) == 0.0:
                reason = FeatureExclusionReason.CONSTANT_COLUMN
            else:
                reason = FeatureExclusionReason.ELIGIBLE_FEATURE
        else:
            reason = FeatureExclusionReason.BLOCKED_REASON_UNKNOWN

    return FeatureColumnRecord(
        feature_name=str(name),
        source=source,
        feature_family=feature_family,
        interval=interval,
        coverage=coverage,
        dtype=dtype,
        exclusion_reason=reason,
        used_by_default=reason == FeatureExclusionReason.ELIGIBLE_FEATURE,
    )
