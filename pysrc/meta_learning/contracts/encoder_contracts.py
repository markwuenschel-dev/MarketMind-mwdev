"""Context encoder contract stubs for Phase II meta-learning surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER

# F-5 DEFERRED: D-tier declarations for encoder artifact outputs are out of
# scope for this stub. See OI-23 / GATE-I-F-05.

TASK_EPISODE_ENCODER_FEATURE_SCHEMA_VERSION: Final[str] = "task_episode_c_t.v1"


@runtime_checkable
class RegimeLabelRecord(Protocol):
    effective_at: datetime
    decision_ts: datetime
    regime_class: str
    boundary_flag: str
    change_probability: float
    run_length_mode: int
    run_length_expectation: float
    transition_probability: float
    posterior_entropy: float
    trend_score_raw: float
    vol_score_raw: float


TASK_EPISODE_BOUNDARY_FLAG_ORDER: Final[tuple[str, ...]] = (
    "cold_start",
    "change_point",
    "transition",
    "stable",
)
TASK_EPISODE_ENCODER_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "regime_class_bull",
    "regime_class_bear",
    "regime_class_sideways",
    "regime_class_high_vol",
    "regime_class_crisis",
    "boundary_flag_cold_start",
    "boundary_flag_change_point",
    "boundary_flag_transition",
    "boundary_flag_stable",
    "change_probability",
    "run_length_mode",
    "run_length_expectation",
    "transition_probability",
    "posterior_entropy",
    "trend_score_raw",
    "vol_score_raw",
)


@dataclass(frozen=True)
class EncoderInputContract:
    """Context encoder input boundary."""

    regime_features: np.ndarray[Any, Any]
    pit_boundary: datetime
    signal_set_version: int
    schema_version: str = "v1"


@dataclass(frozen=True)
class EncoderOutputContract:
    """Context encoder output boundary."""

    # VALIDATE NOTE: The current architectural placeholder for embedding
    # dimension is 64. AQ-01 is open. This stub does not freeze D as a code
    # constant. Any component consuming regime_embedding must tolerate
    # dimension as a runtime property, not a compile-time constant.
    regime_embedding: np.ndarray[Any, np.dtype[np.float32]]
    schema_version: str = "v1"


def _normalize_contract_datetime(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise DataPreconditionError(
            f"{field_name} must be datetime", details={"type": type(value).__name__}
        )
    if value.tzinfo is None:
        raise DataPreconditionError(
            f"{field_name} must be timezone-aware",
            details={field_name: value.isoformat()},
        )
    return value.astimezone(UTC)


def build_task_episode_encoder_input(
    *,
    regime_label: RegimeLabelRecord,
    pit_boundary: datetime,
    signal_set_version: int,
) -> EncoderInputContract:
    """Lower a governed task episode to the single authoritative ``c_t`` contract.

    OI-60 freezes ``c_t`` as the PIT-boundary regime-label context for the task:
    a fixed-width, float32 vector of support-side regime class, BOCPD boundary,
    and scalar diagnostic fields from the :class:`RegimeLabelRecord` available
    no later than ``pit_boundary``.  This function deliberately performs no
    support/query row reduction; query rows and post-PIT data are outside the
    contract.
    """

    if not isinstance(regime_label, RegimeLabelRecord):
        raise DataPreconditionError(
            "regime_label must be RegimeLabelRecord for task episode encoder lowering",
            details={"type": type(regime_label).__name__},
        )
    pit_utc = _normalize_contract_datetime(pit_boundary, field_name="pit_boundary")
    effective = _normalize_contract_datetime(
        regime_label.effective_at, field_name="regime_label.effective_at"
    )
    decision = _normalize_contract_datetime(
        regime_label.decision_ts, field_name="regime_label.decision_ts"
    )
    if effective > pit_utc or decision > pit_utc:
        raise DataPreconditionError(
            "task episode c_t lowering requires regime label available at pit_boundary",
            details={
                "effective_at": effective.isoformat(),
                "decision_ts": decision.isoformat(),
                "pit_boundary": pit_utc.isoformat(),
            },
        )
    if regime_label.regime_class not in REGIME_CLASS_ORDER:
        raise DataPreconditionError(
            "regime_class is outside the governed encoder lowering vocabulary",
            details={"regime_class": regime_label.regime_class},
        )
    if regime_label.boundary_flag not in TASK_EPISODE_BOUNDARY_FLAG_ORDER:
        raise DataPreconditionError(
            "boundary_flag is outside the governed encoder lowering vocabulary",
            details={"boundary_flag": regime_label.boundary_flag},
        )
    if not isinstance(signal_set_version, int) or isinstance(signal_set_version, bool):
        raise DataPreconditionError(
            "signal_set_version must be int for task episode encoder lowering",
            details={"signal_set_version": signal_set_version},
        )

    class_features = [
        1.0 if regime_label.regime_class == bucket else 0.0 for bucket in REGIME_CLASS_ORDER
    ]
    boundary_features = [
        1.0 if regime_label.boundary_flag == boundary_flag else 0.0
        for boundary_flag in TASK_EPISODE_BOUNDARY_FLAG_ORDER
    ]
    scalar_features = [
        float(regime_label.change_probability),
        float(regime_label.run_length_mode),
        float(regime_label.run_length_expectation),
        float(regime_label.transition_probability),
        float(regime_label.posterior_entropy),
        float(regime_label.trend_score_raw),
        float(regime_label.vol_score_raw),
    ]
    features = np.asarray(class_features + boundary_features + scalar_features, dtype=np.float32)
    if features.shape != (len(TASK_EPISODE_ENCODER_FEATURE_NAMES),):
        raise DataPreconditionError(
            "task episode c_t feature schema width mismatch",
            details={
                "expected": len(TASK_EPISODE_ENCODER_FEATURE_NAMES),
                "actual": int(features.size),
                "feature_schema_version": TASK_EPISODE_ENCODER_FEATURE_SCHEMA_VERSION,
            },
        )
    if not bool(np.isfinite(features).all()):
        raise DataPreconditionError(
            "task episode c_t features must be finite",
            details={"feature_schema_version": TASK_EPISODE_ENCODER_FEATURE_SCHEMA_VERSION},
        )
    return EncoderInputContract(
        regime_features=np.ascontiguousarray(features, dtype=np.float32),
        pit_boundary=pit_utc,
        signal_set_version=signal_set_version,
        schema_version="v1",
    )


# RELIABILITY CROSSWALK: EncoderOutputContract.regime_embedding is upstream
# input to the Signal Reliability Layer
# (OI-41 / signal_reliability_schema_v0_1_1.md). The reliability layer
# consumes regime_embedding z after the context encoder runs. Compatibility is
# semantic: any change to embedding dtype or shape must be evaluated against
# the reliability layer's downstream consumption path before committing. No
# separate regime_embedding_dim schema field is defined by the reliability spec.
class ContextEncoderProtocol(Protocol):
    """Protocol for encoder implementations used by the Phase II policy path.

    Encoder freeze semantics align with MLN-05 (frozen live inference); see
    :mod:`pysrc.meta_learning.inference_boundary`.
    """

    def encode(self, input: EncoderInputContract) -> EncoderOutputContract: ...

    def is_frozen(self) -> bool:
        """Must return True during inner-loop adaptation. Frozen encoder invariant."""
        ...


__all__ = [
    "ContextEncoderProtocol",
    "EncoderInputContract",
    "EncoderOutputContract",
    "TASK_EPISODE_BOUNDARY_FLAG_ORDER",
    "TASK_EPISODE_ENCODER_FEATURE_NAMES",
    "TASK_EPISODE_ENCODER_FEATURE_SCHEMA_VERSION",
    "build_task_episode_encoder_input",
]
