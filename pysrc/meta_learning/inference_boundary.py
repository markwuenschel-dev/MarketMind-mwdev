"""MLN-05 frozen inference boundary — single source of truth for live vs training parameter roles.

Companion suite / Resolution Ledger normative lock: **live inference** uses a **frozen**
``theta_day_prime`` checkpoint only; **no gradients** and **no intraday mutation** of that object on
the live path. **theta_meta** is nightly meta-initialization; **theta_task_prime** is ephemeral,
training-only inner-loop state and must **not** be promoted to serving.

**What is live:** only a **gate-passed**, **promoted** ``theta_day_prime`` artifact reference.

**What stays offline-only:** gradient computation, ``theta_task_prime`` materialization, and
``theta_meta`` updates occur only on training / nightly paths — never on live execution.

**Closure:** MLN-05 is operationally enforceable where call sites use these helpers. A full program
closure still requires wiring every allocator/trainer entrypoint; this module is the canonical
contract those surfaces must import.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Literal

from pysrc.core.errors import DataPreconditionError

CONTRACT_VERSION: Final[str] = "mln05.inference_boundary.v2"


class ParameterRole(StrEnum):
    """Distinct meta-learning parameter objects (not interchangeable)."""

    THETA_META = "theta_meta"
    """Learned nightly meta-initialization; persisted; not the live serving tensor set."""

    THETA_TASK_PRIME = "theta_task_prime"
    """Ephemeral per-task adapted state; **training path only**; never a serving checkpoint."""

    THETA_DAY_PRIME = "theta_day_prime"
    """Nightly-adapted **live inference** state; frozen on the live path after promotion."""


class ExecutionPath(StrEnum):
    """Where code is running."""

    LIVE_INFERENCE = "live_inference"
    """Market-hours or live-serving feedforward path; frozen ``theta_day_prime`` only."""

    TRAINING = "training"
    """Offline / nightly learning; may compute gradients and materialize ``theta_task_prime``."""


class RolloutStage(StrEnum):
    """Promotion ladder stages that assume **frozen** promoted checkpoints (Architecture Vision)."""

    SHADOW = "shadow"
    CAPPED_BLEND = "capped_blend"
    FULL_PROMOTION = "full_promotion"


TrainingOutcome = Literal["success", "failed", "skipped"]


def rollout_stage_assumes_frozen_live_checkpoint(stage: RolloutStage | str) -> bool:
    """Shadow, capped blend, and full promotion all assume frozen live inference, not online adaptation."""
    _ = stage  # every documented stage is frozen-checkpoint-based
    return True


@dataclass(frozen=True, slots=True)
class ThetaDayPrimeCheckpointRef:
    """Reference to a promoted nightly live-eligible checkpoint (identity for audit / rollback)."""

    checkpoint_id: str
    """Stable id (e.g. CAS ``cas.v1:...`` or registry key)."""

    artifact_role: ParameterRole = ParameterRole.THETA_DAY_PRIME
    """Must be :attr:`ParameterRole.THETA_DAY_PRIME` for live serving."""


def validate_parameter_roles(*, checkpoint_role: ParameterRole, expected: ParameterRole) -> None:
    """Fail if a checkpoint is labeled with the wrong :class:`ParameterRole`."""
    if checkpoint_role != expected:
        raise DataPreconditionError(
            "checkpoint ParameterRole mismatch (MLN-05)",
            details={"expected": expected.value, "got": checkpoint_role.value},
        )


def validate_frozen_inference_request(
    *,
    execution_path: ExecutionPath,
    checkpoint_role: ParameterRole,
    allows_gradients: bool,
) -> None:
    """
    Live inference must use ``theta_day_prime`` only, frozen (no gradients).

    Training path may use other roles and gradients as appropriate.
    """
    if execution_path == ExecutionPath.LIVE_INFERENCE:
        if allows_gradients:
            raise DataPreconditionError(
                "live inference path must not compute or retain gradients (MLN-05)",
                details={"execution_path": execution_path.value},
            )
        if checkpoint_role != ParameterRole.THETA_DAY_PRIME:
            raise DataPreconditionError(
                "live inference must load frozen theta_day_prime only; "
                "theta_meta and theta_task_prime are not live-serving objects (MLN-05)",
                details={
                    "execution_path": execution_path.value,
                    "checkpoint_role": checkpoint_role.value,
                },
            )


def assert_no_live_gradients(*, execution_path: ExecutionPath, allows_gradients: bool) -> None:
    """Explicit guard for autograd-style flags on the live path."""
    if execution_path == ExecutionPath.LIVE_INFERENCE and allows_gradients:
        raise DataPreconditionError(
            "live path forbids gradient computation (MLN-05)",
            details={"execution_path": execution_path.value, "allows_gradients": allows_gradients},
        )


def ensure_training_only_task_prime(
    *, checkpoint_role: ParameterRole, execution_path: ExecutionPath
) -> None:
    """``theta_task_prime`` must never be used as the live serving checkpoint."""
    if (
        checkpoint_role == ParameterRole.THETA_TASK_PRIME
        and execution_path == ExecutionPath.LIVE_INFERENCE
    ):
        raise DataPreconditionError(
            "theta_task_prime is training-only and must not be used on the live inference path (MLN-05)",
            details={},
        )


def promote_theta_day_prime(
    *,
    current_live: ThetaDayPrimeCheckpointRef,
    candidate: ThetaDayPrimeCheckpointRef,
    gate_passed: bool,
    nightly_training_succeeded: bool,
) -> tuple[ThetaDayPrimeCheckpointRef, ThetaDayPrimeCheckpointRef | None]:
    """
    Nightly promotion: on **success** and **gate pass**, candidate becomes live and prior live becomes
    rollback target. Otherwise **current live is unchanged** (training failure must not contaminate live).

    Returns ``(new_live, rollback_target_or_none)``. When unchanged, returns ``(current_live, None)``.
    """
    validate_parameter_roles(
        checkpoint_role=candidate.artifact_role, expected=ParameterRole.THETA_DAY_PRIME
    )
    validate_parameter_roles(
        checkpoint_role=current_live.artifact_role, expected=ParameterRole.THETA_DAY_PRIME
    )
    if not gate_passed or not nightly_training_succeeded:
        return current_live, None
    return candidate, current_live


def rollback_theta_day_prime(
    *,
    current_live: ThetaDayPrimeCheckpointRef,
    rollback_target: ThetaDayPrimeCheckpointRef,
) -> ThetaDayPrimeCheckpointRef:
    """Operator rollback: restore previously promoted ``theta_day_prime``."""
    validate_parameter_roles(
        checkpoint_role=rollback_target.artifact_role, expected=ParameterRole.THETA_DAY_PRIME
    )
    validate_parameter_roles(
        checkpoint_role=current_live.artifact_role, expected=ParameterRole.THETA_DAY_PRIME
    )
    return rollback_target


def build_inference_boundary_audit_block(
    *,
    previous_live_theta_day_prime_ref: str,
    live_theta_day_prime_ref: str,
    rollback_theta_day_prime_ref: str,
    theta_meta_ref: str | None,
    training_outcome: TrainingOutcome,
    rollout_stage: RolloutStage | str | None = None,
) -> dict[str, Any]:
    """
    Governed artifact subdocument for ``meta_validity_report.json`` (optional MLN-05 audit).

    ``previous_live_theta_day_prime_ref`` is the live ``theta_day_prime`` identity **before** this run.
    :func:`validate_inference_boundary_audit_block` requires ``live_theta_day_prime_ref`` to match it when
    ``training_outcome`` is ``failed`` or ``skipped`` (no silent live mutation on failed nightly).
    """
    return {
        "schema_version": CONTRACT_VERSION,
        "previous_live_theta_day_prime_ref": str(previous_live_theta_day_prime_ref).strip(),
        "live_theta_day_prime_ref": str(live_theta_day_prime_ref).strip(),
        "rollback_theta_day_prime_ref": str(rollback_theta_day_prime_ref).strip(),
        "theta_meta_ref": None if theta_meta_ref is None else str(theta_meta_ref).strip(),
        "training_outcome": training_outcome,
        "ephemeral_theta_task_prime_training_only": True,
        "rollout_stage_assumes_frozen_live_checkpoint": True,
        "rollout_stage": None
        if rollout_stage is None
        else (
            rollout_stage.value if isinstance(rollout_stage, RolloutStage) else str(rollout_stage)
        ),
    }


def validate_inference_boundary_audit_block(block: Mapping[str, Any]) -> None:
    """Validate optional MLN-05 ``inference_boundary`` / audit block on governed reports."""
    ver = block.get("schema_version")
    if ver != CONTRACT_VERSION:
        raise DataPreconditionError(
            "inference_boundary.schema_version mismatch (MLN-05)",
            details={"schema_version": ver, "expected": CONTRACT_VERSION},
        )
    for key in (
        "previous_live_theta_day_prime_ref",
        "live_theta_day_prime_ref",
        "rollback_theta_day_prime_ref",
    ):
        v = block.get(key)
        if not isinstance(v, str) or not v.strip():
            raise DataPreconditionError(
                f"inference_boundary.{key} must be a non-empty string",
                details={key: v},
            )
    tm = block.get("theta_meta_ref")
    if tm is not None and (not isinstance(tm, str) or not str(tm).strip()):
        raise DataPreconditionError(
            "inference_boundary.theta_meta_ref must be null or a non-empty string",
            details={"theta_meta_ref": tm},
        )
    outcome = block.get("training_outcome")
    if outcome not in ("success", "failed", "skipped"):
        raise DataPreconditionError(
            "inference_boundary.training_outcome must be success|failed|skipped",
            details={"training_outcome": outcome},
        )
    prev_live = str(block["previous_live_theta_day_prime_ref"]).strip()
    cur_live = str(block["live_theta_day_prime_ref"]).strip()
    if outcome in ("failed", "skipped") and cur_live != prev_live:
        raise DataPreconditionError(
            "when training_outcome is failed or skipped, live_theta_day_prime_ref must equal "
            "previous_live_theta_day_prime_ref (failed/skipped nightly must not change live checkpoint)",
            details={
                "training_outcome": outcome,
                "live_theta_day_prime_ref": cur_live,
                "previous_live_theta_day_prime_ref": prev_live,
            },
        )
    if block.get("ephemeral_theta_task_prime_training_only") is not True:
        raise DataPreconditionError(
            "inference_boundary.ephemeral_theta_task_prime_training_only must be true (MLN-05)",
            details={},
        )
    if block.get("rollout_stage_assumes_frozen_live_checkpoint") is not True:
        raise DataPreconditionError(
            "inference_boundary.rollout_stage_assumes_frozen_live_checkpoint must be true (MLN-05)",
            details={},
        )


__all__ = [
    "CONTRACT_VERSION",
    "ExecutionPath",
    "ParameterRole",
    "RolloutStage",
    "ThetaDayPrimeCheckpointRef",
    "TrainingOutcome",
    "assert_no_live_gradients",
    "build_inference_boundary_audit_block",
    "ensure_training_only_task_prime",
    "promote_theta_day_prime",
    "rollback_theta_day_prime",
    "rollout_stage_assumes_frozen_live_checkpoint",
    "validate_frozen_inference_request",
    "validate_inference_boundary_audit_block",
    "validate_parameter_roles",
]
