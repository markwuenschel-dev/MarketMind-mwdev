"""Determinism gate: verify that a result carries the required determinism tier."""

from __future__ import annotations

from pysrc.tuning.core.validation_ir.determinism_checks import (
    VALID_TIERS,
    DeterminismViolationError,
    assert_tier_not_downgraded,
)


def passes_determinism_gate(
    actual_tier: str,
    required_tier: str,
) -> tuple[bool, str]:
    """Return (passed, reason) without raising."""
    if actual_tier not in VALID_TIERS:
        return False, f"Unknown determinism tier: {actual_tier!r}"
    try:
        assert_tier_not_downgraded(
            required_tier,  # type: ignore[arg-type]
            actual_tier,  # type: ignore[arg-type]
        )
        return True, "determinism_ok"
    except DeterminismViolationError as exc:
        return False, str(exc)


__all__ = ["passes_determinism_gate"]
