"""Shared threshold identity and register-shaped record helpers (decoupled from RG09 lane)."""

from __future__ import annotations

from typing import Any, Final, Literal

# Historical IDs retained for register compatibility and gate tests.
THR_RG09_V03: Final[str] = "THR-RG09-V03"  # functional_harvey_t_threshold
THR_RG09_V17: Final[str] = "THR-RG09-V17"  # episode regime-class purity floor

VALIDATE_NOTE: Final[str] = "⚑ VALIDATE — see ThresholdGovernanceRegister.md"


def threshold_value_record[ThresholdValue: (int, float, str)](
    value: ThresholdValue,
    threshold_id: str,
    *,
    state: Literal["PROVISIONAL", "VALIDATED", "REJECTED", "DEPRECATED"] = "VALIDATED",
) -> dict[str, Any]:
    """Register-shaped threshold object for JSON config and artifact cross-walks."""
    return {
        "threshold_id": threshold_id,
        "value": value,
        "state": state,
        "validate_note": VALIDATE_NOTE,
    }


def provisional_threshold_record[ThresholdValue: (int, float, str)](
    value: ThresholdValue,
    threshold_id: str,
    *,
    state: Literal["PROVISIONAL", "VALIDATED", "REJECTED", "DEPRECATED"] = "PROVISIONAL",
) -> dict[str, Any]:
    return threshold_value_record(value, threshold_id, state=state)


__all__ = [
    "THR_RG09_V03",
    "THR_RG09_V17",
    "VALIDATE_NOTE",
    "provisional_threshold_record",
    "threshold_value_record",
]
