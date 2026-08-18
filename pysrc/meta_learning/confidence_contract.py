"""MLN-03 canonical ``confidence_scalar`` semantics (single source of truth).

Default semantics (Phase II)
---------------------------
``confidence_scalar`` is **post-sizing attenuation only**:

    live_position = base_position * confidence_scalar

with ``confidence_scalar ∈ [0, 1]``. It may **reduce** exposure (including full abstention at 0).
It **must not** increase exposure above the base position from ``allocation_weights`` / ``SizingFn``.

**No hidden leverage:** values above 1 are invalid. Any change to default semantics requires an ADR
and threshold governance (MLN-07).

Routing boundary (non-default)
------------------------------
Uncertainty-aware **routing / rejection** is a **Phase II-0 pilot hypothesis**, not default
architecture. **Fail action:** remain **attenuation-only** plus the simpler risk path. Promotion
requires **earned** reject-set evidence (negative EV after costs) and incumbent positive EV there;
if attenuation already captures the economic effect, **do not promote routing**.

Use :func:`is_routing_enabled` — default is **False** unless both explicit pilot opt-in and earned
evidence flags are true.

Artifact surface
-----------------
Phase II ``meta_validity_report.json`` (governed MLN-06 path) carries a ``confidence_calibration``
block (schema ``mln03.confidence_calibration.v1``) with ECE / calibration / recalibration /
routing-pilot separation. See :func:`insufficient_confidence_calibration_block` and
:func:`validate_confidence_calibration_artifact_block`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

from pysrc.core.errors import DataPreconditionError

CONFIDENCE_SCALAR_MIN: Final[float] = 0.0
CONFIDENCE_SCALAR_MAX: Final[float] = 1.0

CONFIDENCE_CALIBRATION_SCHEMA_VERSION: Final[str] = "mln03.confidence_calibration.v1"

REPORTING_GATE_PASS: Final[str] = "PASS"
REPORTING_GATE_FAIL: Final[str] = "FAIL"
REPORTING_GATE_INSUFFICIENT: Final[str] = "INSUFFICIENT"

_VALID_REPORTING_GATES: Final[frozenset[str]] = frozenset(
    {REPORTING_GATE_PASS, REPORTING_GATE_FAIL, REPORTING_GATE_INSUFFICIENT}
)
_VALID_COMPONENT_STATUS: Final[frozenset[str]] = frozenset(
    {REPORTING_GATE_PASS, REPORTING_GATE_FAIL, REPORTING_GATE_INSUFFICIENT, "unavailable"}
)


def validate_confidence_scalar(value: float | int) -> float:
    """Require a finite scalar in ``[0, 1]``."""
    x = float(value)
    if not math.isfinite(x):
        raise DataPreconditionError(
            "confidence_scalar must be finite",
            details={"value": value},
        )
    if x < CONFIDENCE_SCALAR_MIN or x > CONFIDENCE_SCALAR_MAX:
        raise DataPreconditionError(
            "confidence_scalar must be in [0, 1] (post-sizing attenuation only; MLN-03)",
            details={"value": value},
        )
    return x


def apply_confidence_attenuation(
    *, base_position: float | int, confidence_scalar: float | int
) -> float:
    """
    ``live_position = base_position * confidence_scalar`` with MLN-03 validation.

    Never increases absolute exposure relative to ``base_position`` for ``confidence_scalar`` in [0, 1].
    """
    c = validate_confidence_scalar(confidence_scalar)
    b = float(base_position)
    if not math.isfinite(b):
        raise DataPreconditionError(
            "base_position must be finite", details={"base_position": base_position}
        )
    out = b * c
    if not math.isfinite(out):
        raise DataPreconditionError(
            "attenuation result non-finite", details={"base_position": b, "c": c}
        )
    if abs(out) > abs(b) + 1e-9:
        raise DataPreconditionError(
            "confidence attenuation must not increase absolute exposure (MLN-03)",
            details={"base_position": b, "confidence_scalar": c, "result": out},
        )
    return out


def is_routing_enabled(
    *,
    pilot_explicit_opt_in: bool = False,
    reject_set_negative_evidence_after_costs: bool = False,
) -> bool:
    """
    Uncertainty-aware routing is **off** unless both conditions hold.

    This is intentionally strict so call sites cannot treat routing as default.
    """
    return bool(pilot_explicit_opt_in and reject_set_negative_evidence_after_costs)


def insufficient_confidence_calibration_block(*, reason: str) -> dict[str, Any]:
    """Governed placeholder when calibration / ECE were not measured on this run."""
    return {
        "schema_version": CONFIDENCE_CALIBRATION_SCHEMA_VERSION,
        "default_confidence_semantics": "post_sizing_attenuation",
        "formula": "live_position = base_position * confidence_scalar",
        "reporting_gate": REPORTING_GATE_INSUFFICIENT,
        "calibration_method_status": {"status": REPORTING_GATE_INSUFFICIENT, "detail": reason},
        "ece": {"status": REPORTING_GATE_INSUFFICIENT, "value": None, "threshold_note": None},
        "reliability_diagnostics": {"status": REPORTING_GATE_INSUFFICIENT, "reference": None},
        "recalibration": {
            "status": REPORTING_GATE_INSUFFICIENT,
            "required": None,
            "performed": None,
        },
        "routing_pilot": {
            "enabled": False,
            "explicit_opt_in_required": True,
            "reject_set_negative_ev_after_costs_required": True,
            "note": (
                "Uncertainty-aware routing is a Phase II-0 pilot only; default remains attenuation-only "
                "(Core §2.5 / MLN-03)."
            ),
        },
    }


def synthetic_confidence_calibration_pass_block(
    *,
    ece_value: float,
    calibration_method: str = "synthetic_test_platt",
    reliability_reference: str = "test://reliability_curve",
) -> dict[str, Any]:
    """Test / harness helper — not a claim of empirical calibration."""
    ev = float(ece_value)
    if not math.isfinite(ev) or ev < 0.0:
        raise DataPreconditionError(
            "ece_value must be finite and non-negative", details={"ece_value": ece_value}
        )
    return {
        "schema_version": CONFIDENCE_CALIBRATION_SCHEMA_VERSION,
        "default_confidence_semantics": "post_sizing_attenuation",
        "formula": "live_position = base_position * confidence_scalar",
        "reporting_gate": REPORTING_GATE_PASS,
        "calibration_method_status": {"status": REPORTING_GATE_PASS, "method": calibration_method},
        "ece": {"status": REPORTING_GATE_PASS, "value": ev, "threshold_note": "synthetic"},
        "reliability_diagnostics": {
            "status": REPORTING_GATE_PASS,
            "reference": reliability_reference,
        },
        "recalibration": {"status": REPORTING_GATE_PASS, "required": False, "performed": False},
        "routing_pilot": {
            "enabled": False,
            "explicit_opt_in_required": True,
            "reject_set_negative_ev_after_costs_required": True,
            "note": "Routing remains disabled in synthetic pass fixture.",
        },
    }


def validate_confidence_calibration_artifact_block(obj: Mapping[str, Any]) -> None:
    """Fail closed if governed confidence / calibration block is malformed."""
    if not isinstance(obj, Mapping):
        raise DataPreconditionError("confidence_calibration must be an object", details={})
    if str(obj.get("schema_version")) != CONFIDENCE_CALIBRATION_SCHEMA_VERSION:
        raise DataPreconditionError(
            "confidence_calibration.schema_version mismatch",
            details={"got": obj.get("schema_version")},
        )
    gate = str(obj.get("reporting_gate", ""))
    if gate not in _VALID_REPORTING_GATES:
        raise DataPreconditionError(
            "confidence_calibration.reporting_gate invalid",
            details={"reporting_gate": obj.get("reporting_gate")},
        )
    for key in (
        "default_confidence_semantics",
        "formula",
        "calibration_method_status",
        "ece",
        "reliability_diagnostics",
        "recalibration",
        "routing_pilot",
    ):
        if key not in obj:
            raise DataPreconditionError(
                f"confidence_calibration missing {key!r}",
                details={"missing": key},
            )
    sem = str(obj.get("default_confidence_semantics", ""))
    if sem != "post_sizing_attenuation":
        raise DataPreconditionError(
            "confidence_calibration.default_confidence_semantics must be post_sizing_attenuation (MLN-03)",
            details={"value": sem},
        )
    cal = obj["calibration_method_status"]
    ece = obj["ece"]
    rel = obj["reliability_diagnostics"]
    if not isinstance(cal, Mapping) or str(cal.get("status", "")) not in _VALID_COMPONENT_STATUS:
        raise DataPreconditionError("invalid calibration_method_status", details={"obj": cal})
    if not isinstance(ece, Mapping) or str(ece.get("status", "")) not in _VALID_COMPONENT_STATUS:
        raise DataPreconditionError("invalid ece block", details={"obj": ece})
    if not isinstance(rel, Mapping) or str(rel.get("status", "")) not in _VALID_COMPONENT_STATUS:
        raise DataPreconditionError("invalid reliability_diagnostics", details={"obj": rel})
    rp = obj["routing_pilot"]
    if not isinstance(rp, Mapping) or "enabled" not in rp:
        raise DataPreconditionError(
            "routing_pilot must be an object with enabled: bool", details={}
        )
    if not isinstance(rp.get("enabled"), bool):
        raise DataPreconditionError(
            "routing_pilot.enabled must be bool", details={"enabled": rp.get("enabled")}
        )
    if rp.get("enabled") is True:
        raise DataPreconditionError(
            "routing_pilot.enabled must be False on governed Phase II meta_validity path (MLN-03); "
            "uncertainty routing is non-default and requires ADR-governed emission policy before True",
            details={},
        )


__all__ = [
    "CONFIDENCE_CALIBRATION_SCHEMA_VERSION",
    "CONFIDENCE_SCALAR_MAX",
    "CONFIDENCE_SCALAR_MIN",
    "REPORTING_GATE_FAIL",
    "REPORTING_GATE_INSUFFICIENT",
    "REPORTING_GATE_PASS",
    "apply_confidence_attenuation",
    "insufficient_confidence_calibration_block",
    "is_routing_enabled",
    "synthetic_confidence_calibration_pass_block",
    "validate_confidence_calibration_artifact_block",
    "validate_confidence_scalar",
]
