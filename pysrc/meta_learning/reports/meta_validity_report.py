"""Schema-owned builder for ``meta_validity_report``-shaped dicts (MLC-3).

Every trainer run (PASS or FAIL) must emit a structurally complete JSON-safe
payload. Missing semantics are represented as ``null``, not omitted keys.

``inner_loop_gain`` is a nested object (not a bare float) with nullable slots for
Harvey t and per-regime breakdowns until later gate evidence exists.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

REQUIRED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "run_id",
        "overall_result",
        "reporting_gate",
        "inner_loop_gain",
        "shuffle_test_p_value",
        "proxy_IC_pearson_r",
        "crisis_holdout_ic",
        "forgetting_ic_degradation_pct",
        "task_pool_counts",
        "confidence_calibration",
        "fail_reasons",
        "theta_day_prime_promoted",
        "timestamp_utc",
    }
)

INNER_LOOP_GAIN_KEYS: frozenset[str] = frozenset({"mean_query_ic", "harvey_t", "by_regime_class"})

TASK_POOL_COUNTS_REQUIRED: frozenset[str] = frozenset(
    {"batch_size", "crisis_count", "crisis_required", "bucket_counts"}
)
TASK_POOL_COUNTS_OPTIONAL: frozenset[str] = frozenset({"phase"})

CONFIDENCE_CALIBRATION_KEYS: frozenset[str] = frozenset({"ece", "note"})


class MetaValidityReportBuildError(ValueError):
    """Raised when a report dict would be structurally invalid."""


def _json_float(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def validate_inner_loop_gain_block(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise MetaValidityReportBuildError("inner_loop_gain must be a mapping")
    keys = frozenset(obj.keys())
    if keys != INNER_LOOP_GAIN_KEYS:
        raise MetaValidityReportBuildError(
            f"inner_loop_gain keys must be exactly {sorted(INNER_LOOP_GAIN_KEYS)}, got {sorted(keys)}"
        )
    mean_ic = obj["mean_query_ic"]
    if mean_ic is not None and not _json_float(mean_ic):
        raise MetaValidityReportBuildError(
            "inner_loop_gain.mean_query_ic must be float, int, or null"
        )
    ht = obj["harvey_t"]
    if ht is not None and not _json_float(ht):
        raise MetaValidityReportBuildError("inner_loop_gain.harvey_t must be float, int, or null")
    br = obj["by_regime_class"]
    if br is None:
        return {
            "mean_query_ic": float(mean_ic) if mean_ic is not None else None,
            "harvey_t": float(ht) if ht is not None else None,
            "by_regime_class": None,
        }
    if not isinstance(br, Mapping):
        raise MetaValidityReportBuildError(
            "inner_loop_gain.by_regime_class must be a mapping or null"
        )
    out_reg: dict[str, float] = {}
    for k, v in br.items():
        if not isinstance(k, str):
            raise MetaValidityReportBuildError("by_regime_class keys must be strings")
        if not _json_float(v):
            raise MetaValidityReportBuildError(f"by_regime_class[{k!r}] must be numeric")
        out_reg[k] = float(v)
    return {
        "mean_query_ic": float(mean_ic) if mean_ic is not None else None,
        "harvey_t": float(ht) if ht is not None else None,
        "by_regime_class": out_reg,
    }


def validate_task_pool_counts_block(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise MetaValidityReportBuildError("task_pool_counts must be a mapping or null")
    keys = frozenset(obj.keys())
    extra = keys - TASK_POOL_COUNTS_REQUIRED - TASK_POOL_COUNTS_OPTIONAL
    if extra:
        raise MetaValidityReportBuildError(f"task_pool_counts unknown keys: {sorted(extra)}")
    missing = TASK_POOL_COUNTS_REQUIRED - keys
    if missing:
        raise MetaValidityReportBuildError(f"task_pool_counts missing keys: {sorted(missing)}")
    bs = obj["batch_size"]
    cc = obj["crisis_count"]
    cr = obj["crisis_required"]
    if not isinstance(bs, int) or bs < 0:
        raise MetaValidityReportBuildError("task_pool_counts.batch_size must be a non-negative int")
    if not isinstance(cc, int) or cc < 0:
        raise MetaValidityReportBuildError(
            "task_pool_counts.crisis_count must be a non-negative int"
        )
    if not isinstance(cr, int) or cr < 0:
        raise MetaValidityReportBuildError(
            "task_pool_counts.crisis_required must be a non-negative int"
        )
    bc = obj["bucket_counts"]
    if not isinstance(bc, Mapping):
        raise MetaValidityReportBuildError("task_pool_counts.bucket_counts must be a mapping")
    bc_out: dict[str, int] = {}
    for k, v in bc.items():
        if not isinstance(k, str):
            raise MetaValidityReportBuildError("bucket_counts keys must be strings")
        if not isinstance(v, int) or v < 0:
            raise MetaValidityReportBuildError(f"bucket_counts[{k!r}] must be a non-negative int")
        bc_out[k] = int(v)
    out: dict[str, Any] = {
        "batch_size": int(bs),
        "crisis_count": int(cc),
        "crisis_required": int(cr),
        "bucket_counts": bc_out,
    }
    if "phase" in obj:
        ph = obj["phase"]
        if ph is not None and not isinstance(ph, str):
            raise MetaValidityReportBuildError("task_pool_counts.phase must be a string or null")
        out["phase"] = ph
    return out


def validate_confidence_calibration_block(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if not isinstance(obj, Mapping):
        raise MetaValidityReportBuildError("confidence_calibration must be a mapping or null")
    keys = frozenset(obj.keys())
    if keys != CONFIDENCE_CALIBRATION_KEYS:
        raise MetaValidityReportBuildError(
            f"confidence_calibration keys must be exactly {sorted(CONFIDENCE_CALIBRATION_KEYS)}, got {sorted(keys)}"
        )
    ece = obj["ece"]
    if ece is not None and not _json_float(ece):
        raise MetaValidityReportBuildError("confidence_calibration.ece must be numeric or null")
    note = obj["note"]
    if note is not None and not isinstance(note, str):
        raise MetaValidityReportBuildError("confidence_calibration.note must be a string or null")
    return {
        "ece": float(ece) if ece is not None else None,
        "note": note,
    }


def validate_meta_validity_report_keys(payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_KEYS - frozenset(payload.keys())
    if missing:
        raise MetaValidityReportBuildError(f"meta_validity_report missing keys: {sorted(missing)}")
    extra = frozenset(payload.keys()) - REQUIRED_TOP_LEVEL_KEYS
    if extra:
        raise MetaValidityReportBuildError(f"meta_validity_report unknown keys: {sorted(extra)}")
    validate_inner_loop_gain_block(payload["inner_loop_gain"])
    tpc = payload["task_pool_counts"]
    if tpc is not None:
        validate_task_pool_counts_block(tpc)
    cc = payload["confidence_calibration"]
    if cc is not None:
        validate_confidence_calibration_block(cc)


def scaffold_inner_loop_gain(
    *,
    mean_query_ic: float | None = None,
    harvey_t: float | None = None,
    by_regime_class: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Canonical nested ``inner_loop_gain`` object for MLC-3 reports."""
    return validate_inner_loop_gain_block(
        {"mean_query_ic": mean_query_ic, "harvey_t": harvey_t, "by_regime_class": by_regime_class}
    )


def scaffold_task_pool_counts(
    *,
    batch_size: int,
    crisis_count: int,
    crisis_required: int,
    bucket_counts: Mapping[str, int],
    phase: str | None = None,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "batch_size": batch_size,
        "crisis_count": crisis_count,
        "crisis_required": crisis_required,
        "bucket_counts": dict(bucket_counts),
    }
    if phase is not None:
        raw["phase"] = phase
    return validate_task_pool_counts_block(raw)


def scaffold_confidence_calibration(
    *,
    ece: float | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    v = validate_confidence_calibration_block({"ece": ece, "note": note})
    if v is None:
        raise MetaValidityReportBuildError("scaffold_confidence_calibration internal error")
    return v


def build_meta_validity_report(
    *,
    schema_version: str,
    run_id: str,
    overall_result: str,
    reporting_gate: str,
    inner_loop_gain: Mapping[str, Any] | None,
    shuffle_test_p_value: float | None,
    proxy_IC_pearson_r: float | None,
    crisis_holdout_ic: float | None,
    forgetting_ic_degradation_pct: float | None,
    task_pool_counts: Mapping[str, Any] | None,
    confidence_calibration: Mapping[str, Any] | None,
    fail_reasons: list[str],
    theta_day_prime_promoted: bool,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable report dict; raises on structural invalidity."""
    if not isinstance(run_id, str) or not run_id:
        raise MetaValidityReportBuildError("run_id must be a non-empty string")
    if overall_result not in {"PASS", "FAIL"}:
        raise MetaValidityReportBuildError(
            f"overall_result must be PASS or FAIL, got {overall_result!r}"
        )
    ts = timestamp_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    ilg = (
        validate_inner_loop_gain_block(inner_loop_gain)
        if inner_loop_gain is not None
        else scaffold_inner_loop_gain()
    )
    tpc_out: dict[str, Any] | None
    if task_pool_counts is None:
        tpc_out = None
    else:
        tpc_out = validate_task_pool_counts_block(task_pool_counts)
    cc_raw = validate_confidence_calibration_block(confidence_calibration)
    out: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "overall_result": overall_result,
        "reporting_gate": reporting_gate,
        "inner_loop_gain": ilg,
        "shuffle_test_p_value": shuffle_test_p_value,
        "proxy_IC_pearson_r": proxy_IC_pearson_r,
        "crisis_holdout_ic": crisis_holdout_ic,
        "forgetting_ic_degradation_pct": forgetting_ic_degradation_pct,
        "task_pool_counts": tpc_out if tpc_out else None,
        "confidence_calibration": cc_raw,
        "fail_reasons": list(fail_reasons),
        "theta_day_prime_promoted": bool(theta_day_prime_promoted),
        "timestamp_utc": ts,
    }
    validate_meta_validity_report_keys(out)
    return out


@dataclass
class MetaValidityReportBuilder:
    """Mutable accumulator that emits the same governed dict as :func:`build_meta_validity_report`."""

    schema_version: str = "v1"
    run_id: str = ""
    overall_result: str = "FAIL"
    reporting_gate: str = "MLC3_SCAFFOLD"
    inner_loop_gain: dict[str, Any] = field(default_factory=lambda: scaffold_inner_loop_gain())
    shuffle_test_p_value: float | None = None
    proxy_IC_pearson_r: float | None = None
    crisis_holdout_ic: float | None = None
    forgetting_ic_degradation_pct: float | None = None
    task_pool_counts: dict[str, Any] | None = None
    confidence_calibration: dict[str, Any] | None = None
    fail_reasons: list[str] = field(default_factory=list)
    theta_day_prime_promoted: bool = False
    timestamp_utc: str | None = None

    def build(self) -> dict[str, Any]:
        return build_meta_validity_report(
            schema_version=self.schema_version,
            run_id=self.run_id,
            overall_result=self.overall_result,
            reporting_gate=self.reporting_gate,
            inner_loop_gain=self.inner_loop_gain,
            shuffle_test_p_value=self.shuffle_test_p_value,
            proxy_IC_pearson_r=self.proxy_IC_pearson_r,
            crisis_holdout_ic=self.crisis_holdout_ic,
            forgetting_ic_degradation_pct=self.forgetting_ic_degradation_pct,
            task_pool_counts=self.task_pool_counts,
            confidence_calibration=self.confidence_calibration,
            fail_reasons=self.fail_reasons,
            theta_day_prime_promoted=self.theta_day_prime_promoted,
            timestamp_utc=self.timestamp_utc,
        )


__all__ = [
    "CONFIDENCE_CALIBRATION_KEYS",
    "INNER_LOOP_GAIN_KEYS",
    "MetaValidityReportBuildError",
    "MetaValidityReportBuilder",
    "REQUIRED_TOP_LEVEL_KEYS",
    "TASK_POOL_COUNTS_OPTIONAL",
    "TASK_POOL_COUNTS_REQUIRED",
    "build_meta_validity_report",
    "scaffold_confidence_calibration",
    "scaffold_inner_loop_gain",
    "scaffold_task_pool_counts",
    "validate_confidence_calibration_block",
    "validate_inner_loop_gain_block",
    "validate_meta_validity_report_keys",
    "validate_task_pool_counts_block",
]
