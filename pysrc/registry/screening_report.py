"""Screening report builder: in-memory only; reason_family derived from REASON_CODE_TO_FAMILY.

Callers pass only reason_code when adding a stage; builder looks up reason_family
via REASON_CODE_TO_FAMILY. No disk write in this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pysrc.registry.screening_taxonomy import (
    REASON_CODE_TO_FAMILY,
    ReasonCode,
    ScreeningStage,
    ScreeningStatus,
)


@dataclass
class _StageRecord:
    stage: str
    status: str
    reason_code: str | None
    reason_family: str | None  # Derived from REASON_CODE_TO_FAMILY when reason_code set
    reason_detail: str | None
    metrics: dict[str, Any]
    duration_ms: int
    timestamp: str


@dataclass
class _CandidateRecord:
    candidate_run_id: str
    spec_hash: str
    signal_name: str
    slot_index: int | None
    stages: list[_StageRecord] = field(default_factory=list)
    final_status: str = ""
    final_stage: str = ""
    final_reason_code: str | None = None


class ScreeningReportBuilder:
    """Build screening_report.json payload. reason_family derived from reason_code only."""

    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        screening_run_id: str,
        pit_boundary: str,
        data_snapshot_hash: str,
        seed: int,
    ) -> None:
        self.screening_run_id = screening_run_id
        self.pit_boundary = pit_boundary
        self.data_snapshot_hash = data_snapshot_hash
        self.seed = seed
        self._candidates: list[_CandidateRecord] = []

    def add_candidate(
        self,
        spec_hash: str,
        signal_name: str,
        slot_index: int | None = None,
        evaluation_ordinal: int = 0,
    ) -> None:
        """Start a new candidate; candidate_run_id is deterministic from run_id + spec_hash + ordinal (full 64-char SHA-256)."""
        raw = f"{self.screening_run_id}:{spec_hash}:{evaluation_ordinal}"
        candidate_run_id = hashlib.sha256(raw.encode()).hexdigest()
        rec = _CandidateRecord(
            candidate_run_id=candidate_run_id,
            spec_hash=spec_hash,
            signal_name=signal_name,
            slot_index=slot_index,
        )
        self._candidates.append(rec)

    def add_stage(
        self,
        candidate_index: int,
        stage: ScreeningStage | str,
        status: ScreeningStatus | str,
        reason_code: ReasonCode | str | None = None,
        reason_detail: str | None = None,
        metrics: dict[str, Any] | None = None,
        duration_ms: int = 0,
        timestamp: str | None = None,
    ) -> None:
        """Add a stage for the candidate. reason_family is derived from reason_code via REASON_CODE_TO_FAMILY."""
        if candidate_index < 0 or candidate_index >= len(self._candidates):
            raise IndexError(f"candidate_index {candidate_index} out of range")
        stage_str = stage.value if isinstance(stage, ScreeningStage) else str(stage)
        status_str = status.value if isinstance(status, ScreeningStatus) else str(status)
        ts = timestamp or datetime.now(UTC).isoformat()
        reason_family: str | None = None
        code_str: str | None = None
        if reason_code is not None:
            code = (
                reason_code if isinstance(reason_code, ReasonCode) else ReasonCode(str(reason_code))
            )
            code_str = code.value
            reason_family = REASON_CODE_TO_FAMILY[code].value
        rec = _StageRecord(
            stage=stage_str,
            status=status_str,
            reason_code=code_str,
            reason_family=reason_family,
            reason_detail=reason_detail,
            metrics=metrics or {},
            duration_ms=duration_ms,
            timestamp=ts,
        )
        self._candidates[candidate_index].stages.append(rec)

    def set_final(
        self,
        candidate_index: int,
        final_status: str,
        final_stage: str,
        final_reason_code: str | None = None,
    ) -> None:
        """Set final_status, final_stage, final_reason_code for the candidate."""
        if candidate_index < 0 or candidate_index >= len(self._candidates):
            raise IndexError(f"candidate_index {candidate_index} out of range")
        self._candidates[candidate_index].final_status = final_status
        self._candidates[candidate_index].final_stage = final_stage
        self._candidates[candidate_index].final_reason_code = final_reason_code

    def serialize(self) -> dict[str, Any]:
        """Build the full report dict for JSON output."""
        intake_rejected = 0
        lane_0_rejected = 0
        lane_1_rejected = 0
        lane_2_rejected = 0
        promotion_rejected = 0
        promoted = 0
        errors = 0
        rejection_distribution: dict[str, int] = {}
        candidates_payload: list[dict[str, Any]] = []
        for c in self._candidates:
            if c.final_status == "PROMOTED":
                promoted += 1
            elif c.final_status == "ERROR":
                errors += 1
            else:
                stage_key = c.final_stage.replace(".", "_").lower()
                if "intake" in stage_key:
                    intake_rejected += 1
                elif "lane_0" in stage_key or "lane0" in stage_key:
                    lane_0_rejected += 1
                elif "lane_1" in stage_key or "lane1" in stage_key:
                    lane_1_rejected += 1
                elif "lane_2" in stage_key or "lane2" in stage_key:
                    lane_2_rejected += 1
                elif "promotion" in stage_key:
                    promotion_rejected += 1
                else:
                    intake_rejected += 1
                if c.final_reason_code:
                    rejection_distribution[c.final_reason_code] = (
                        rejection_distribution.get(c.final_reason_code, 0) + 1
                    )
            stages_payload = [
                {
                    "stage": s.stage,
                    "status": s.status,
                    "reason_code": s.reason_code,
                    "reason_family": s.reason_family,
                    "reason_detail": s.reason_detail,
                    "metrics": s.metrics,
                    "duration_ms": s.duration_ms,
                    "timestamp": s.timestamp,
                }
                for s in c.stages
            ]
            candidates_payload.append(
                {
                    "candidate_run_id": c.candidate_run_id,
                    "spec_hash": c.spec_hash,
                    "signal_name": c.signal_name,
                    "slot_index": c.slot_index,
                    "stages": stages_payload,
                    "final_status": c.final_status,
                    "final_stage": c.final_stage,
                    "final_reason_code": c.final_reason_code,
                }
            )
        rejection_distribution_list = [
            {"reason_code": k, "count": v} for k, v in sorted(rejection_distribution.items())
        ]
        summary: dict[str, Any] = {
            "total_candidates": len(self._candidates),
            "intake_rejected": intake_rejected,
            "lane_0_rejected": lane_0_rejected,
            "lane_1_rejected": lane_1_rejected,
            "lane_2_rejected": lane_2_rejected,
            "promotion_rejected": promotion_rejected,
            "promoted": promoted,
            "errors": errors,
            "rejection_distribution": rejection_distribution_list,
        }
        return {
            "schema_version": self.SCHEMA_VERSION,
            "screening_run_id": self.screening_run_id,
            "pit_boundary": self.pit_boundary,
            "data_snapshot_hash": self.data_snapshot_hash,
            "seed": self.seed,
            "candidates": candidates_payload,
            "summary": summary,
        }
