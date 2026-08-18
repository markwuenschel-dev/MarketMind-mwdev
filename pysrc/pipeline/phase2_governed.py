"""Research-first stub for pipeline orchestration evidence (ADR-003).

Full governed II-0B emission lived under archive/research_lanes/ii0c/. Active
research runs do not require Resolution Ledger or promotion evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHASE2_GOVERNED_EVIDENCE_SUBDIR = "phase2_ii0b_governed_non_promotable"
PHASE2_GOVERNED_SUMMARY_FILENAME = "phase2_ii0b_governed_non_promotable.json"
PHASE2_GOVERNED_SUMMARY_SCHEMA = "research_first.orchestration.v1"


def emit_governed_phase2_orchestration_evidence(
    *,
    bundle_dir: Path,
    strategy_id: str,
    run_id: str,
    strategy_context: Any | None = None,
    source_prices: Any | None = None,
    features: Any | None = None,
    signals: Any | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _ = (
        bundle_dir,
        strategy_id,
        run_id,
        strategy_context,
        source_prices,
        features,
        signals,
        run_metadata,
    )
    return {
        "schema_version": PHASE2_GOVERNED_SUMMARY_SCHEMA,
        "non_promotable": True,
        "governed_artifact_contract": False,
        "current_governed_evidence": False,
        "research_first_lane": True,
        "note": (
            "Governed II-0B orchestration evidence is skipped on the research-first "
            "path (ADR-003). Restore archive/research_lanes/ii0c/pipeline/phase2_governed "
            "only if promotion gates return."
        ),
        "phase2_ml_evidence_shell": {
            "status": "SKIPPED_RESEARCH_FIRST",
            "reason_codes": ["ADR003_RESEARCH_FIRST"],
        },
    }


__all__ = [
    "PHASE2_GOVERNED_EVIDENCE_SUBDIR",
    "PHASE2_GOVERNED_SUMMARY_FILENAME",
    "PHASE2_GOVERNED_SUMMARY_SCHEMA",
    "emit_governed_phase2_orchestration_evidence",
]
