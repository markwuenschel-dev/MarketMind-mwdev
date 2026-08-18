from __future__ import annotations

from pysrc.preprocessor.contracts.executor import ExecutionEvidence


def evidence_to_mapping(evidence: ExecutionEvidence) -> dict[str, object]:
    return {
        "events": list(evidence.events),
        "metrics": dict(evidence.metrics),
    }
