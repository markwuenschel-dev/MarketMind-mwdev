from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class CanonicalFrameCIStatus(StrEnum):
    """Machine-readable CI certification state for CanonicalFrame support."""

    UNCERTIFIED = "uncertified"
    PYTHON_ONLY_D2 = "python_only_d2"
    CROSSLANG_D2_CERTIFIED = "crosslang_d2_certified"
    CROSSLANG_D3_CERTIFIED = "crosslang_d3_certified"


@dataclass(frozen=True)
class CanonicalFrameCIEvidence:
    """Named evidence inputs for CanonicalFrame certification state."""

    python_certified: bool
    golden_vectors_present: bool
    cross_language_certified: bool
    d3_primitives_promoted: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def status(self) -> CanonicalFrameCIStatus:
        if self.cross_language_certified:
            if not self.d3_primitives_promoted:
                return CanonicalFrameCIStatus.CROSSLANG_D2_CERTIFIED
            return CanonicalFrameCIStatus.CROSSLANG_D3_CERTIFIED
        if self.python_certified:
            return CanonicalFrameCIStatus.PYTHON_ONLY_D2
        return CanonicalFrameCIStatus.UNCERTIFIED

    def to_dict(self) -> dict[str, object]:
        return {
            "python_certified": self.python_certified,
            "golden_vectors_present": self.golden_vectors_present,
            "cross_language_certified": self.cross_language_certified,
            "d3_primitives_promoted": list(self.d3_primitives_promoted),
            "notes": list(self.notes),
            "status": self.status.value,
        }


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADR007_GOLDEN_ROOT = _REPO_ROOT / "tests" / "golden" / "adr007"
_ADR007_PARITY_REPORT_PATH = (
    _REPO_ROOT / "artifacts" / "phase_i_e" / "oi15" / "adr007_parity_report.json"
)


def _default_ci_evidence(*, note: str | None = None) -> CanonicalFrameCIEvidence:
    notes = [
        "Python replay covers the committed ADR-007 golden-vector suites at D2.",
        "Cross-language parity is only certified when the governed aggregate parity report artifact exists and resolves to crosslang_d2_certified.",
    ]
    if note is not None:
        notes.append(note)
    return CanonicalFrameCIEvidence(
        python_certified=True,
        golden_vectors_present=_ADR007_GOLDEN_ROOT.exists(),
        cross_language_certified=False,
        notes=tuple(notes),
    )


def load_canonical_frame_ci_evidence(
    path: Path | None = None,
) -> CanonicalFrameCIEvidence:
    report_path = path or _ADR007_PARITY_REPORT_PATH
    if not report_path.exists():
        return _default_ci_evidence(note=f"Parity report not found at {report_path.as_posix()}.")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _default_ci_evidence(
            note=f"Parity report at {report_path.as_posix()} could not be loaded: {exc}."
        )

    report_status = str(payload.get("status", CanonicalFrameCIStatus.UNCERTIFIED.value))
    certified_statuses = {
        CanonicalFrameCIStatus.CROSSLANG_D2_CERTIFIED.value,
        CanonicalFrameCIStatus.CROSSLANG_D3_CERTIFIED.value,
    }
    cross_language_certified = report_status in certified_statuses and bool(
        payload.get("cross_language_certified")
    )
    if not cross_language_certified:
        return _default_ci_evidence(
            note=(
                "Parity report at "
                f"{report_path.as_posix()} did not certify cross-language D2 "
                f"(status={report_status!r})."
            )
        )

    d3_primitives = tuple(str(item) for item in payload.get("d3_primitives_promoted", ()))
    notes = [
        "Python, C++20, and Java 21 replay the committed ADR-007 golden-vector suites in CI.",
        f"Cross-language parity is certified from governed artifact {report_path.as_posix()} at D2 on the committed fixture corpus; no primitive is promoted to D3 by this flag alone.",
    ]
    for note in payload.get("notes", ()):
        notes.append(str(note))
    return CanonicalFrameCIEvidence(
        python_certified=True,
        golden_vectors_present=_ADR007_GOLDEN_ROOT.exists(),
        cross_language_certified=True,
        d3_primitives_promoted=d3_primitives,
        notes=tuple(notes),
    )


CANONICAL_FRAME_CI_EVIDENCE = load_canonical_frame_ci_evidence()

CANONICAL_FRAME_CI_STATUS = CANONICAL_FRAME_CI_EVIDENCE.status
CANONICAL_FRAME_CI_STATUS_VALUE = CANONICAL_FRAME_CI_STATUS.value
CANONICAL_FRAME_CI_EVIDENCE_DICT = CANONICAL_FRAME_CI_EVIDENCE.to_dict()


__all__ = [
    "CanonicalFrameCIEvidence",
    "CanonicalFrameCIStatus",
    "CANONICAL_FRAME_CI_EVIDENCE",
    "CANONICAL_FRAME_CI_EVIDENCE_DICT",
    "CANONICAL_FRAME_CI_STATUS",
    "CANONICAL_FRAME_CI_STATUS_VALUE",
    "load_canonical_frame_ci_evidence",
]
