from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysrc.ops.hashing.canonical_frame import (
    CanonicalFrameCIStatus,
    load_canonical_frame_ci_evidence,
)

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d2")]


def test_canonical_frame_status_enum_exposes_cross_language_d2(
    deterministic_seed: int,
) -> None:
    assert deterministic_seed >= 0
    assert CanonicalFrameCIStatus.CROSSLANG_D2_CERTIFIED.value == "crosslang_d2_certified"


def test_load_canonical_frame_ci_evidence_falls_back_without_report(tmp_path: Path) -> None:
    evidence = load_canonical_frame_ci_evidence(tmp_path / "missing.json")

    assert evidence.cross_language_certified is False
    assert evidence.status == CanonicalFrameCIStatus.PYTHON_ONLY_D2


def test_load_canonical_frame_ci_evidence_reads_governed_report(tmp_path: Path) -> None:
    report_path = tmp_path / "adr007_parity_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": CanonicalFrameCIStatus.CROSSLANG_D2_CERTIFIED.value,
                "cross_language_certified": True,
                "notes": ["fixture parity passed"],
            }
        ),
        encoding="utf-8",
    )

    evidence = load_canonical_frame_ci_evidence(report_path)

    assert evidence.cross_language_certified is True
    assert evidence.status == CanonicalFrameCIStatus.CROSSLANG_D2_CERTIFIED
    assert any("fixture parity passed" in note for note in evidence.notes)
