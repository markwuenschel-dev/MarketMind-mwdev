"""Phase II-0B canonical ownership guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
pytestmark = pytest.mark.usefixtures("deterministic_seed")
_GOVERNANCE_FILES = {
    ROOT / "pysrc/meta/phase2_artifact_contract.py",
    ROOT / "pysrc/meta/seed_policy.py",
    ROOT / "pysrc/meta/threshold_governance.py",
}


def _require_governance_files_or_skip() -> None:
    missing = sorted(
        str(path.relative_to(ROOT)) for path in _GOVERNANCE_FILES if not path.is_file()
    )
    if missing:
        pytest.skip(
            "research-first P2 lane does not require governed Phase 2 files: " + ", ".join(missing)
        )


@pytest.mark.determinism("d1")
def test_phase2_governed_surfaces_have_single_canonical_root() -> None:
    forbidden = {
        ROOT / "py/meta/phase2_artifact_contract.py",
        ROOT / "py/meta/seed_policy.py",
        ROOT / "py/meta/threshold_governance.py",
    }
    for path in forbidden:
        assert not path.exists(), path

    _require_governance_files_or_skip()
    for path in _GOVERNANCE_FILES:
        assert path.is_file(), path


@pytest.mark.determinism("d1")
def test_no_new_alias_path_for_governed_artifact_emitter() -> None:
    _require_governance_files_or_skip()
    matches = [
        path
        for path in (ROOT / "pysrc").rglob("*.py")
        if "def emit_phase2_artifacts" in path.read_text(encoding="utf-8")
    ]
    assert matches == [ROOT / "pysrc/meta/phase2_artifact_contract.py"]
