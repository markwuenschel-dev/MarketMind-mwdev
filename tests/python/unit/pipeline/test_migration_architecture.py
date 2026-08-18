"""Architecture guardrails after P2 source-module migration."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
PYSRC = ROOT / "pysrc"
TESTS = ROOT / "tests" / "python"

OBSOLETE_PACKAGES = (
    "phase2_broad_reset",
    "phase2_panel_model",
    "phase2_local_meta_router",
    "phase2_deep_v1",
    "phase2_router_v2",
)


@pytest.mark.determinism("d1")
def test_no_active_phase2_source_packages() -> None:
    for name in OBSOLETE_PACKAGES:
        assert not (PYSRC / "meta" / name).exists(), name


@pytest.mark.determinism("d1")
def test_no_active_imports_of_removed_phase2_packages() -> None:
    tokens = tuple(f"pysrc.meta.{pkg}" for pkg in OBSOLETE_PACKAGES)
    hits: list[str] = []
    for path in (*PYSRC.rglob("*.py"), *TESTS.rglob("*.py")):
        if "archive" in path.parts or "quarantine" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token in text:
                hits.append(f"{path.relative_to(ROOT)}: {token}")
    assert hits == [], hits


@pytest.mark.determinism("d1")
def test_research_directory_has_no_generated_artifact_patterns() -> None:
    research = ROOT / "research"
    forbidden_suffixes = {".parquet", ".pkl", ".pt", ".onnx", ".log"}
    for path in research.rglob("*"):
        if path.is_file() and path.suffix in forbidden_suffixes:
            pytest.fail(f"generated artifact under research/: {path.relative_to(ROOT)}")
