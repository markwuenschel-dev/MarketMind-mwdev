"""Import boundaries for the generic meta-learning package."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.python.unit.architecture.import_boundary import imports_matching

ROOT = Path(__file__).resolve().parents[4]
META_LEARNING = ROOT / "pysrc" / "meta_learning"


@pytest.mark.determinism("d1")
def test_meta_learning_does_not_import_meta(deterministic_seed: int) -> None:
    _ = deterministic_seed

    violations = [
        f"{path.relative_to(ROOT)} -> {module}"
        for path, module in imports_matching(META_LEARNING, prefix="pysrc.meta")
        if module == "pysrc.meta" or module.startswith("pysrc.meta.")
    ]

    assert violations == [], violations
