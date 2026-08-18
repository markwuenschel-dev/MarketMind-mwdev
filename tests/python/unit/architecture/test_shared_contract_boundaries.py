"""Boundary tests for neutral cross-package contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.python.unit.architecture.import_boundary import imports_matching

ROOT = Path(__file__).resolve().parents[4]
PYSRC = ROOT / "pysrc"


@pytest.mark.determinism("d1")
def test_contracts_do_not_depend_on_product_implementations(deterministic_seed: int) -> None:
    _ = deterministic_seed

    violations = [
        f"{path.relative_to(ROOT)} -> {module}"
        for path, module in imports_matching(PYSRC / "contracts", prefix="pysrc.")
        if not module.startswith("pysrc.contracts")
    ]

    assert violations == [], violations


@pytest.mark.determinism("d1")
def test_pipeline_no_longer_owns_shared_meta_router_schemas(deterministic_seed: int) -> None:
    _ = deterministic_seed

    contracts_root = PYSRC / "pipeline" / "contracts"
    assert not (contracts_root / "meta_router.py").exists()
    assert not (contracts_root / "feature_channel.py").exists()
