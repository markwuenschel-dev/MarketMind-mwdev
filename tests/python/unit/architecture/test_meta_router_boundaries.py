"""Import-boundary tests for MetaRouter production scaffolding."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.python.unit.architecture.import_boundary import imports_matching

ROOT = Path(__file__).resolve().parents[4]
PYSRC = ROOT / "pysrc"


def _assert_no_imports(package_rel: str, forbidden_prefix: str) -> None:
    package_root = PYSRC / package_rel.replace("pysrc.", "").replace("/", "\\").replace(
        "pysrc\\", ""
    )
    if not package_root.is_dir():
        package_root = PYSRC / Path(package_rel.replace("pysrc/", ""))
    violations = imports_matching(package_root, prefix=forbidden_prefix)
    formatted = [f"{path.relative_to(ROOT)} -> {module}" for path, module in violations]
    assert formatted == [], formatted


@pytest.mark.determinism("d1")
def test_models_does_not_import_meta_router_pipeline() -> None:
    _assert_no_imports("models", "pysrc.pipeline.meta_router")


@pytest.mark.determinism("d1")
def test_meta_does_not_import_meta_router_pipeline() -> None:
    _assert_no_imports("meta", "pysrc.pipeline.meta_router")


@pytest.mark.determinism("d1")
def test_meta_learning_does_not_import_meta_router_pipeline() -> None:
    _assert_no_imports("meta_learning", "pysrc.pipeline.meta_router")


@pytest.mark.determinism("d1")
def test_panel_does_not_import_meta_router_pipeline() -> None:
    _assert_no_imports("pipeline/panel", "pysrc.pipeline.meta_router")


@pytest.mark.determinism("d1")
def test_candidate_portfolios_does_not_import_meta_router_pipeline() -> None:
    _assert_no_imports("pipeline/candidate_portfolios", "pysrc.pipeline.meta_router")


@pytest.mark.determinism("d1")
def test_meta_router_does_not_import_raw_market_data_sources() -> None:
    package_root = PYSRC / "pipeline" / "meta_router"
    forbidden = (
        "pysrc.pipeline.stages.market_data",
        "pysrc.data",
        "data_loader",
    )
    violations: list[str] = []
    for path, module in imports_matching(package_root, prefix="pysrc."):
        if any(token in module for token in forbidden):
            violations.append(f"{path.relative_to(ROOT)} -> {module}")
    assert violations == [], violations


@pytest.mark.determinism("d1")
def test_meta_router_does_not_import_tuning_meta_learning() -> None:
    _assert_no_imports("pipeline/meta_router", "pysrc.tuning.core.meta_learning")


@pytest.mark.determinism("d1")
def test_meta_router_does_not_import_legacy_router_lane() -> None:
    package_root = PYSRC / "pipeline" / "meta_router"
    violations = imports_matching(package_root, prefix="pysrc.pipeline.router")
    formatted = [f"{path.relative_to(ROOT)} -> {module}" for path, module in violations]
    assert formatted == [], formatted


@pytest.mark.determinism("d1")
def test_backtesting_does_not_import_meta_router_implementation() -> None:
    package_root = PYSRC / "backtesting"
    violations = imports_matching(package_root, prefix="pysrc.pipeline.meta_router")
    formatted = [f"{path.relative_to(ROOT)} -> {module}" for path, module in violations]
    assert formatted == [], formatted


@pytest.mark.determinism("d1")
def test_meta_router_does_not_import_pairwise_advantage() -> None:
    package_root = PYSRC / "pipeline" / "meta_router"
    violations = imports_matching(package_root, prefix="pysrc.meta.pairwise_advantage")
    formatted = [f"{path.relative_to(ROOT)} -> {module}" for path, module in violations]
    assert formatted == [], formatted


@pytest.mark.determinism("d1")
def test_retired_router_package_is_not_live() -> None:
    assert not (PYSRC / "pipeline" / "router").exists()
