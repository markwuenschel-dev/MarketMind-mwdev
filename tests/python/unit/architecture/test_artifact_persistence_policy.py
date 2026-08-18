"""Artifact persistence policy tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pysrc.artifact_registry.persistence_policy import (
    STANDARD_DURABLE_PRODUCTS,
    finalize_run_artifacts,
    is_within_run_dir,
    teardown_ephemeral_run_dir,
)
from pysrc.artifact_registry.run_layout import allocate_run_dir, resolve_artifact_roots
from pysrc.contracts.meta_router import MetaRouterConfig
from pysrc.pipeline.meta_router_products import RUN_META, SMOKE_SUMMARY


@pytest.mark.determinism("d0")
def test_smoke_mode_persists_at_most_summary_and_run_meta(tmp_path: Path) -> None:
    roots = resolve_artifact_roots(tmp_path)
    run_dir = allocate_run_dir(lane="local_meta_router", smoke=True, roots=roots)
    (run_dir / "predictions").mkdir(exist_ok=True)
    (run_dir / "diagnostics").mkdir(exist_ok=True)
    pd.DataFrame({"x": [1]}).to_parquet(run_dir / "predictions" / "model_prediction_panel.parquet")
    (run_dir / "diagnostics" / "meta_router_training_panel.parquet").write_bytes(b"stub")

    config = MetaRouterConfig(smoke_test=True, keep_smoke_artifacts=False)
    report = finalize_run_artifacts(run_dir, config, summary={"ok": True})

    remaining = {p.name for p in run_dir.iterdir() if p.is_file()}
    assert remaining <= {RUN_META, SMOKE_SUMMARY}
    assert report.smoke_ephemeral is True
    assert SMOKE_SUMMARY in remaining


@pytest.mark.determinism("d0")
def test_standard_mode_keeps_durable_allowlist(tmp_path: Path) -> None:
    roots = resolve_artifact_roots(tmp_path)
    run_dir = allocate_run_dir(lane="local_meta_router", smoke=False, roots=roots)
    pred = run_dir / "predictions"
    diag = run_dir / "diagnostics"
    reports = run_dir / "reports"
    for directory in (pred, diag, reports):
        directory.mkdir(exist_ok=True)
    pd.DataFrame({"x": [1]}).to_parquet(pred / "model_prediction_panel.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(diag / "candidate_portfolio_output_panel.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(diag / "meta_router_decision_panel.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(pred / "meta_router_portfolio_output.parquet")
    (reports / "meta_router_evaluation_report.json").write_text("{}", encoding="utf-8")
    (run_dir / "panel_slice.parquet").write_bytes(b"intermediate")

    config = MetaRouterConfig(smoke_test=False, persist_intermediates=False)
    report = finalize_run_artifacts(run_dir, config)

    assert (pred / "model_prediction_panel.parquet").is_file()
    assert not (run_dir / "panel_slice.parquet").is_file()
    assert any("model_prediction_panel.parquet" in path for path in report.kept_paths)


@pytest.mark.determinism("d0")
def test_debug_mode_persists_intermediates(tmp_path: Path) -> None:
    roots = resolve_artifact_roots(tmp_path)
    run_dir = allocate_run_dir(lane="local_meta_router", smoke=False, roots=roots)
    intermediate = run_dir / "diagnostics" / "meta_router_training_panel.parquet"
    intermediate.parent.mkdir(parents=True, exist_ok=True)
    intermediate.write_bytes(b"stub")

    config = MetaRouterConfig(smoke_test=False, persist_intermediates=True)
    finalize_run_artifacts(run_dir, config)
    assert intermediate.is_file()


@pytest.mark.determinism("d0")
def test_no_output_outside_run_directory(tmp_path: Path) -> None:
    roots = resolve_artifact_roots(tmp_path)
    run_dir = allocate_run_dir(lane="local_meta_router", smoke=False, roots=roots)
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    assert is_within_run_dir(outside, run_dir) is False
    assert is_within_run_dir(run_dir / "run_meta.json", run_dir) is True


@pytest.mark.determinism("d0")
def test_teardown_ephemeral_run_dir(tmp_path: Path) -> None:
    roots = resolve_artifact_roots(tmp_path)
    run_dir = allocate_run_dir(lane="local_meta_router", smoke=True, roots=roots)
    teardown_ephemeral_run_dir(run_dir)
    assert not run_dir.exists()


@pytest.mark.determinism("d1")
def test_standard_allowlist_contains_required_products() -> None:
    required = {
        "run_meta.json",
        "model_prediction_panel.parquet",
        "candidate_portfolio_output_panel.parquet",
        "meta_router_decision_panel.parquet",
        "meta_router_portfolio_output.parquet",
        "meta_router_evaluation_report.json",
    }
    assert required.issubset(STANDARD_DURABLE_PRODUCTS)
