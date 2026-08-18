"""Gate 7: assemble Appendix C promotion bundle from panel model-matrix run artifacts."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.pipeline.p2_config_loader import PortfolioSpec

_DEFAULT_EVALUATION_COST_BPS = 10.0
_PURGE_WINDOW_DAYS = 2


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_gate6_pass(run_dir: Path) -> dict[str, Any]:
    gate6_path = run_dir / "reports" / "gate6_promotion_report.json"
    if not gate6_path.is_file():
        raise FileNotFoundError(f"Missing Gate 6 report: {gate6_path}")
    report = _load_json(gate6_path)
    if not report.get("gate_pass"):
        raise ValueError("Gate 6 report gate_pass is false; cannot assemble promotion bundle")
    return report


def build_panel_promotion_plan_config(
    run_dir: Path,
    *,
    model_id: str,
    strategy_id: str,
    portfolio_spec: PortfolioSpec,
    panel_path: Path,
    evaluation_cost_bps: float = _DEFAULT_EVALUATION_COST_BPS,
) -> dict[str, Any]:
    return {
        "strategy": strategy_id,
        "promotion_model_id": model_id,
        "source_run_id": run_dir.name,
        "source_product_id": "model_prediction_panel",
        "panel_path": str(panel_path),
        "production_chain": "predictions → threshold intents → positions → simulate",
        "portfolio": {
            "top_k": portfolio_spec.top_k,
            "single_name_cap": portfolio_spec.single_name_cap,
            "cost_bps": evaluation_cost_bps,
            "capacity_constraints": portfolio_spec.capacity_constraints,
        },
        "pdr": "PDR-001",
        "gate": "gate7_promotion_bundle",
    }


def build_splits_manifest_from_run(run_dir: Path) -> dict[str, Any]:
    """Derive walk-forward splits from prediction panel fold date ranges."""

    pred_path = run_dir / "predictions" / "model_prediction_panel.parquet"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing predictions: {pred_path}")
    frame = pd.read_parquet(pred_path, columns=["date", "fold_id"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        raise ValueError("Cannot build splits manifest: prediction panel has no dates")

    global_start = frame["date"].min()
    splits: list[dict[str, Any]] = []
    for fold_idx, fold_id in enumerate(sorted(frame["fold_id"].astype(str).unique().tolist())):
        fold_dates = frame.loc[frame["fold_id"].astype(str) == fold_id, "date"]
        test_start = fold_dates.min().to_pydatetime().replace(tzinfo=UTC)
        test_end = fold_dates.max().to_pydatetime().replace(tzinfo=UTC)
        train_end = test_start - timedelta(days=_PURGE_WINDOW_DAYS)
        train_start = global_start.to_pydatetime().replace(tzinfo=UTC)
        splits.append(
            {
                "fold_id": fold_idx,
                "train_start": train_start.isoformat().replace("+00:00", "Z"),
                "train_end": train_end.isoformat().replace("+00:00", "Z"),
                "test_start": test_start.isoformat().replace("+00:00", "Z"),
                "test_end": test_end.isoformat().replace("+00:00", "Z"),
                "train_count": 0,
                "test_count": int(fold_dates.nunique()),
                "purged_count": 0,
                "embargoed_count": 0,
                "non_contiguous_train": False,
                "fold_label": fold_id,
            }
        )

    return {
        "schema_version": "1.0.0",
        "split_method": "walk_forward",
        "purge_window": _PURGE_WINDOW_DAYS,
        "embargo_window": 0,
        "splits": splits,
        "n_splits": len(splits),
        "timestamp_column": "date",
    }


def _synthesize_execution_assumptions(*, cost_bps: float) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "execution_model_id": "fill.identity",
        "cost_model_id": "panel_promotion_cost",
        "commission_bps": float(cost_bps),
        "slippage_bps": 0.0,
        "fill_ratio": 1.0,
        "seed": 42,
    }


def _copy_sidecar(src: Path, dest: Path) -> None:
    if src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def assemble_promotion_bundle(
    run_dir: Path,
    *,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    model_id: str = "xgboost",
    strategy_id: str | None = None,
    out_dir: Path | None = None,
    evaluation_cost_bps: float = _DEFAULT_EVALUATION_COST_BPS,
) -> Path:
    """Write Appendix C promotion bundle for the promotion model."""

    run_dir = Path(run_dir)
    gate6 = _require_gate6_pass(run_dir)
    strategy_id = strategy_id or f"prediction_threshold_{model_id}"
    bundle_dir = Path(out_dir or run_dir / "bundles" / f"promotion_{model_id}")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    stat_report = dict(gate6["stat_validity_report"])
    config = build_panel_promotion_plan_config(
        run_dir,
        model_id=model_id,
        strategy_id=strategy_id,
        portfolio_spec=portfolio_spec,
        panel_path=panel_path,
        evaluation_cost_bps=evaluation_cost_bps,
    )
    config_hash = BundleWriter.compute_config_hash(config)
    plan_hash = f"sha256:{config_hash}"

    writer = BundleWriter(bundle_dir)
    writer.write_plan(
        plan_hash=plan_hash,
        config_hash=plan_hash,
        as_of_time=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        config=config,
    )
    plan_path = bundle_dir / "plan.json"
    plan_payload = _load_json(plan_path)
    plan_payload["determinism_tier"] = "D1"
    plan_payload["planner_version"] = "panel_promotion_bundle/1"
    plan_payload["feature_op_registry_version"] = "1.0.0"
    atomic_write_json(plan_path, plan_payload)

    writer.write_env_fingerprint()

    panel_frame = pd.read_parquet(panel_path, columns=["date", "instrument"])
    panel_frame["date"] = pd.to_datetime(panel_frame["date"], errors="coerce")
    symbols = sorted(panel_frame["instrument"].astype(str).unique().tolist())[:500]
    writer.write_dataset_manifest(
        dataset_id=str(panel_path),
        symbols=symbols,
        row_count=int(len(panel_frame)),
        time_range={
            "start": str(panel_frame["date"].min()),
            "end": str(panel_frame["date"].max()),
        },
        pit_compliant=True,
        knowledge_time_column="date",
    )
    writer.write_preprocessing_report(
        steps=[
            {"name": "panel_model_matrix_train"},
            {"name": "gate3_candidate_portfolios"},
            {"name": "gate5_production_bridge"},
            {"name": "gate6_promotion_stats"},
        ],
        timings={},
        warnings=[],
    )

    splits_payload = build_splits_manifest_from_run(run_dir)
    writer.write_splits_manifest(
        splits=list(splits_payload["splits"]),
        split_method=str(splits_payload["split_method"]),
        purge_window=int(splits_payload["purge_window"]),
        embargo_window=int(splits_payload["embargo_window"]),
    )

    atomic_write_json(bundle_dir / "stat_validity_report.json", stat_report)
    writer._written.append("stat_validity_report.json")

    exec_assumptions = _synthesize_execution_assumptions(cost_bps=evaluation_cost_bps)
    backtest_bundle = Path.cwd() / "artifacts" / "gate6_backtest_stat" / strategy_id
    backtest_exec = backtest_bundle / "execution_assumptions.json"
    if backtest_exec.is_file():
        raw = _load_json(backtest_exec)
        raw["commission_bps"] = float(evaluation_cost_bps)
        raw["cost_model_id"] = "panel_promotion_cost"
        exec_assumptions = raw
    atomic_write_json(bundle_dir / "execution_assumptions.json", exec_assumptions)
    writer._written.append("execution_assumptions.json")

    reports_dir = run_dir / "reports"
    _copy_sidecar(
        reports_dir / "gate5_production_smoke.json", bundle_dir / "gate5_production_smoke.json"
    )
    _copy_sidecar(
        reports_dir / "gate6_promotion_report.json", bundle_dir / "gate6_promotion_report.json"
    )
    _copy_sidecar(
        reports_dir / "promotion_model_ledger.json", bundle_dir / "promotion_model_ledger.json"
    )

    writer.write_bundle_manifest()
    return bundle_dir


def run_gate7_promotion_bundle_for_model_matrix_run(
    run_dir: Path,
    *,
    panel_path: Path,
    portfolio_spec: PortfolioSpec,
    model_id: str = "xgboost",
    strategy_id: str | None = None,
    evaluation_cost_bps: float = _DEFAULT_EVALUATION_COST_BPS,
) -> dict[str, Any]:
    bundle_dir = assemble_promotion_bundle(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=portfolio_spec,
        model_id=model_id,
        strategy_id=strategy_id,
        evaluation_cost_bps=evaluation_cost_bps,
    )
    return {
        "bundle_path": str(bundle_dir),
        "model_id": model_id,
        "strategy_id": strategy_id or f"prediction_threshold_{model_id}",
    }


__all__ = [
    "assemble_promotion_bundle",
    "build_panel_promotion_plan_config",
    "build_splits_manifest_from_run",
    "run_gate7_promotion_bundle_for_model_matrix_run",
]
