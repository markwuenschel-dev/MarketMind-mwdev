"""Panel train-matrix and target root-cause investigation CLI handlers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

if TYPE_CHECKING:
    from pysrc.pipeline.p2_config_loader import PortfolioSpec

from pysrc.artifact_registry.run_layout import allocate_run_dir
from pysrc.pipeline.panel.model_matrix_diagnostics import (
    run_model_matrix_diagnostics,
    run_target_root_cause_investigation,
)
from pysrc.pipeline.panel.model_matrix_persistence_probe import (
    run_model_matrix_persistence_probe_from_yaml,
)
from pysrc.pipeline.panel.model_matrix_target_audit import (
    build_prediction_train_range_audit,
    build_train_target_fold_audit,
)
from pysrc.pipeline.panel.train_model_matrix import (
    TrainRowPolicy,
    train_model_matrix_from_yaml,
)


def resolve_run_dir(run_dir: Path) -> Path:
    """Resolve an existing model-matrix run directory."""

    path = run_dir.resolve()
    if not path.is_dir():
        raise click.ClickException(f"Run directory not found: {path}")
    report = path / "reports" / "model_matrix_report.json"
    if not report.is_file():
        raise click.ClickException(f"Missing model_matrix_report.json under {path}")
    return path


def _load_report(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "reports" / "model_matrix_report.json"
    return cast(dict[str, Any], json.loads(report_path.read_text(encoding="utf-8")))


def emit_train_target_audit(run_dir: Path, *, random_seed: int | None = None) -> dict[str, Path]:
    from pysrc.pipeline.panel.model_matrix_diagnostics import _resolve_config_and_source

    run_dir = resolve_run_dir(run_dir)
    report = _load_report(run_dir)
    config, source, target_column, _, boundaries, seed = _resolve_config_and_source(
        run_dir,
        report,
        config=None,
        config_path=None,
        random_seed=random_seed,
    )
    if source.panel_path is None:
        raise click.ClickException("Canonical panel path required for train-target audit")
    train_policy_raw = report.get("train_row_policy") or {}
    train_row_policy = TrainRowPolicy(
        general_max_rows=train_policy_raw.get("general_max_rows"),
        quantile_max_rows=train_policy_raw.get("quantile_max_rows"),
    )
    audit = build_train_target_fold_audit(
        source.panel_path,
        target_column=target_column,
        boundaries=boundaries,
        train_row_policy=train_row_policy,
        random_seed=seed,
    )
    out_path = run_dir / "reports" / "train_target_fold_audit.json"
    from pysrc.artifact_registry._atomic import atomic_write_json
    from pysrc.pipeline.panel.model_matrix_diagnostics import _json_ready

    atomic_write_json(out_path, _json_ready(audit))
    return {"train_target_fold_audit": out_path}


def emit_pred_range_audit(run_dir: Path, *, random_seed: int | None = None) -> dict[str, Path]:
    from pysrc.pipeline.panel.model_matrix_diagnostics import _resolve_config_and_source

    run_dir = resolve_run_dir(run_dir)
    report = _load_report(run_dir)
    pred_path = run_dir / "predictions" / "model_prediction_panel.parquet"
    if not pred_path.is_file():
        raise click.ClickException(f"Missing predictions: {pred_path}")
    _, source, target_column, _, boundaries, seed = _resolve_config_and_source(
        run_dir,
        report,
        config=None,
        config_path=None,
        random_seed=random_seed,
    )
    if source.panel_path is None:
        raise click.ClickException("Canonical panel path required for pred-range audit")
    audit = build_prediction_train_range_audit(
        pred_path,
        source.panel_path,
        boundaries=boundaries,
        target_column=target_column,
        random_seed=seed,
    )
    out_path = run_dir / "reports" / "prediction_train_range_audit.json"
    from pysrc.artifact_registry._atomic import atomic_write_json
    from pysrc.pipeline.panel.model_matrix_diagnostics import _json_ready

    atomic_write_json(out_path, _json_ready(audit))
    return {"prediction_train_range_audit": out_path}


def emit_investigate_targets(
    run_dir: Path,
    *,
    config_path: Path | None = None,
    random_seed: int | None = None,
    skip_diversity: bool = False,
) -> dict[str, Path]:
    return run_target_root_cause_investigation(
        resolve_run_dir(run_dir),
        config_path=config_path,
        random_seed=random_seed,
        include_diversity=not skip_diversity,
    )


def emit_audit_model_matrix(
    run_dir: Path,
    *,
    config_path: Path | None = None,
    random_seed: int | None = None,
) -> dict[str, Path]:
    return run_model_matrix_diagnostics(
        resolve_run_dir(run_dir),
        config_path=config_path,
        random_seed=random_seed,
    )


def emit_probe_persistence(
    *,
    config_path: Path | None = None,
    reference_run_dir: Path | None = None,
    smoke_test: bool = False,
    random_seed: int = 42,
    key_sample_size: int = 64,
    max_train_rows: int | None = None,
    quantile_max_train_rows: int | None = None,
) -> dict[str, Path]:
    if config_path is None:
        raise click.ClickException("panel probe-persistence requires -c/--config")
    return run_model_matrix_persistence_probe_from_yaml(
        config_path=config_path,
        smoke_test=smoke_test,
        random_seed=random_seed,
        reference_run_dir=reference_run_dir,
        key_sample_size=key_sample_size,
        max_train_rows_per_fold=max_train_rows,
        quantile_max_train_rows_per_fold=quantile_max_train_rows,
    )


def emit_train_matrix_canary(
    config_path: Path,
    *,
    random_seed: int = 42,
    processed_data_root: str | None = None,
    max_train_rows: int | None = None,
    quantile_max_train_rows: int | None = None,
    max_instruments: int | None = None,
    max_folds: int | None = None,
) -> dict[str, Path]:
    from uuid import uuid4

    run_path = allocate_run_dir(
        lane="panel_model",
        run_id=f"model_matrix_canary_{uuid4().hex[:12]}",
        smoke=False,
    )
    artifacts = train_model_matrix_from_yaml(
        config_path,
        canary=True,
        random_seed=random_seed,
        processed_data_root=processed_data_root,
        max_train_rows_per_fold=max_train_rows,
        quantile_max_train_rows_per_fold=quantile_max_train_rows,
        panel_walk_forward_folds=max_folds,
        panel_max_instruments=max_instruments,
        run_dir=run_path,
    )
    diag = run_model_matrix_diagnostics(run_path, config_path=config_path, random_seed=random_seed)
    return {**artifacts, **diag}


def emit_gate3_viability(
    run_dir: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, Path]:
    from pysrc.pipeline.candidate_portfolios.viability import (
        run_gate3_viability_for_model_matrix_run,
    )
    from pysrc.pipeline.p2_config_loader import (
        DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG,
        load_p2_config,
        parse_meta_router_experiment,
        resolve_config_path,
    )
    from pysrc.pipeline.panel.model_matrix_diagnostics import _resolve_config_and_source

    run_dir = resolve_run_dir(run_dir)
    report = _load_report(run_dir)
    _, source, target_column, _, _, _ = _resolve_config_and_source(
        run_dir,
        report,
        config=None,
        config_path=config_path,
        random_seed=None,
    )
    if source.panel_path is None:
        raise click.ClickException("Canonical panel path required for gate3 viability")
    cfg_path = resolve_config_path(
        str(config_path) if config_path is not None else None,
        default=DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG,
    )
    spec = parse_meta_router_experiment(load_p2_config(cfg_path))
    return run_gate3_viability_for_model_matrix_run(
        run_dir,
        panel_path=source.panel_path,
        portfolio_spec=spec.portfolio,
        target_column=target_column,
    )


def emit_gate4_robustness(
    run_dir: Path,
    *,
    config_path: Path | None = None,
    rebuild_gate3: bool = False,
) -> dict[str, Path]:
    from pysrc.pipeline.candidate_portfolios.viability import (
        run_gate4_robustness_for_model_matrix_run,
    )
    from pysrc.pipeline.p2_config_loader import (
        DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG,
        load_p2_config,
        parse_meta_router_experiment,
        resolve_config_path,
    )
    from pysrc.pipeline.panel.model_matrix_diagnostics import _resolve_config_and_source

    run_dir = resolve_run_dir(run_dir)
    report = _load_report(run_dir)
    _, source, target_column, _, _, _ = _resolve_config_and_source(
        run_dir,
        report,
        config=None,
        config_path=config_path,
        random_seed=None,
    )
    if source.panel_path is None:
        raise click.ClickException("Canonical panel path required for gate4 robustness")
    cfg_path = resolve_config_path(
        str(config_path) if config_path is not None else None,
        default=DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG,
    )
    spec = parse_meta_router_experiment(load_p2_config(cfg_path))
    return run_gate4_robustness_for_model_matrix_run(
        run_dir,
        panel_path=source.panel_path,
        portfolio_spec=spec.portfolio,
        target_column=target_column,
        rebuild_gate3=rebuild_gate3,
    )


def emit_production_smoke(
    run_dir: Path,
    *,
    model_id: str = "xgboost",
    config_path: Path | None = None,
) -> dict[str, Path]:
    from pysrc.pipeline.candidate_portfolios.production_bridge import (
        run_production_smoke_for_model_matrix_run,
    )
    from pysrc.pipeline.p2_config_loader import (
        DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG,
        load_p2_config,
        parse_meta_router_experiment,
        resolve_config_path,
    )
    from pysrc.pipeline.panel.model_matrix_diagnostics import _resolve_config_and_source

    run_dir = resolve_run_dir(run_dir)
    report = _load_report(run_dir)
    _, source, target_column, _, _, _ = _resolve_config_and_source(
        run_dir,
        report,
        config=None,
        config_path=config_path,
        random_seed=None,
    )
    if source.panel_path is None:
        raise click.ClickException("Canonical panel path required for production smoke")
    cfg_path = resolve_config_path(
        str(config_path) if config_path is not None else None,
        default=DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG,
    )
    spec = parse_meta_router_experiment(load_p2_config(cfg_path))
    result = run_production_smoke_for_model_matrix_run(
        run_dir,
        panel_path=source.panel_path,
        portfolio_spec=spec.portfolio,
        model_id=model_id,
        target_column=target_column,
    )
    return {"gate5_production_smoke": Path(result["report_path"])}


def emit_gate6_promotion(
    run_dir: Path,
    *,
    model_id: str = "xgboost",
    config_path: Path | None = None,
) -> dict[str, Path]:
    from pysrc.pipeline.candidate_portfolios.promotion_stats import (
        run_gate6_promotion_for_model_matrix_run,
    )
    from pysrc.pipeline.p2_config_loader import (
        DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG,
        load_p2_config,
        parse_meta_router_experiment,
        resolve_config_path,
    )
    from pysrc.pipeline.panel.model_matrix_diagnostics import _resolve_config_and_source

    run_dir = resolve_run_dir(run_dir)
    report = _load_report(run_dir)
    _, source, target_column, _, _, _ = _resolve_config_and_source(
        run_dir,
        report,
        config=None,
        config_path=config_path,
        random_seed=None,
    )
    if source.panel_path is None:
        raise click.ClickException("Canonical panel path required for gate6 promotion")
    cfg_path = resolve_config_path(
        str(config_path) if config_path is not None else None,
        default=DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG,
    )
    spec = parse_meta_router_experiment(load_p2_config(cfg_path))
    result = run_gate6_promotion_for_model_matrix_run(
        run_dir,
        panel_path=source.panel_path,
        portfolio_spec=spec.portfolio,
        model_id=model_id,
        target_column=target_column,
    )
    return {"gate6_promotion_report": Path(result["report_path"])}


def _resolve_panel_portfolio_context(
    run_dir: Path,
    *,
    config_path: Path | None,
) -> tuple[Path, Path, str, PortfolioSpec]:
    from pysrc.pipeline.p2_config_loader import (
        DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG,
        load_p2_config,
        parse_meta_router_experiment,
        resolve_config_path,
    )
    from pysrc.pipeline.panel.model_matrix_diagnostics import _resolve_config_and_source

    run_dir = resolve_run_dir(run_dir)
    report = _load_report(run_dir)
    _, source, target_column, _, _, _ = _resolve_config_and_source(
        run_dir,
        report,
        config=None,
        config_path=config_path,
        random_seed=None,
    )
    if source.panel_path is None:
        raise click.ClickException("Canonical panel path required")
    cfg_path = resolve_config_path(
        str(config_path) if config_path is not None else None,
        default=DEFAULT_CANDIDATE_PORTFOLIOS_CONFIG,
    )
    spec = parse_meta_router_experiment(load_p2_config(cfg_path))
    return run_dir, source.panel_path, target_column, spec.portfolio


def emit_assemble_promotion_bundle(
    run_dir: Path,
    *,
    model_id: str = "xgboost",
    config_path: Path | None = None,
    validate: bool = False,
) -> dict[str, Path]:
    from pysrc.artifact_registry._atomic import atomic_write_json
    from pysrc.cli.gate import ExitCode, validate_bundle, write_gate_report
    from pysrc.pipeline.candidate_portfolios.promotion_bundle import assemble_promotion_bundle
    from pysrc.pipeline.candidate_portfolios.promotion_stats import build_promotion_model_ledger
    from pysrc.pipeline.panel.model_matrix_diagnostics import _json_ready

    run_dir, panel_path, _target_column, portfolio_spec = _resolve_panel_portfolio_context(
        run_dir,
        config_path=config_path,
    )
    ledger = build_promotion_model_ledger(run_dir, selected_model=model_id)
    ledger_path = run_dir / "reports" / "promotion_model_ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ledger_path, _json_ready(ledger))

    bundle_dir = assemble_promotion_bundle(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=portfolio_spec,
        model_id=model_id,
    )
    paths: dict[str, Path] = {
        "promotion_bundle": bundle_dir,
        "promotion_model_ledger": ledger_path,
    }
    if validate:
        gate_report, exit_code = validate_bundle(bundle_dir)
        gate_result_path = bundle_dir / "gate_result.json"
        write_gate_report(gate_report, gate_result_path, bundle_dir)
        paths["gate_result"] = gate_result_path
        if exit_code != ExitCode.PASS:
            raise click.ClickException(f"mm-gate validate failed with exit code {exit_code.value}")
    return paths


def emit_promotion_finish(
    run_dir: Path,
    *,
    model_id: str = "xgboost",
    config_path: Path | None = None,
    run_stat_battery: bool = False,
) -> dict[str, Path]:
    from pysrc.pipeline.candidate_portfolios.promotion_stats import (
        run_pdr001_finish_for_model_matrix_run,
    )

    run_dir, panel_path, target_column, portfolio_spec = _resolve_panel_portfolio_context(
        run_dir,
        config_path=config_path,
    )
    result = run_pdr001_finish_for_model_matrix_run(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=portfolio_spec,
        model_id=model_id,
        target_column=target_column,
        run_stat_battery=run_stat_battery,
    )
    if not result.get("finish_pass"):
        raise click.ClickException("PDR-001 finish report: finish_pass is false")
    return {
        "pdr001_finish_report": Path(result["report_path"]),
        "panel_promotion_manifest": Path(result["manifest_path"]),
        "promotion_bundle": Path(result["bundle_path"]),
    }


def emit_policy_smoke(
    run_dir: Path,
    *,
    config_path: Path | None = None,
    emit_target_plans: bool = False,
    report_name: str | None = None,
) -> dict[str, Path]:
    from pysrc.pipeline.candidate_portfolios.policy_bridge import (
        run_policy_smoke_for_model_matrix_run,
    )
    from pysrc.pipeline.p2_config_loader import (
        load_p2_config,
        resolve_config_path,
        yaml_to_meta_router_config,
    )

    run_dir, panel_path, target_column, portfolio_spec = _resolve_panel_portfolio_context(
        run_dir,
        config_path=config_path,
    )
    cfg_path = resolve_config_path(
        str(config_path) if config_path is not None else None,
        default="research/p2/configs/candidate_portfolios.yaml",
    )
    yaml_dict = load_p2_config(cfg_path)
    router_config = yaml_to_meta_router_config(yaml_dict, config_path=cfg_path)
    result = run_policy_smoke_for_model_matrix_run(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=portfolio_spec,
        target_column=target_column,
        emit_target_plans=emit_target_plans,
        config=router_config,
        report_name=report_name,
    )
    if int(result.get("routed_test_economics", {}).get("n_days", 0)) <= 0:
        raise click.ClickException("Policy smoke: no routed test economics")
    paths: dict[str, Path] = {"policy_allocation_report": Path(result["report_path"])}
    plans_path = result.get("portfolio_target_plans_path")
    if plans_path:
        paths["portfolio_target_plans"] = Path(str(plans_path))
    return paths


def emit_policy_sweep(
    run_dir: Path,
    *,
    config_path: Path | None = None,
    report_name: str | None = None,
) -> dict[str, Path]:
    from pysrc.pipeline.candidate_portfolios.policy_bridge import run_policy_sweep

    run_dir, panel_path, target_column, portfolio_spec = _resolve_panel_portfolio_context(
        run_dir,
        config_path=config_path,
    )
    result = run_policy_sweep(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=portfolio_spec,
        target_column=target_column,
        report_name=report_name,
    )
    if int(result.get("best_routed_test_economics", {}).get("n_days", 0)) <= 0:
        raise click.ClickException("Policy sweep: no routed test economics for best config")
    return {"policy_sweep_results": Path(result["report_path"])}


def emit_meta_router_eval(
    run_dir: Path,
    *,
    config_path: Path | None = None,
    report_name: str | None = None,
) -> dict[str, Path]:
    from pysrc.pipeline.candidate_portfolios.meta_router_eval import run_meta_router_evaluation
    from pysrc.pipeline.p2_config_loader import (
        DEFAULT_META_ROUTER_CONFIG,
        load_p2_config,
        resolve_config_path,
        yaml_to_meta_router_config,
    )

    run_dir, panel_path, target_column, portfolio_spec = _resolve_panel_portfolio_context(
        run_dir,
        config_path=config_path,
    )
    cfg_path = resolve_config_path(
        str(config_path) if config_path is not None else None,
        default=DEFAULT_META_ROUTER_CONFIG,
    )
    router_config = yaml_to_meta_router_config(load_p2_config(cfg_path), config_path=cfg_path)
    yaml_dict = load_p2_config(cfg_path)
    result = run_meta_router_evaluation(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=portfolio_spec,
        config=router_config,
        target_column=target_column,
        yaml_dict=yaml_dict,
        report_name=report_name,
    )
    if int(result.get("baseline_test_economics", {}).get("n_days", 0)) <= 0:
        raise click.ClickException("Meta-router eval: no baseline test economics")
    return {"meta_router_evaluation_report": Path(result["report_path"])}
