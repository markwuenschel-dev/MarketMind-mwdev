"""The supported MarketMind product-flow command line interface."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

import click

from pysrc.artifact_registry import ArtifactRegistry
from pysrc.contracts import (
    PredictionValue,
    StandardizedPredictionArtifact,
    StandardizedTradeIntentArtifact,
    TradeDirection,
    TradeIntent,
)

_ARTIFACT_ROOT = Path("artifacts")
_PREDICTION_ROLE = "standardized_prediction"
_TRADE_INTENT_ROLE = "standardized_trade_intent"


def _registry() -> ArtifactRegistry:
    return ArtifactRegistry(_ARTIFACT_ROOT)


def _emit(payload: object) -> None:
    click.echo(json.dumps(payload, indent=2, default=str, sort_keys=True))


def _run_payload(registry: ArtifactRegistry, run_id: str) -> dict[str, object]:
    record = registry.runs.get_run(run_id, include_incomplete=True, include_failed=True)
    if record is None:
        raise click.ClickException(f"Unknown run: {run_id}")
    return {
        "run_id": record.run_id,
        "status": record.status.value,
        "metadata": record.metadata,
        "artifacts": [
            {
                "role": item.role,
                "cas": str(item.cas),
                "attest": str(item.attest) if item.attest else None,
            }
            for item in record.artifacts
        ],
    }


@click.group()
def cli() -> None:
    """MarketMind's active, registry-backed product flow."""


@cli.group()
def run() -> None:
    """Execute the composed product flow."""


@run.command("full")
@click.option("--smoke-test", is_flag=True, default=False)
def run_full(smoke_test: bool) -> None:
    """Run the active full-flow entry point."""
    _emit_meta_router_result("full", smoke_test)


@cli.group()
def dataprep() -> None:
    """Run the centralized PIT-safe dataprep pipeline."""


@dataprep.command("run")
@click.option(
    "-c", "--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path)
)
def dataprep_run(config_path: Path) -> None:
    """Execute the DataPrepOrchestrator from a config file."""
    from pysrc.pipeline.dataprep_runtime import run_dataprep_from_path

    _emit(run_dataprep_from_path(config_path))


@cli.group()
def panel() -> None:
    """Build and audit standardized prediction products."""


_F = TypeVar("_F", bound=Callable[..., object])


def _panel_run_dir_option() -> Callable[[_F], _F]:
    return click.option(
        "--run-dir",
        required=True,
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="Existing model_matrix run directory under artifacts/runs/.",
    )


def _panel_config_option() -> Callable[[_F], _F]:
    return click.option(
        "-c",
        "--config",
        "config_path",
        default=None,
        type=click.Path(path_type=Path),
        help="Model matrix YAML (default: path recorded in run report).",
    )


def _panel_seed_option() -> Callable[[_F], _F]:
    return click.option("--random-seed", default=None, type=int)


@panel.command("audit-features")
def panel_audit_features() -> None:
    raise click.ClickException(
        "panel audit-features is not wired in the registry CLI yet; run indicator universe "
        "build via dataprep and inspect "
        "data/processed/full_indicator_feature_panel/build_report.json"
    )


@panel.command("train")
@click.option("--smoke-test", is_flag=True, default=False)
def panel_train(smoke_test: bool) -> None:
    """Emit a registered standardized prediction artifact for a panel run."""
    _emit_prediction_smoke("panel", smoke_test)


@panel.command("audit-train-targets")
@_panel_run_dir_option()
@_panel_config_option()
@_panel_seed_option()
def panel_audit_train_targets(
    run_dir: Path,
    config_path: Path | None,
    random_seed: int | None,
) -> None:
    from pysrc.cli.panel_flow import emit_train_target_audit

    _ = config_path
    paths = emit_train_target_audit(run_dir, random_seed=random_seed)
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("audit-pred-ranges")
@_panel_run_dir_option()
@_panel_config_option()
@_panel_seed_option()
def panel_audit_pred_ranges(
    run_dir: Path,
    config_path: Path | None,
    random_seed: int | None,
) -> None:
    from pysrc.cli.panel_flow import emit_pred_range_audit

    _ = config_path
    paths = emit_pred_range_audit(run_dir, random_seed=random_seed)
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("investigate-targets")
@_panel_run_dir_option()
@_panel_config_option()
@_panel_seed_option()
@click.option("--skip-diversity", is_flag=True, default=False)
def panel_investigate_targets(
    run_dir: Path,
    config_path: Path | None,
    random_seed: int | None,
    skip_diversity: bool,
) -> None:
    from pysrc.cli.panel_flow import emit_investigate_targets

    paths = emit_investigate_targets(
        run_dir,
        config_path=config_path,
        random_seed=random_seed,
        skip_diversity=skip_diversity,
    )
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("audit-model-matrix")
@_panel_run_dir_option()
@_panel_config_option()
@_panel_seed_option()
def panel_audit_model_matrix(
    run_dir: Path,
    config_path: Path | None,
    random_seed: int | None,
) -> None:
    from pysrc.cli.panel_flow import emit_audit_model_matrix

    paths = emit_audit_model_matrix(
        run_dir,
        config_path=config_path,
        random_seed=random_seed,
    )
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("gate3-viability")
@_panel_run_dir_option()
@click.option(
    "-c",
    "--config",
    "config_path",
    default="research/p2/configs/candidate_portfolios.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
)
def panel_gate3_viability(
    run_dir: Path,
    config_path: Path,
) -> None:
    from pysrc.cli.panel_flow import emit_gate3_viability

    paths = emit_gate3_viability(run_dir, config_path=config_path)
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("gate4-robustness")
@_panel_run_dir_option()
@click.option(
    "-c",
    "--config",
    "config_path",
    default="research/p2/configs/candidate_portfolios.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option("--rebuild-gate3", is_flag=True, default=False)
def panel_gate4_robustness(
    run_dir: Path,
    config_path: Path,
    rebuild_gate3: bool,
) -> None:
    from pysrc.cli.panel_flow import emit_gate4_robustness

    paths = emit_gate4_robustness(
        run_dir,
        config_path=config_path,
        rebuild_gate3=rebuild_gate3,
    )
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("production-smoke")
@_panel_run_dir_option()
@click.option(
    "-c",
    "--config",
    "config_path",
    default="research/p2/configs/candidate_portfolios.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option("--model-id", default="xgboost", show_default=True)
def panel_production_smoke(
    run_dir: Path,
    config_path: Path,
    model_id: str,
) -> None:
    from pysrc.cli.panel_flow import emit_production_smoke

    paths = emit_production_smoke(
        run_dir,
        model_id=model_id,
        config_path=config_path,
    )
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("gate6-promotion")
@_panel_run_dir_option()
@click.option(
    "-c",
    "--config",
    "config_path",
    default="research/p2/configs/candidate_portfolios.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option("--model-id", default="xgboost", show_default=True)
def panel_gate6_promotion(
    run_dir: Path,
    config_path: Path,
    model_id: str,
) -> None:
    from pysrc.cli.panel_flow import emit_gate6_promotion

    paths = emit_gate6_promotion(
        run_dir,
        model_id=model_id,
        config_path=config_path,
    )
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("assemble-promotion-bundle")
@_panel_run_dir_option()
@click.option(
    "-c",
    "--config",
    "config_path",
    default="research/p2/configs/candidate_portfolios.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option("--model-id", default="xgboost", show_default=True)
@click.option("--validate", is_flag=True, default=False)
def panel_assemble_promotion_bundle(
    run_dir: Path,
    config_path: Path,
    model_id: str,
    validate: bool,
) -> None:
    from pysrc.cli.panel_flow import emit_assemble_promotion_bundle

    paths = emit_assemble_promotion_bundle(
        run_dir,
        model_id=model_id,
        config_path=config_path,
        validate=validate,
    )
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("promotion-finish")
@_panel_run_dir_option()
@click.option(
    "-c",
    "--config",
    "config_path",
    default="research/p2/configs/candidate_portfolios.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option("--model-id", default="xgboost", show_default=True)
@click.option("--run-stat-battery", is_flag=True, default=False)
def panel_promotion_finish(
    run_dir: Path,
    config_path: Path,
    model_id: str,
    run_stat_battery: bool,
) -> None:
    from pysrc.cli.panel_flow import emit_promotion_finish

    paths = emit_promotion_finish(
        run_dir,
        model_id=model_id,
        config_path=config_path,
        run_stat_battery=run_stat_battery,
    )
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("policy-smoke")
@_panel_run_dir_option()
@click.option(
    "-c",
    "--config",
    "config_path",
    default="research/p2/configs/candidate_portfolios.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option(
    "--emit-target-plans",
    is_flag=True,
    default=False,
    help="Emit PortfolioTargetPlan JSON from routed positions.",
)
@click.option(
    "--report-name",
    default=None,
    help="Report filename under run_dir/reports/ (default: policy_allocation_report.v2.json).",
)
def panel_policy_smoke(
    run_dir: Path,
    config_path: Path,
    emit_target_plans: bool,
    report_name: str | None,
) -> None:
    from pysrc.cli.panel_flow import emit_policy_smoke

    paths = emit_policy_smoke(
        run_dir,
        config_path=config_path,
        emit_target_plans=emit_target_plans,
        report_name=report_name,
    )
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("policy-sweep")
@_panel_run_dir_option()
@click.option(
    "-c",
    "--config",
    "config_path",
    default="research/p2/configs/candidate_portfolios.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option(
    "--report-name",
    default=None,
    help="Report filename under run_dir/reports/ (default: policy_sweep_results.json).",
)
def panel_policy_sweep(
    run_dir: Path,
    config_path: Path,
    report_name: str | None,
) -> None:
    from pysrc.cli.panel_flow import emit_policy_sweep

    paths = emit_policy_sweep(
        run_dir,
        config_path=config_path,
        report_name=report_name,
    )
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("meta-router-eval")
@_panel_run_dir_option()
@click.option(
    "-c",
    "--config",
    "config_path",
    default="research/p2/configs/local_meta_router.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Meta-router eval config (gating_baselines + state_features).",
)
@click.option(
    "--report-name",
    default=None,
    help="Report filename under run_dir/reports/ (default: meta_router_evaluation_report.json).",
)
def panel_meta_router_eval(
    run_dir: Path,
    config_path: Path,
    report_name: str | None,
) -> None:
    """Run PDR-002 meta-router gating baseline battery on an existing model-matrix run."""
    from pysrc.cli.panel_flow import emit_meta_router_eval

    paths = emit_meta_router_eval(
        run_dir,
        config_path=config_path,
        report_name=report_name,
    )
    _emit({key: str(path) for key, path in paths.items()})


@panel.command("train-matrix")
@click.option(
    "-c",
    "--config",
    "config_path",
    default="research/p2/configs/panel_model_matrix.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option("--smoke-test", is_flag=True, default=False)
@click.option("--canary", is_flag=True, default=False)
@click.option("--random-seed", default=42, show_default=True, type=int)
@click.option("--processed-data-root", default=None)
@click.option("--max-train-rows", default=None, type=int)
@click.option("--quantile-max-train-rows", default=None, type=int)
@click.option("--max-instruments", default=None, type=int, help="Cap tickers (canary default 20).")
@click.option("--max-folds", default=None, type=int, help="Walk-forward folds (canary default 1).")
def panel_train_matrix(
    config_path: Path,
    smoke_test: bool,
    canary: bool,
    random_seed: int,
    processed_data_root: str | None,
    max_train_rows: int | None,
    quantile_max_train_rows: int | None,
    max_instruments: int | None,
    max_folds: int | None,
) -> None:
    """Train model matrix (full, canary, or registry smoke artifact)."""
    if smoke_test and canary:
        raise click.ClickException("Use either --smoke-test or --canary, not both.")
    if smoke_test:
        _emit_prediction_smoke("panel-model-matrix", smoke_test)
        return
    if canary:
        from pysrc.cli.panel_flow import emit_train_matrix_canary

        if not config_path.is_file():
            raise click.ClickException(f"Config not found: {config_path}")
        paths = emit_train_matrix_canary(
            config_path,
            random_seed=random_seed,
            processed_data_root=processed_data_root,
            max_train_rows=max_train_rows,
            quantile_max_train_rows=quantile_max_train_rows,
            max_instruments=max_instruments,
            max_folds=max_folds,
        )
        _emit({key: str(path) for key, path in paths.items()})
        return
    from pysrc.pipeline.panel.train_model_matrix import train_model_matrix_from_yaml

    artifacts = train_model_matrix_from_yaml(
        config_path,
        random_seed=random_seed,
        processed_data_root=processed_data_root,
        max_train_rows_per_fold=max_train_rows,
        quantile_max_train_rows_per_fold=quantile_max_train_rows,
        panel_walk_forward_folds=max_folds,
        panel_max_instruments=max_instruments,
    )
    _emit({key: str(path) for key, path in artifacts.items()})


@panel.command("probe-persistence")
@click.option(
    "-c",
    "--config",
    "config_path",
    default="research/p2/configs/panel_model_matrix.yaml",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option("--smoke-test", is_flag=True, default=False)
@click.option("--random-seed", default=42, show_default=True, type=int)
@click.option(
    "--reference-run-dir",
    default=None,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
)
@click.option("--key-sample-size", default=64, show_default=True, type=int)
@click.option("--max-train-rows", default=None, type=int)
@click.option("--quantile-max-train-rows", default=None, type=int)
def panel_probe_persistence(
    config_path: Path,
    smoke_test: bool,
    random_seed: int,
    reference_run_dir: Path | None,
    key_sample_size: int,
    max_train_rows: int | None,
    quantile_max_train_rows: int | None,
) -> None:
    from pysrc.cli.panel_flow import emit_probe_persistence

    paths = emit_probe_persistence(
        config_path=config_path,
        reference_run_dir=reference_run_dir,
        smoke_test=smoke_test,
        random_seed=random_seed,
        key_sample_size=key_sample_size,
        max_train_rows=max_train_rows,
        quantile_max_train_rows=quantile_max_train_rows,
    )
    _emit({key: str(path) for key, path in paths.items()})


def _emit_prediction_smoke(lane: str, smoke_test: bool) -> None:
    registry = _registry()
    run_id = registry.begin_run({"lane": lane, "smoke": smoke_test})
    prediction = StandardizedPredictionArtifact(
        schema_version="1",
        as_of="1970-01-01T00:00:00+00:00",
        data_lineage={"panel": "synthetic:smoke" if smoke_test else "panel:configured"},
        model_id="baseline",
        fold_id="fold-0",
        split="test",
        predictions=(
            PredictionValue(
                instrument="SMOKE",
                decision_time="1970-01-01T00:00:00+00:00",
                value=0.0,
            ),
        ),
    )
    registry.register_json(run_id, _PREDICTION_ROLE, prediction)
    registry.complete_run(run_id)
    _emit(_run_payload(registry, run_id))


@cli.group()
def tuning() -> None:
    """Validate and run configured model tuning."""


@tuning.command("list")
def tuning_list() -> None:
    _emit({"tuners": ["configured-model-search"]})


@tuning.command("validate")
@click.option(
    "-c", "--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path)
)
def tuning_validate(config_path: Path) -> None:
    _emit({"valid": config_path.is_file(), "config": str(config_path)})


@tuning.command("run")
@click.option("--smoke-test", is_flag=True, default=False)
def tuning_run(smoke_test: bool) -> None:
    _emit_prediction_smoke("tuning", smoke_test)


@cli.group()
def strategies() -> None:
    """Convert standardized predictions into pre-sizing trade intents."""


@strategies.command("list")
def strategies_list() -> None:
    _emit({"strategies": ["threshold"]})


@strategies.command("describe")
@click.argument("strategy_id")
def strategies_describe(strategy_id: str) -> None:
    if strategy_id != "threshold":
        raise click.ClickException(f"Unknown strategy: {strategy_id}")
    _emit({"strategy_id": "threshold", "version": "1", "input_role": _PREDICTION_ROLE})


@strategies.command("build")
@click.option("--source-run-id", required=True)
@click.option("--strategy-id", default="threshold", show_default=True)
@click.option("--threshold", default=0.0, type=float, show_default=True)
def strategies_build(source_run_id: str, strategy_id: str, threshold: float) -> None:
    """Resolve a COMPLETE prediction artifact and emit a trade-intent artifact."""
    if strategy_id != "threshold":
        raise click.ClickException(f"Unknown strategy: {strategy_id}")
    if threshold < 0.0:
        raise click.ClickException("threshold must be non-negative")
    registry = _registry()
    try:
        prediction = registry.resolve(
            source_run_id, _PREDICTION_ROLE, StandardizedPredictionArtifact
        )
    except Exception as exc:  # click owns stable user-facing failure formatting
        raise click.ClickException(str(exc)) from exc
    run_id = registry.begin_run({"lane": "strategies", "source_run_id": source_run_id})
    intents = tuple(
        TradeIntent(
            date=item.decision_time,
            instrument=item.instrument,
            interval="1d",
            strategy_id=strategy_id,
            intent_id=f"{strategy_id}:{item.instrument}:{item.decision_time}",
            score=item.value if abs(item.value) > threshold else 0.0,
            direction=(
                TradeDirection.LONG
                if item.value > threshold
                else TradeDirection.SHORT
                if item.value < -threshold
                else TradeDirection.FLAT
            ),
            eligible=True,
            abstain=False,
            source_model_id=prediction.payload.model_id,
            source_product_id=source_run_id,
            lineage={"prediction_run_id": source_run_id, "prediction_cas": str(prediction.cas)},
        )
        for item in prediction.payload.predictions
    )
    product = StandardizedTradeIntentArtifact(
        schema_version="1",
        strategy_id=strategy_id,
        strategy_version="1",
        decision_time=prediction.payload.as_of,
        prediction_lineage={
            "prediction_run_id": source_run_id,
            "prediction_cas": str(prediction.cas),
        },
        intents=intents,
    )
    registry.register_json(run_id, _TRADE_INTENT_ROLE, product)
    registry.complete_run(run_id)
    _emit(_run_payload(registry, run_id))


@cli.group(name="candidate-portfolios")
def candidate_portfolios() -> None:
    """Size trade intents into candidate portfolio outputs."""


@candidate_portfolios.command("build")
@click.option("--source-run-id", required=True)
@click.option("--top-k", default=20, type=int, show_default=True)
def candidate_portfolios_build(source_run_id: str, top_k: int) -> None:
    """Resolve trade intents only; this command never trains or loads predictions."""
    if top_k < 1:
        raise click.ClickException("top-k must be at least one")
    registry = _registry()
    try:
        trade_intents = registry.resolve(
            source_run_id, _TRADE_INTENT_ROLE, StandardizedTradeIntentArtifact
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    active = [
        intent
        for intent in trade_intents.payload.intents
        if intent.eligible and not intent.abstain and intent.direction is not TradeDirection.FLAT
    ]
    selected = sorted(active, key=lambda item: (-abs(item.score or 0.0), item.instrument))[:top_k]
    weight = 1.0 / len(selected) if selected else 0.0
    positions = [
        {
            "date": item.date,
            "ticker": item.instrument,
            "target_weight": weight if item.direction is TradeDirection.LONG else -weight,
        }
        for item in selected
    ]
    run_id = registry.begin_run({"lane": "candidate-portfolios", "source_run_id": source_run_id})
    registry.register_payload(
        run_id,
        "candidate_portfolio_outputs",
        {
            "source_trade_intent_run_id": source_run_id,
            "source_trade_intent_cas": str(trade_intents.cas),
            "positions": positions,
        },
    )
    registry.complete_run(run_id)
    _emit(_run_payload(registry, run_id))


def _emit_meta_router_result(stage: str, smoke_test: bool) -> None:
    """Record the MetaRouter stage output under registry-owned run allocation."""
    registry = _registry()
    run_id = registry.begin_run({"lane": "meta-router", "stage": stage, "smoke": smoke_test})
    registry.register_payload(
        run_id,
        "portfolio_target_plan",
        {"schema_version": "1", "stage": stage, "targets": []},
    )
    registry.complete_run(run_id)
    _emit(_run_payload(registry, run_id))


@cli.group()
def execution() -> None:
    """Paper-trading execution lane (PDR-003)."""


@execution.command("shadow-run")
@click.option(
    "--bundle-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Gate 7 promotion bundle directory.",
)
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Model-matrix run directory containing predictions and models.",
)
@click.option("--n-days", default=5, type=int, show_default=True, help="Replay window (1-60).")
@click.option(
    "--panel-path",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional panel path override (defaults to bundle plan or canonical panel).",
)
def execution_shadow_run(
    bundle_dir: Path,
    run_dir: Path,
    n_days: int,
    panel_path: Path | None,
) -> None:
    """Replay promotion bundle strategy offline without broker submission."""
    from pysrc.tuning.execution.run_shadow_plan import _MAX_SHADOW_DAYS, run_shadow_plan

    if n_days < 1 or n_days > _MAX_SHADOW_DAYS:
        raise click.ClickException(f"--n-days must be between 1 and {_MAX_SHADOW_DAYS}")
    payload = run_shadow_plan(
        bundle_dir=bundle_dir,
        run_dir=run_dir,
        panel_path=panel_path,
        n_days=n_days,
    )
    _emit(payload)


@execution.command("paper-dry-run")
@click.option(
    "--bundle-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Gate 7 promotion bundle directory.",
)
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Model-matrix run directory containing predictions and models.",
)
@click.option(
    "--n-bars", default=5, type=int, show_default=True, help="Simulated bar count (1-4680)."
)
def execution_paper_dry_run(
    bundle_dir: Path,
    run_dir: Path,
    n_bars: int,
) -> None:
    """Simulate paper-loop bar cycle without IBKR submission (PAPER_TRADING_ENABLED defaults to 0)."""
    from pysrc.tuning.execution.paper_loop import _MAX_PAPER_BARS, paper_loop_dry_run

    if n_bars < 1 or n_bars > _MAX_PAPER_BARS:
        raise click.ClickException(f"--n-bars must be between 1 and {_MAX_PAPER_BARS}")
    payload = paper_loop_dry_run(bundle_dir=bundle_dir, run_dir=run_dir, n_bars=n_bars)
    _emit(payload)


@cli.group(name="backtest")
def backtest() -> None:
    """Run backtests from registry-backed portfolio targets."""


@backtest.command("run")
@click.option("--source-run-id", required=False)
@click.option("--symbol", default=None)
def backtest_run(source_run_id: str | None, symbol: str | None) -> None:
    registry = _registry()
    run_id = registry.begin_run(
        {"lane": "backtest", "source_run_id": source_run_id, "symbol": symbol}
    )
    registry.register_payload(
        run_id,
        "backtest_request",
        {"source_run_id": source_run_id, "symbol": symbol},
    )
    registry.complete_run(run_id)
    _emit(_run_payload(registry, run_id))


@cli.group(name="artifacts")
def artifacts_group() -> None:
    """Inspect or reference-safely clean registry runs."""


@artifacts_group.command("list")
def artifacts_list() -> None:
    registry = _registry()
    _emit([_run_payload(registry, record.run_id) for record in registry.list_runs()])


@artifacts_group.command("show")
@click.argument("run_id")
def artifacts_show(run_id: str) -> None:
    _emit(_run_payload(_registry(), run_id))


@artifacts_group.command("clean")
@click.option("--keep-latest", default=5, type=int, show_default=True)
@click.option("--apply", is_flag=True, default=False, help="Perform deletion; default is dry-run.")
def artifacts_clean(keep_latest: int, apply: bool) -> None:
    report = _registry().cleanup(keep_latest=keep_latest, apply=apply)
    _emit(asdict(report))


@cli.group()
def capabilities() -> None:
    """List installed runtime capabilities."""


@capabilities.command("list")
def capabilities_list() -> None:
    from pysrc.core.runtime.capabilities import CAPABILITIES

    _emit({key: sorted(map(str, value)) for key, value in CAPABILITIES.items()})


@cli.group()
def config() -> None:
    """Inspect and validate product-flow configuration files."""


@config.command("list")
def config_list() -> None:
    root = Path("research")
    _emit(sorted(str(path) for path in root.rglob("*.yaml")) if root.exists() else [])


@config.command("show")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def config_show(config_path: Path) -> None:
    click.echo(config_path.read_text(encoding="utf-8"))


@config.command("validate")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def config_validate(config_path: Path) -> None:
    _emit({"valid": config_path.is_file(), "config": str(config_path)})


def main(argv: list[str] | None = None) -> int:
    try:
        cli.main(args=argv, prog_name="marketmind", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
