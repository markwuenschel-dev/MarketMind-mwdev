from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import pandas as pd

from pysrc.artifact_registry import LocalCAS
from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.artifact_registry.run_registry import RunRegistry, RunStatus
from pysrc.backtesting.data.pit import PITSafeDataView
from pysrc.cli.gate import ReasonCode, emit_gate_failure_report
from pysrc.data.dataview import DataView
from pysrc.data.dataview_asof_adapter import DataViewAsOfAdapter
from pysrc.pipeline import orchestrator as pipeline_orchestrator
from pysrc.strategies.momentum.alpha_ir import AlphaIR
from pysrc.strategies.momentum.artifacts.cpcv_path_scores import build_cpcv_path_score_surface
from pysrc.strategies.momentum.artifacts.signal_card import RunMeta, build_signal_card_payload
from pysrc.strategies.momentum.artifacts.stat_validity import build_stat_validity_payload
from pysrc.strategies.momentum.exceptions import (
    ConvergenceError,
    CostGateRejection,
    MissingExecutionAssumptionsError,
)
from pysrc.strategies.momentum.strategy import MomentumStrategy
from pysrc.strategies.pipeline_strategy import MaterializationError, StrategyContext, TradeIntent

_SHARED_TRIAL_COUNTER_FAMILY = "momentum.phase_i.production_v1"


class OrchestratorHooks(Protocol):
    def apply_crash_override(
        self,
        *,
        trade_intent: TradeIntent,
        alpha_ir: AlphaIR,
        ctx: StrategyContext,
        strategy: MomentumStrategy,
    ) -> TradeIntent | None: ...

    def apply_cost_gate(
        self,
        *,
        trade_intent: TradeIntent,
        alpha_ir: AlphaIR,
        ctx: StrategyContext,
        strategy: MomentumStrategy,
        execution_assumptions_path: Path,
        bundle_dir: Path,
    ) -> TradeIntent | None: ...


@dataclass(frozen=True)
class RunResult:
    trade_intent: TradeIntent
    alpha_ir: AlphaIR
    bundle_dir: Path
    artifacts: dict[str, Path] = field(default_factory=dict)


def _build_execution_assumptions_payload(strategy: Any) -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "strategy": "momentum",
        "variant": str(strategy.params["variant"]),
        "commission_bps": float(strategy.params.get("commission_bps", 5.0)),
        "slippage_bps": float(strategy.params.get("slippage_bps", 1.0)),
        "cost_model_id": str(strategy.params.get("cost_model_id", "momentum.phase_i.default")),
    }


def _write_execution_assumptions(store: BundleBacktestArtifactStore, strategy: Any) -> Path:
    payload = _build_execution_assumptions_payload(strategy)
    store.put_json("execution_assumptions.json", payload)
    return store.bundle_dir / "execution_assumptions.json"


def _coerce_returns(features: pd.DataFrame | object) -> list[float]:
    if isinstance(features, pd.DataFrame) and "returns" in features.columns:
        return [float(value) for value in features["returns"].dropna().tolist()]
    return [0.0] * 12


def _trial_counter_family(variant: str) -> str:
    if variant in {"xsec", "tsmom", "dual"}:
        return _SHARED_TRIAL_COUNTER_FAMILY
    return f"momentum.{variant}"


def _finalize_managed_run(
    run_registry: RunRegistry | None,
    run_id: str | None,
    *,
    status: RunStatus,
) -> None:
    if run_registry is None or run_id is None:
        return
    run_record = run_registry.get_run(run_id, include_incomplete=True, include_failed=True)
    if run_record is None or run_record.status is not RunStatus.REGISTERING:
        return
    run_registry.finalize_run(run_id, status)


def _build_pit_input(
    ctx: StrategyContext,
) -> tuple[PITSafeDataView | None, list[datetime] | None, pd.DataFrame | None]:
    if not isinstance(ctx.prices, pd.DataFrame):
        return None, None, None

    prices = ctx.prices
    required = {"symbol", "valid_time", "knowledge_time"}
    present = required.intersection(prices.columns)
    if present and present != required:
        missing = ", ".join(sorted(required.difference(prices.columns)))
        raise MaterializationError(
            "Governed momentum PIT corpus inputs must include symbol, valid_time, "
            f"and knowledge_time columns; missing {missing}."
        )
    if not required.issubset(prices.columns):
        return None, None, None
    if "close" not in prices.columns:
        raise MaterializationError(
            "Governed momentum PIT corpus inputs must include a 'close' column."
        )

    symbols = sorted({str(symbol) for symbol in prices["symbol"].dropna().unique().tolist()})
    if len(symbols) != 1:
        raise MaterializationError(
            "Governed momentum PIT corpus inputs currently require a single symbol on the "
            "canonical verification route."
        )

    dataview = DataView(pit_required=True)
    dataview.register_source(
        prices,
        valid_time_col="valid_time",
        knowledge_time_col="knowledge_time",
    )
    knowledge_dates = sorted(
        {
            datetime.combine(pd.Timestamp(value).date(), datetime.min.time()).replace(tzinfo=UTC)
            for value in prices["knowledge_time"].tolist()
        }
    )
    if not knowledge_dates:
        raise MaterializationError(
            "Governed momentum PIT corpus inputs require at least one knowledge_time value."
        )

    adapter = DataViewAsOfAdapter(dataview=dataview, symbols=symbols, fields=["close"])
    pit_view = PITSafeDataView(
        view=adapter,
        metadata={
            "knowledge_dates": knowledge_dates,
            "pit_enforced": True,
            "pit_front_door": "pysrc.data.dataview.DataView",
        },
    )
    return pit_view, knowledge_dates, prices


class _MomentumRuntimeAdapter:
    def __init__(
        self,
        strategy: Any,
        *,
        ctx: StrategyContext,
        orchestrator_hooks: OrchestratorHooks | None,
        target_bundle_dir: Path,
        enable_crash_override: bool,
        run_registry: RunRegistry | None,
        pbo_path_pairs: list[dict[str, Any]] | None,
        splits_manifest_override: dict[str, Any] | None,
    ) -> None:
        self.params = getattr(strategy, "params", {"variant": "xsec"})
        self._strategy = strategy
        self._ctx = ctx
        self._orchestrator_hooks = orchestrator_hooks
        self._target_bundle_dir = target_bundle_dir
        self._enable_crash_override = enable_crash_override
        self._run_registry = run_registry
        self._pbo_path_pairs = pbo_path_pairs
        self._splits_manifest_override = splits_manifest_override
        self._store: BundleBacktestArtifactStore | None = None
        self._bound_run_id: str | None = None
        self.trade_intent: TradeIntent | None = None
        self.alpha_ir: AlphaIR | None = None
        self.execution_assumptions_path: Path | None = None

    def bind_runtime_artifact_store(
        self,
        *,
        store: BundleBacktestArtifactStore,
        run_id: str,
        bundle_dir: Path,
    ) -> None:
        self._store = store
        self._bound_run_id = run_id
        self._target_bundle_dir = bundle_dir

    def generate_trade_intent(self, ctx: StrategyContext) -> TradeIntent:
        trade_intent = cast(TradeIntent, self._strategy.generate_trade_intent(ctx))
        alpha_ir = trade_intent.raw.get("alpha_ir")
        if not isinstance(alpha_ir, AlphaIR):
            raise TypeError("Momentum entry.run requires TradeIntent.raw['alpha_ir']")
        if self._store is None:
            raise MissingExecutionAssumptionsError(
                "execution_assumptions.json requires a bound canonical artifact store."
            )

        execution_assumptions_path = _write_execution_assumptions(self._store, self._strategy)

        if self._enable_crash_override:
            raise NotImplementedError(
                "MOM-007: crash trigger awaits dedicated governed source adapter — see OI-34"
            )

        if self._orchestrator_hooks is not None and hasattr(
            self._orchestrator_hooks, "apply_crash_override"
        ):
            updated_intent = self._orchestrator_hooks.apply_crash_override(
                trade_intent=trade_intent,
                alpha_ir=alpha_ir,
                ctx=self._ctx,
                strategy=self._strategy,
            )
            if isinstance(updated_intent, TradeIntent):
                trade_intent = updated_intent

        if self._orchestrator_hooks is not None:
            if not execution_assumptions_path.exists():
                raise MissingExecutionAssumptionsError(
                    "execution_assumptions.json is required before the governed cost gate runs."
                )
            try:
                updated_intent = self._orchestrator_hooks.apply_cost_gate(
                    trade_intent=trade_intent,
                    alpha_ir=alpha_ir,
                    ctx=self._ctx,
                    strategy=self._strategy,
                    execution_assumptions_path=execution_assumptions_path,
                    bundle_dir=self._target_bundle_dir,
                )
                if isinstance(updated_intent, TradeIntent):
                    trade_intent = updated_intent
            except ConvergenceError:
                raise
            except CostGateRejection as exc:
                emit_gate_failure_report(
                    self._target_bundle_dir,
                    gate_id="cost_gate",
                    reason_code=ReasonCode.COST_GATE_REJECTED.value,
                    message=exc.message,
                    evidence={
                        "variant": exc.variant,
                        "run_id": exc.run_id,
                        "upstream_reason_code": exc.reason_code,
                    },
                )
                raise

        self.trade_intent = trade_intent
        self.alpha_ir = alpha_ir
        self.execution_assumptions_path = execution_assumptions_path
        return trade_intent

    def write_artifacts(
        self,
        store: BundleBacktestArtifactStore,
        *,
        feats: pd.DataFrame | object = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        if self.alpha_ir is None:
            raise TypeError("Momentum runtime adapter requires AlphaIR before artifact emission.")

        active_run_id = run_id or self._bound_run_id or str(uuid4())
        run_meta = RunMeta(run_id=active_run_id, bundle_dir=str(self._target_bundle_dir))
        store.put_json("signal_card.json", build_signal_card_payload(self.alpha_ir, run_meta))

        features = feats
        if not isinstance(features, pd.DataFrame) and self.trade_intent is not None:
            candidate = self.trade_intent.raw.get("features")
            if isinstance(candidate, pd.DataFrame):
                features = candidate

        n_trials = 1
        if self._run_registry is not None:
            n_trials = int(
                self._run_registry.record_trial(_trial_counter_family(self.alpha_ir.variant))
            )

        stat_payload = build_stat_validity_payload(
            alpha_ir=self.alpha_ir,
            run_meta=run_meta,
            returns=_coerce_returns(features),
            n_trials=n_trials,
            pbo_path_pairs=kwargs.get("pbo_path_pairs", self._pbo_path_pairs),
        )
        store.put_json("stat_validity_report.json", stat_payload)

        if (
            self.trade_intent is not None
            and self._splits_manifest_override is not None
            and str(self._splits_manifest_override.get("split_method", "none")) == "cpcv"
        ):
            if self.execution_assumptions_path is None:
                raise MissingExecutionAssumptionsError(
                    "execution_assumptions.json is required before CPCV path-score emission."
                )
            splits_manifest = json.loads(
                (self._target_bundle_dir / "splits_manifest.json").read_text(encoding="utf-8")
            )
            execution_assumptions = json.loads(
                self.execution_assumptions_path.read_text(encoding="utf-8")
            )
            cpcv_payload = build_cpcv_path_score_surface(
                variant=self.alpha_ir.variant,
                trade_intent=self.trade_intent,
                prices=self._ctx.prices,
                splits_manifest=splits_manifest,
                commission_bps=float(execution_assumptions["commission_bps"]),
                slippage_bps=float(execution_assumptions["slippage_bps"]),
                cost_model_id=str(execution_assumptions["cost_model_id"]),
            )
            store.put_json("cpcv_path_scores.json", cpcv_payload)


def run(
    ctx: StrategyContext,
    *,
    variant: str = "xsec",
    orchestrator_hooks: OrchestratorHooks | None = None,
    bundle_dir: Path | None = None,
    run_id: str | None = None,
    run_registry: RunRegistry | None = None,
    cas: LocalCAS | None = None,
    enable_crash_override: bool = False,
    strategy: MomentumStrategy | None = None,
    **params: Any,
) -> RunResult:
    runtime_params = dict(params)
    pbo_path_pairs = runtime_params.pop("pbo_path_pairs", None)
    splits_manifest_override = runtime_params.pop("splits_manifest_override", None)
    active_strategy = strategy or MomentumStrategy(variant=variant, **runtime_params)
    target_bundle_dir = Path(bundle_dir or ctx.cache_dir)
    managed_run_id: str | None = None
    if run_registry is not None and run_id is None:
        managed_run_id = run_registry.begin_run(
            metadata={
                "strategy": "momentum",
                "variant": str(active_strategy.params["variant"]),
                "bundle_dir": str(target_bundle_dir),
            }
        )
    active_run_id = run_id or managed_run_id or str(uuid4())
    pit_input, knowledge_dates, source_prices = _build_pit_input(ctx)
    runtime_strategy = _MomentumRuntimeAdapter(
        active_strategy,
        ctx=ctx,
        orchestrator_hooks=orchestrator_hooks,
        target_bundle_dir=target_bundle_dir,
        enable_crash_override=enable_crash_override,
        run_registry=run_registry,
        pbo_path_pairs=pbo_path_pairs,
        splits_manifest_override=splits_manifest_override,
    )

    try:
        pipeline_orchestrator.run(
            strategy_id="momentum",
            ctx=ctx,
            strategy_kwargs={},
            bundle_dir=target_bundle_dir,
            pit_input=pit_input,
            knowledge_dates=knowledge_dates,
            source_prices=source_prices,
            run_metadata={"pbo_path_pairs": pbo_path_pairs},
            strategy_instance=runtime_strategy,
            run_id=active_run_id,
            run_registry=run_registry,
            cas=cas,
            splits_manifest_override=splits_manifest_override,
        )
        trade_intent = runtime_strategy.trade_intent
        alpha_ir = runtime_strategy.alpha_ir
        execution_assumptions_path = runtime_strategy.execution_assumptions_path
        if trade_intent is None or alpha_ir is None or execution_assumptions_path is None:
            raise TypeError("Momentum entry.run did not receive canonical runtime outputs.")
        _finalize_managed_run(run_registry, managed_run_id, status=RunStatus.COMPLETE)

        return RunResult(
            trade_intent=trade_intent,
            alpha_ir=alpha_ir,
            bundle_dir=target_bundle_dir,
            artifacts={
                "execution_assumptions.json": execution_assumptions_path,
                "signal_card.json": target_bundle_dir / "signal_card.json",
                "stat_validity_report.json": target_bundle_dir / "stat_validity_report.json",
                **(
                    {"cpcv_path_scores.json": target_bundle_dir / "cpcv_path_scores.json"}
                    if (target_bundle_dir / "cpcv_path_scores.json").exists()
                    else {}
                ),
            },
        )
    except ConvergenceError:
        _finalize_managed_run(run_registry, managed_run_id, status=RunStatus.FAILED)
        raise
    except MaterializationError as exc:
        _finalize_managed_run(run_registry, managed_run_id, status=RunStatus.FAILED)
        if "converg" in str(exc).lower():
            raise ConvergenceError(str(exc)) from exc
        raise
    except Exception:
        _finalize_managed_run(run_registry, managed_run_id, status=RunStatus.FAILED)
        raise
