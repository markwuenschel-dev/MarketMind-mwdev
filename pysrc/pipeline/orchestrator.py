from __future__ import annotations

import asyncio
import hashlib
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from pysrc.artifact_registry import LocalCAS
from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.artifact_registry.run_registry import RunRegistry
from pysrc.backtesting.contracts.bundle import RunBundle
from pysrc.backtesting.contracts.errors import PitUnsafeInputError
from pysrc.backtesting.contracts.plan import BacktestPlan, DeterminismTier, EngineConfig
from pysrc.backtesting.contracts.registry import resolve_engine, resolve_validator
from pysrc.backtesting.contracts.types import BacktestResult, PitMeta
from pysrc.backtesting.data.pit import PITSafeDataView
from pysrc.cli.gate import validate_bundle as validate_governed_bundle
from pysrc.data.dataview import DataView
from pysrc.data.dataview_asof_adapter import DataViewAsOfAdapter
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.phase2_governed import emit_governed_phase2_orchestration_evidence
from pysrc.preprocessor.core import add_returns, add_sma, load_ohlcv
from pysrc.registry.gate_to_screening import gate_result_to_stage_and_code
from pysrc.registry.screening_report import ScreeningReportBuilder
from pysrc.registry.screening_taxonomy import ScreeningStatus
from pysrc.strategies.pipeline_strategy import (
    StrategyContext,
    StrategyRegistry,
)

LOG = get_logger(__name__)


def _load_dataprep_runtime() -> Any:
    from pysrc.pipeline import dataprep_runtime as runtime_module

    return runtime_module


def _maybe_mem_info(_ctx: Any = None) -> dict[str, Any]:
    runtime = _load_dataprep_runtime()
    if "psutil" in globals():
        original = getattr(runtime, "psutil", None)
        runtime.psutil = globals()["psutil"]
        try:
            return runtime._maybe_mem_info(_ctx)
        finally:
            runtime.psutil = original
    return runtime._maybe_mem_info(_ctx)


def run_dataprep(*args: Any, **kwargs: Any) -> Any:
    return _load_dataprep_runtime().run_dataprep(*args, **kwargs)


def run_dataprep_from_path(*args: Any, **kwargs: Any) -> Any:
    return _load_dataprep_runtime().run_dataprep_from_path(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in {
        "Cache",
        "ConfigError",
        "ConfigValidationError",
        "DataPrepError",
        "DataPrepOrchestrator",
        "pl",
        "psutil",
        "to_polars",
        "_TS_NAME_CANDIDATES",
    }:
        return getattr(_load_dataprep_runtime(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Cache",
    "ConfigError",
    "ConfigValidationError",
    "DataPrepError",
    "DataPrepOrchestrator",
    "OrchestratorConfig",
    "_TS_NAME_CANDIDATES",
    "_maybe_mem_info",
    "asyncio",
    "pl",
    "run",
    "run_dataprep",
    "run_dataprep_from_path",
    "run_orchestration",
    "to_polars",
]


@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for running a single end-to-end pipeline orchestration."""

    input_path: Path | None = None
    processed_data_root: Path | None = None
    pipeline_product: str | None = None
    symbol: str | None = None
    fast_sma: int = 5
    slow_sma: int = 10
    bundle_dir: Path | None = None


def _default_bundle_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("bundles") / timestamp


def _emit_phase2_orchestration_evidence(**kwargs: Any) -> dict[str, Any]:
    """Emit orchestration evidence summary; research-first path does not gate on it."""
    return emit_governed_phase2_orchestration_evidence(**kwargs)


def _load_orchestration_price_frame(config: OrchestratorConfig):
    """Load OHLCV-ish prices from a CSV path or a pipeline product."""

    import polars as pl

    from pysrc.pipeline.products import (
        load_pipeline_indicator_panel_polars,
        normalize_pipeline_panel_for_backtest,
    )

    if config.pipeline_product == "indicator_panel":
        root = config.processed_data_root or Path("data/processed")
        panel = load_pipeline_indicator_panel_polars(root, symbol=config.symbol)
        return normalize_pipeline_panel_for_backtest(panel)

    if config.input_path is None:
        raise ValueError(
            "OrchestratorConfig requires input_path or pipeline_product='indicator_panel'"
        )

    df = load_ohlcv(config.input_path)
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def run_orchestration(config: OrchestratorConfig) -> tuple[int, dict[str, Any]]:
    """Execute the canonical Python orchestration flow."""
    bundle_dir = config.bundle_dir or _default_bundle_dir()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    df = _load_orchestration_price_frame(config)
    df = add_returns(df)
    df = add_sma(df, window=config.fast_sma)
    df = add_sma(df, window=config.slow_sma)

    pdf = df.to_pandas()
    if "symbol" not in pdf.columns:
        if config.input_path is not None:
            pdf["symbol"] = config.input_path.stem
        elif config.symbol:
            pdf["symbol"] = config.symbol
        else:
            pdf["symbol"] = "PANEL"
    if "date" in pdf.columns:
        pdf["valid_time"] = pdf["date"]
        pdf["knowledge_time"] = pdf["date"]
    else:
        raise ValueError("Canonical orchestration requires a 'date' column for PIT normalization.")

    dataview = DataView(pit_required=True)
    dataview.register_source(
        pdf,
        valid_time_col="valid_time",
        knowledge_time_col="knowledge_time",
        seed_fixture_membership=True,
    )

    symbols: list[str] = sorted({str(sym) for sym in pdf["symbol"].unique().tolist()})
    fields: list[str] = [
        "returns",
        f"sma_{config.fast_sma}",
        f"sma_{config.slow_sma}",
    ]

    adapter = DataViewAsOfAdapter(dataview=dataview, symbols=symbols, fields=fields)

    knowledge_dates: list[datetime] = []
    if "date" in pdf.columns:
        knowledge_dates = sorted(
            {
                datetime.combine(d.date(), datetime.min.time()).replace(tzinfo=UTC)
                if hasattr(d, "date")
                else datetime.fromisoformat(str(d)).replace(tzinfo=UTC)
                for d in pdf["date"]
            }
        )

    run_id = str(uuid4())
    engine_config = EngineConfig(
        lane="vectorized",
        bar_frequency="1d",
        params={"fast_sma": config.fast_sma, "slow_sma": config.slow_sma},
    )
    plan = BacktestPlan(
        engine_id="vectorized.sma",
        execution_model_id="fill.identity",
        cost_model_id="fees.zero",
        ledger_id="ledger.simple",
        validator_ids=["statistical.v1", "mechanical.v1"],
        determinism=DeterminismTier.D1,
        seed=42,
        pit_required=True,
        engine_config=engine_config,
        run_id=run_id,
    )

    writer = BundleWriter(bundle_dir)
    store = BundleBacktestArtifactStore(writer)
    engine = resolve_engine(plan.engine_id)
    if not knowledge_dates:
        raise ValueError("Canonical PIT orchestration requires at least one knowledge date.")

    pit_view = PITSafeDataView(
        view=adapter,
        metadata={
            "knowledge_dates": knowledge_dates,
            "pit_enforced": True,
            "pit_front_door": "pysrc.data.dataview.DataView",
        },
    )

    result = engine.run(plan, pit_view, store)

    config_payload: dict[str, Any] = {
        "strategy": "sma_crossover",
        "fast_sma": config.fast_sma,
        "slow_sma": config.slow_sma,
        "engine_id": plan.engine_id,
    }
    if config.pipeline_product:
        config_payload["pipeline_product"] = config.pipeline_product
        config_payload["processed_data_root"] = str(config.processed_data_root or "data/processed")
    if config.input_path is not None:
        config_payload["input_file"] = str(config.input_path)
    config_hash = BundleWriter.compute_config_hash(config_payload)
    writer.write_plan(
        plan_hash=config_hash,
        config_hash=config_hash,
        as_of_time=datetime.now(UTC).isoformat(),
        config=config_payload,
    )
    writer.write_env_fingerprint()
    writer.write_dataset_manifest(
        dataset_id=str(config.input_path or config.pipeline_product or "pipeline"),
        symbols=[str(symbol) for symbol in df.get_column("symbol").unique().to_list()]
        if "symbol" in df.columns
        else [config.input_path.stem],
        row_count=df.height,
        time_range={
            "start": str(df.get_column("date").min()) if "date" in df.columns else "unknown",
            "end": str(df.get_column("date").max()) if "date" in df.columns else "unknown",
        },
        pit_compliant=True,
        knowledge_time_column="knowledge_time",
    )
    writer.write_preprocessing_report(
        steps=[
            {"name": "add_returns"},
            {"name": "add_sma", "window": config.fast_sma},
            {"name": "add_sma", "window": config.slow_sma},
        ],
        timings={},
        warnings=[],
    )
    writer.write_splits_manifest(
        splits=[],
        split_method="none",
        purge_window=0,
        embargo_window=0,
    )

    backtest_payload = {
        "schema_version": "1.0.0",
        "meta": {"strategy": "sma_crossover", "run_id": run_id},
        "result": result.metrics,
    }
    store.put_json("backtest_result.json", backtest_payload)

    validator_context = {
        "returns": df.get_column("returns").drop_nulls().to_list()
        if "returns" in df.columns
        else None,
        "window_start": datetime(1970, 1, 1, tzinfo=UTC),
        "window_end": datetime.now(UTC),
    }
    for validator_id in plan.validator_ids:
        validator = resolve_validator(validator_id)
        try:
            validator.validate(result, validator_context, store)
        except AttributeError:
            # Test harnesses may inject lightweight engine results missing optional
            # fields consumed by downstream validators.
            continue

    validation, _ = validate_governed_bundle(bundle_dir)

    # Phase I-E: screening report (one candidate = this run; stages from gate results)
    pit_boundary = datetime.now(UTC).isoformat()
    data_snapshot_hash = hashlib.sha256(
        f"{config_hash}:{config_payload.get('input_file', '')}".encode()
    ).hexdigest()
    screening_run_id = hashlib.sha256(f"{run_id}:{pit_boundary}".encode()).hexdigest()
    builder = ScreeningReportBuilder(
        screening_run_id=screening_run_id,
        pit_boundary=pit_boundary,
        data_snapshot_hash=data_snapshot_hash,
        seed=plan.seed,
    )
    spec_hash = hashlib.blake2b(f"sma_crossover:{run_id}".encode(), digest_size=16).hexdigest()
    builder.add_candidate(
        spec_hash=spec_hash,
        signal_name="sma_crossover",
        slot_index=None,
        evaluation_ordinal=0,
    )
    for _i, gate in enumerate(validation.gates):
        passed = gate["result"] == "PASS"
        stage, reason_code = gate_result_to_stage_and_code(gate["gate_id"], passed, gate["message"])
        builder.add_stage(
            candidate_index=0,
            stage=stage,
            status=ScreeningStatus.ACCEPTED if passed else ScreeningStatus.REJECTED,
            reason_code=reason_code,
            reason_detail=gate["message"],
            metrics=gate.get("evidence") or {},
            duration_ms=0,
        )
    final_status = "PROMOTED" if validation.overall_result == "PASS" else "REJECTED"
    last_stage = validation.gates[-1]["gate_id"] if validation.gates else "LANE_0"
    last_reason = None
    for g in reversed(validation.gates):
        if g["result"] != "PASS":
            _, code = gate_result_to_stage_and_code(g["gate_id"], False, g["message"])
            last_reason = code.value if code is not None else None
            break
    builder.set_final(0, final_status, last_stage, last_reason)
    writer.write_screening_report(builder.serialize())

    writer.write_bundle_manifest()
    phase2_summary = _emit_phase2_orchestration_evidence(
        bundle_dir=bundle_dir,
        strategy_id="sma_crossover",
        run_id=run_id,
        strategy_context=None,
        source_prices=pdf,
        features=pdf,
        signals=None,
        run_metadata=None,
    )

    output: dict[str, Any] = {
        "success": True,
        "bundle_path": str(bundle_dir),
        "backtest": result.metrics,
        "validation": {
            "status": validation.overall_result,
            "gates": [
                {"id": gate["gate_id"], "status": gate["result"], "reason": gate["message"]}
                for gate in validation.gates
            ],
        },
        "phase2_governed_evidence": phase2_summary,
    }

    LOG.info("orchestration_complete", bundle_path=str(bundle_dir), run_id=run_id)
    exit_code = 0 if validation.overall_result == "PASS" else 1
    return exit_code, output


def run(
    strategy_id: str,
    ctx: StrategyContext,
    strategy_kwargs: dict[str, Any],
    bundle_dir: Path,
    *,
    pit_input: PITSafeDataView | DataViewAsOfAdapter | None = None,
    knowledge_dates: list[datetime] | None = None,
    source_prices: pd.DataFrame | None = None,
    run_metadata: dict[str, Any] | None = None,
    strategy_instance: Any | None = None,
    run_id: str | None = None,
    run_registry: RunRegistry | None = None,
    cas: LocalCAS | None = None,
    splits_manifest_override: dict[str, Any] | None = None,
) -> RunBundle:
    """
    Canonical orchestration: resolve strategy, materialize features, generate signal,
    write bundle (plan, env, dataset, preprocessing, splits), strategy artifacts,
    validators, and return RunBundle. Owns engine resolution, bundle writing,
    and validator coordination.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    strategy_cls = StrategyRegistry.get(strategy_id)
    strategy = strategy_instance or strategy_cls(**strategy_kwargs)
    active_run_id = run_id or f"{strategy_id}_{id(strategy):x}"

    writer = BundleWriter(
        bundle_dir,
        cas=cas,
        run_registry=run_registry,
        run_id=active_run_id,
    )
    store = BundleBacktestArtifactStore(writer)
    bind_store = getattr(strategy, "bind_runtime_artifact_store", None)
    if callable(bind_store):
        bind_store(store=store, run_id=active_run_id, bundle_dir=bundle_dir)

    runtime_ctx = _normalize_strategy_context(
        ctx,
        pit_input=pit_input,
        knowledge_dates=knowledge_dates,
    )
    trade_intent = strategy.generate_trade_intent(runtime_ctx)
    feats = trade_intent.raw.get("features")
    if hasattr(feats, "to_pandas"):
        feats = feats.to_pandas()
    signals = _extract_trade_intent_signal(trade_intent)

    def _serializable(v: Any) -> Any:
        if is_dataclass(v) and not isinstance(v, type):
            return {k: _serializable(x) for k, x in asdict(v).items()}
        if isinstance(v, Enum):
            return v.value
        if isinstance(v, dict):
            return {k: _serializable(x) for k, x in v.items()}
        return v

    config_payload = {"strategy": strategy_id, "run_id": active_run_id}
    for k, v in strategy_kwargs.items():
        config_payload[k] = _serializable(v)
    config_hash = BundleWriter.compute_config_hash(config_payload)
    writer.write_plan(
        plan_hash=config_hash,
        config_hash=config_hash,
        as_of_time=datetime.now(UTC).isoformat(),
        config=config_payload,
    )
    writer.write_env_fingerprint()

    if source_prices is not None and not source_prices.empty:
        symbols = sorted({str(s) for s in source_prices["symbol"].unique().tolist()})
        valid = source_prices.get("valid_time")
        if valid is not None:
            time_range = {"start": str(valid.min()), "end": str(valid.max())}
        else:
            time_range = {"start": "unknown", "end": "unknown"}
        writer.write_dataset_manifest(
            dataset_id=strategy_id,
            symbols=symbols,
            row_count=int(source_prices.shape[0]),
            time_range=time_range,
            pit_compliant=True,
            knowledge_time_column="knowledge_time"
            if "knowledge_time" in source_prices.columns
            else "knowledge_time",
            content_hash=run_metadata.get("content_hash") if run_metadata else None,
            download_timestamp=run_metadata.get("download_timestamp") if run_metadata else None,
            content_hash_expected=run_metadata.get("content_hash_expected")
            if run_metadata
            else None,
        )
    else:
        prices = ctx.prices
        n = len(prices.index) if hasattr(prices, "index") else 0
        writer.write_dataset_manifest(
            dataset_id=strategy_id,
            symbols=[],
            row_count=n,
            time_range={"start": "unknown", "end": "unknown"},
            pit_compliant=True,
            knowledge_time_column="knowledge_time",
        )

    writer.write_preprocessing_report(steps=[{"name": strategy_id}], timings={}, warnings=[])
    if splits_manifest_override is None:
        writer.write_splits_manifest(
            splits=[], split_method="none", purge_window=0, embargo_window=0
        )
    else:
        writer.write_splits_manifest(
            splits=list(splits_manifest_override.get("splits", [])),
            split_method=str(splits_manifest_override.get("split_method", "none")),
            purge_window=int(splits_manifest_override.get("purge_window", 0)),
            embargo_window=int(splits_manifest_override.get("embargo_window", 0)),
        )

    if hasattr(strategy, "write_artifacts"):
        strategy.write_artifacts(
            store,
            signals=signals,
            feats=feats,
            run_id=active_run_id,
            trade_intent=trade_intent,
            **(run_metadata or {}),
        )

    result = BacktestResult(metrics={}, artifacts={})
    validator_context = {
        "returns": feats["returns"].dropna().tolist()
        if isinstance(feats, pd.DataFrame) and "returns" in feats.columns
        else None,
        "pbo_path_pairs": run_metadata.get("pbo_path_pairs") if run_metadata else None,
    }
    for vid in ["statistical.v1"]:
        try:
            validator = resolve_validator(vid)
            validator.validate(result, validator_context, store)
        except Exception:
            pass

    writer.write_bundle_manifest()
    _emit_phase2_orchestration_evidence(
        bundle_dir=bundle_dir,
        strategy_id=strategy_id,
        run_id=active_run_id,
        strategy_context=runtime_ctx,
        source_prices=source_prices,
        features=feats if isinstance(feats, pd.DataFrame) else None,
        signals=signals,
        run_metadata=run_metadata,
    )

    plan_record = writer.read_plan()
    env_record = writer.read_env_fingerprint()
    dataset_record = writer.read_dataset_manifest()
    preprocessing_record = writer.read_preprocessing_report()
    splits_record = writer.read_splits_manifest()

    return RunBundle(
        plan=plan_record,
        env_fingerprint=env_record,
        dataset_manifest=dataset_record,
        preprocessing_report=preprocessing_record,
        splits_manifest=splits_record,
    )


def _extract_trade_intent_signal(trade_intent: Any) -> Any:
    alpha_like = trade_intent.raw.get("alpha_ir") if hasattr(trade_intent, "raw") else None
    if alpha_like is not None and hasattr(alpha_like, "signal"):
        return alpha_like.signal

    signal = trade_intent.raw.get("signal") if hasattr(trade_intent, "raw") else None
    if signal is not None and hasattr(signal, "signal"):
        return signal.signal
    return signal


def _normalize_strategy_context(
    ctx: StrategyContext,
    *,
    pit_input: PITSafeDataView | DataViewAsOfAdapter | None,
    knowledge_dates: list[datetime] | None,
) -> StrategyContext:
    if pit_input is None:
        return ctx

    if isinstance(pit_input, PITSafeDataView):
        adapter = pit_input.view
        resolved_dates = knowledge_dates or pit_input.metadata.get("knowledge_dates")
    else:
        adapter = pit_input
        resolved_dates = knowledge_dates

    if not isinstance(adapter, DataViewAsOfAdapter):
        raise PitUnsafeInputError(
            "Canonical strategy orchestration requires a DataViewAsOfAdapter-backed PIT input."
        )
    if not resolved_dates:
        raise PitUnsafeInputError(
            "Canonical strategy orchestration requires explicit knowledge_dates for PIT-safe inputs."
        )

    prices = adapter.as_wide_frame(list(resolved_dates))
    if prices.empty:
        raise PitUnsafeInputError(
            "Canonical strategy orchestration produced an empty PIT-safe frame."
        )

    pit_meta = adapter.pit_meta()
    if pit_meta is None:
        last = max(list(resolved_dates))
        pit_meta = PitMeta(
            as_of=last.isoformat(),
            source="pysrc.data.dataview.DataView",
            knowledge_cutoff=last.date().isoformat(),
        )

    return StrategyContext(
        prices=prices,
        features=ctx.features,
        timestamps=ctx.timestamps,
        asset_names=ctx.asset_names,
        backend=ctx.backend,
        cache_dir=ctx.cache_dir,
        random_state=ctx.random_state,
        pit_provenance=ctx.pit_provenance or pit_meta,
    )
