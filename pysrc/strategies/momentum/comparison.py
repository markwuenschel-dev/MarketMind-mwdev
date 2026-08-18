from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pysrc.artifact_registry import LocalCAS
from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.artifact_registry.run_registry import RunRegistry
from pysrc.backtesting.validation.statistical.cpcv import CPCVConfig, CPCVSplitter
from pysrc.backtesting.validation.statistical.pbo import compute_pbo
from pysrc.backtesting.validation.statistical.pbo_bridge import (
    CANONICAL_PBO_MODE,
    build_pbo_path_pairs,
)
from pysrc.strategies.momentum import entry as momentum_entry
from pysrc.strategies.momentum.artifacts.cpcv_path_scores import (
    compute_payload_hash,
    normalize_close_prices,
)
from pysrc.strategies.momentum.entry import RunResult
from pysrc.strategies.momentum.validation.production_v1 import PRODUCTION_V1_PROFILE
from pysrc.strategies.pipeline_strategy import StrategyContext

_DATE_COLUMNS = ("date", "valid_time", "datetime", "timestamp", "as_of")
_ASSET_COLUMNS = ("asset", "symbol", "ticker", "sid", "instrument")
_VARIANTS = ("xsec", "tsmom", "dual")


@dataclass(frozen=True)
class ComparisonRunResult:
    bundle_dir: Path
    comparison_run_id: str
    variant_runs: dict[str, RunResult]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object at {path}")
    return dict(payload)


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _normalize_prices_for_strategy_input(prices: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(prices, pd.DataFrame) and "close" in prices.columns:
        date_col = _resolve_column(prices, _DATE_COLUMNS)
        asset_col = _resolve_column(prices, _ASSET_COLUMNS)
        if date_col is not None and asset_col is not None:
            normalized = pd.DataFrame(
                {
                    "date": pd.to_datetime(prices[date_col]),
                    "asset": prices[asset_col].astype(str),
                    "close": prices["close"].astype(float),
                }
            )
            return normalized.sort_values(["asset", "date"], kind="stable").reset_index(drop=True)

    wide = normalize_close_prices(prices)
    stacked = wide.stack(dropna=False).rename("close").reset_index()
    stacked.columns = ["date", "asset", "close"]
    stacked["date"] = pd.to_datetime(stacked["date"])
    stacked["asset"] = stacked["asset"].astype(str)
    return stacked.sort_values(["asset", "date"], kind="stable").reset_index(drop=True)


def _verify_shared_artifact_hash(path: Path, *, expected_hash: str, label: str) -> str:
    observed_hash = compute_payload_hash(_load_json(path))
    if observed_hash != expected_hash:
        raise ValueError(
            f"{label} hash mismatch: expected {expected_hash}, observed {observed_hash}"
        )
    return observed_hash


def _build_shared_cpcv_surface(
    prices: pd.DataFrame | pd.Series,
    *,
    purge_window: int = 0,
    embargo_window: int = 0,
    min_train_size: int = 1,
) -> dict[str, Any]:
    normalized_prices = normalize_close_prices(prices)
    cfg = CPCVConfig(
        n_splits=PRODUCTION_V1_PROFILE.cpcv.n_splits,
        n_test_splits=PRODUCTION_V1_PROFILE.cpcv.n_test_paths,
        purge_periods=purge_window,
        embargo_periods=embargo_window,
        min_train_size=min_train_size,
    )
    splitter = CPCVSplitter(cfg)

    splits: list[dict[str, Any]] = []
    for split in splitter.split(normalized_prices):
        splits.append(
            {
                "path_id": f"path-{split.split_index:03d}",
                "split_index": int(split.split_index),
                "train_indices": [int(value) for value in split.train_indices.tolist()],
                "test_indices": [int(value) for value in split.test_indices.tolist()],
                "test_group_ids": [int(value) for value in split.test_group_ids],
                "n_train": int(split.n_train),
                "n_test": int(split.n_test),
            }
        )

    payload = {
        "split_method": "cpcv",
        "purge_window": int(purge_window),
        "embargo_window": int(embargo_window),
        "splits": splits,
    }
    artifact_payload = {"schema_version": BundleWriter.SCHEMA_VERSION, **payload}
    return {
        "payload": payload,
        "artifact_payload": artifact_payload,
        "hash": compute_payload_hash(artifact_payload),
        "config": {
            "n_splits": int(cfg.n_splits),
            "n_test_splits": int(cfg.n_test_splits),
            "purge_window": int(cfg.purge_periods),
            "embargo_window": int(cfg.embargo_periods),
            "min_train_size": int(cfg.min_train_size),
            "n_paths": int(len(splits)),
        },
    }


def _shared_cost_payload(
    *, commission_bps: float, slippage_bps: float, cost_model_id: str
) -> dict[str, Any]:
    return {
        "commission_bps": float(commission_bps),
        "slippage_bps": float(slippage_bps),
        "cost_model_id": str(cost_model_id),
    }


def _observed_cost_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return _shared_cost_payload(
        commission_bps=float(payload["commission_bps"]),
        slippage_bps=float(payload["slippage_bps"]),
        cost_model_id=str(payload["cost_model_id"]),
    )


def _verify_shared_cost_identity(paths: list[Path]) -> tuple[dict[str, Any], str]:
    payloads = [_observed_cost_payload(path) for path in paths]
    shared = payloads[0]
    for payload in payloads[1:]:
        if payload != shared:
            raise ValueError("shared cost model mismatch across child runs")
    return shared, compute_payload_hash(shared)


def _build_child_run_id(
    *,
    variant: str,
    comparison_config_hash: str,
    split_surface_hash: str,
    shared_cost_hash: str,
) -> str:
    child_hash = compute_payload_hash(
        {
            "variant": variant,
            "comparison_config_hash": comparison_config_hash,
            "split_surface_hash": split_surface_hash,
            "shared_cost_hash": shared_cost_hash,
        }
    )
    return f"momentum_{variant}_{child_hash.split(':', 1)[1][:16]}"


def _build_comparison_run_id(
    *,
    comparison_config_hash: str,
    split_surface_hash: str,
    shared_cost_hash: str,
    child_run_ids: Mapping[str, str],
) -> str:
    payload = {
        "comparison_config_hash": comparison_config_hash,
        "split_surface_hash": split_surface_hash,
        "shared_cost_hash": shared_cost_hash,
        "child_run_ids": {key: child_run_ids[key] for key in sorted(child_run_ids)},
    }
    return f"momentum_cmp:{compute_payload_hash(payload).split(':', 1)[1]}"


def _build_dataset_manifest_inputs(
    source_prices: pd.DataFrame | pd.Series,
    normalized_prices: pd.DataFrame,
) -> dict[str, Any]:
    if isinstance(source_prices, pd.DataFrame):
        date_col = _resolve_column(source_prices, _DATE_COLUMNS)
        asset_col = _resolve_column(source_prices, _ASSET_COLUMNS)
        if date_col is not None and asset_col is not None:
            return {
                "dataset_id": "momentum_variant_comparison",
                "symbols": sorted(
                    {str(value) for value in source_prices[asset_col].dropna().unique().tolist()}
                ),
                "row_count": int(source_prices.shape[0]),
                "time_range": {
                    "start": str(pd.to_datetime(source_prices[date_col]).min()),
                    "end": str(pd.to_datetime(source_prices[date_col]).max()),
                },
            }

    row_count = (
        int(source_prices.shape[0])
        if isinstance(source_prices, pd.DataFrame)
        else int(len(source_prices))
    )
    return {
        "dataset_id": "momentum_variant_comparison",
        "symbols": [str(symbol) for symbol in normalized_prices.columns],
        "row_count": row_count,
        "time_range": {
            "start": str(normalized_prices.index.min()),
            "end": str(normalized_prices.index.max()),
        },
    }


def _child_strategy_context(
    source_prices: pd.DataFrame, *, bundle_dir: Path, template: StrategyContext
) -> StrategyContext:
    return StrategyContext(
        prices=source_prices.copy(),
        backend=template.backend,
        cache_dir=bundle_dir,
        pit_provenance=template.pit_provenance,
    )


def _require_child_cpcv_path_scores(path: Path, *, variant: str) -> dict[str, Any]:
    payload = _load_json(path)
    evaluations = payload.get("evaluations")
    summary = payload.get("summary")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError(
            f"child cpcv_path_scores.json must contain a non-empty evaluations list for variant '{variant}'"
        )
    if not isinstance(summary, Mapping):
        raise ValueError(f"child cpcv_path_scores.json summary missing for variant '{variant}'")
    if str(payload.get("variant")) != variant:
        raise ValueError(f"child cpcv_path_scores.json variant mismatch for '{variant}'")
    return payload


def _build_comparison_stat_validity_payload(
    *,
    comparison_run_id: str,
    canonical_pbo: Mapping[str, Any],
    split_surface_hash: str,
    shared_cost_hash: str,
    variant_surfaces: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "strategy": "momentum_variant_comparison",
        "comparison_run_id": comparison_run_id,
        "pbo": dict(canonical_pbo),
        "shared_pbo_hash": compute_payload_hash(dict(canonical_pbo)),
        "split_surface_hash": split_surface_hash,
        "shared_cost_hash": shared_cost_hash,
        "variants": sorted(variant_surfaces),
        "source_artifacts": {
            variant: str(Path("variants") / variant / "cpcv_path_scores.json")
            for variant in sorted(variant_surfaces)
        },
    }


def run_variant_comparison(
    ctx: StrategyContext,
    *,
    bundle_dir: Path | None = None,
    run_registry: RunRegistry | None = None,
    cas: LocalCAS | None = None,
    commission_bps: float = 5.0,
    slippage_bps: float = 1.0,
    cost_model_id: str = "momentum.phase_i.default",
    purge_window: int = 0,
    embargo_window: int = 0,
    min_train_size: int = 1,
) -> ComparisonRunResult:
    target_bundle_dir = Path(bundle_dir or ctx.cache_dir)
    target_bundle_dir.mkdir(parents=True, exist_ok=True)

    prices_wide = normalize_close_prices(ctx.prices)
    strategy_prices = _normalize_prices_for_strategy_input(ctx.prices)
    surface = _build_shared_cpcv_surface(
        prices_wide,
        purge_window=purge_window,
        embargo_window=embargo_window,
        min_train_size=min_train_size,
    )
    shared_cost_model = _shared_cost_payload(
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        cost_model_id=cost_model_id,
    )
    shared_cost_hash = compute_payload_hash(shared_cost_model)
    comparison_base_config = {
        "strategy": "momentum_variant_comparison",
        "variants": list(_VARIANTS),
        "shared_cpcv": surface["config"],
        "split_surface_hash": surface["hash"],
        "shared_cost_hash": shared_cost_hash,
    }
    comparison_config_hash = BundleWriter.compute_config_hash(comparison_base_config)

    variant_runs: dict[str, RunResult] = {}
    child_run_ids: dict[str, str] = {}
    for variant in _VARIANTS:
        child_bundle_dir = target_bundle_dir / "variants" / variant
        child_ctx = _child_strategy_context(
            strategy_prices, bundle_dir=child_bundle_dir, template=ctx
        )
        child_run_id = _build_child_run_id(
            variant=variant,
            comparison_config_hash=comparison_config_hash,
            split_surface_hash=surface["hash"],
            shared_cost_hash=shared_cost_hash,
        )
        child_run_ids[variant] = child_run_id
        variant_runs[variant] = momentum_entry.run(
            child_ctx,
            variant=variant,
            bundle_dir=child_bundle_dir,
            run_id=child_run_id,
            run_registry=run_registry,
            cas=cas,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            cost_model_id=cost_model_id,
            splits_manifest_override=surface["payload"],
        )

    observed_cost_model, observed_cost_hash = _verify_shared_cost_identity(
        [variant_runs[variant].bundle_dir / "execution_assumptions.json" for variant in _VARIANTS]
    )
    if observed_cost_hash != shared_cost_hash:
        raise ValueError("shared cost model hash mismatch against child artifacts")

    variant_surfaces: dict[str, dict[str, Any]] = {}
    child_stat_validity: dict[str, dict[str, Any]] = {}
    all_evaluations: list[dict[str, Any]] = []
    for variant in _VARIANTS:
        child_bundle_dir = variant_runs[variant].bundle_dir
        _verify_shared_artifact_hash(
            child_bundle_dir / "splits_manifest.json",
            expected_hash=surface["hash"],
            label="split surface",
        )
        cpcv_payload = _require_child_cpcv_path_scores(
            child_bundle_dir / "cpcv_path_scores.json",
            variant=variant,
        )
        if str(cpcv_payload["split_surface_hash"]) != surface["hash"]:
            raise ValueError(
                f"split surface hash mismatch in child cpcv_path_scores.json for variant '{variant}'"
            )
        if str(cpcv_payload["shared_cost_hash"]) != observed_cost_hash:
            raise ValueError(
                f"shared cost model hash mismatch in child cpcv_path_scores.json for variant '{variant}'"
            )
        variant_surfaces[variant] = cpcv_payload
        child_stat_validity[variant] = _load_json(child_bundle_dir / "stat_validity_report.json")
        all_evaluations.extend(cpcv_payload["evaluations"])

    shared_pbo_path_pairs = build_pbo_path_pairs(all_evaluations)
    canonical_pbo = json.loads(
        json.dumps(
            compute_pbo(shared_pbo_path_pairs, mode=CANONICAL_PBO_MODE),
            default=_json_default,
        )
    )

    comparison_run_id = _build_comparison_run_id(
        comparison_config_hash=comparison_config_hash,
        split_surface_hash=surface["hash"],
        shared_cost_hash=observed_cost_hash,
        child_run_ids=child_run_ids,
    )
    comparison_stat_validity = _build_comparison_stat_validity_payload(
        comparison_run_id=comparison_run_id,
        canonical_pbo=canonical_pbo,
        split_surface_hash=surface["hash"],
        shared_cost_hash=observed_cost_hash,
        variant_surfaces=variant_surfaces,
    )

    variants_report: dict[str, dict[str, Any]] = {}
    for variant in _VARIANTS:
        stat_validity = child_stat_validity[variant]
        cpcv_payload = variant_surfaces[variant]
        summary = dict(cpcv_payload["summary"])
        variants_report[variant] = {
            "run_id": child_run_ids[variant],
            "bundle_dir": str(Path("variants") / variant),
            "stat_validity_report_path": str(
                Path("variants") / variant / "stat_validity_report.json"
            ),
            "execution_assumptions_path": str(
                Path("variants") / variant / "execution_assumptions.json"
            ),
            "cpcv_path_scores_path": str(Path("variants") / variant / "cpcv_path_scores.json"),
            "sharpe_ratio": float(stat_validity.get("sharpe_ratio", 0.0)),
            "dsr": stat_validity.get("dsr"),
            "child_stat_validity_pbo": stat_validity.get("pbo"),
            **summary,
        }

    ranking = [
        {
            "variant": variant,
            "mean_out_of_sample_net_sharpe": variants_report[variant][
                "mean_out_of_sample_net_sharpe"
            ],
        }
        for variant in sorted(
            _VARIANTS,
            key=lambda name: float(variants_report[name]["mean_out_of_sample_net_sharpe"]),
            reverse=True,
        )
    ]

    report = {
        "schema_version": "1.0.0",
        "strategy": "momentum_variant_comparison",
        "comparison_run_id": comparison_run_id,
        "comparison_stat_validity_path": "comparison_stat_validity.json",
        "shared_cpcv": surface["config"],
        "split_surface_hash": surface["hash"],
        "shared_cost_model": observed_cost_model,
        "shared_cost_hash": observed_cost_hash,
        "shared_pbo": canonical_pbo,
        "shared_pbo_hash": compute_payload_hash(canonical_pbo),
        "shared_path_score_surface_hash": compute_payload_hash({"evaluations": all_evaluations}),
        "cost_identity_valid": True,
        "split_identity_valid": True,
        "variants": variants_report,
        "ranking": ranking,
    }

    writer = BundleWriter(target_bundle_dir, cas=cas)
    store = BundleBacktestArtifactStore(writer)
    plan_config = {**comparison_base_config, "comparison_run_id": comparison_run_id}
    plan_hash = BundleWriter.compute_config_hash(plan_config)
    writer.write_plan(
        plan_hash=plan_hash,
        config_hash=plan_hash,
        as_of_time=pd.Timestamp.now(tz="UTC").isoformat(),
        config=plan_config,
    )
    writer.write_env_fingerprint()
    dataset_manifest = _build_dataset_manifest_inputs(ctx.prices, prices_wide)
    writer.write_dataset_manifest(
        dataset_id=str(dataset_manifest["dataset_id"]),
        symbols=list(dataset_manifest["symbols"]),
        row_count=int(dataset_manifest["row_count"]),
        time_range=dict(dataset_manifest["time_range"]),
        pit_compliant=ctx.pit_provenance is not None,
        knowledge_time_column="knowledge_time",
    )
    writer.write_preprocessing_report(
        steps=[
            {"name": "build_shared_cpcv_surface"},
            {"name": "read_child_cpcv_path_scores"},
            {"name": "momentum_variant_comparison"},
        ],
        timings={},
        warnings=[],
    )
    writer.write_splits_manifest(
        splits=list(surface["payload"]["splits"]),
        split_method=str(surface["payload"]["split_method"]),
        purge_window=int(surface["payload"]["purge_window"]),
        embargo_window=int(surface["payload"]["embargo_window"]),
    )
    store.put_json("comparison_stat_validity.json", comparison_stat_validity)
    store.put_json("momentum_variant_comparison_report.json", report)
    writer.write_bundle_manifest()

    return ComparisonRunResult(
        bundle_dir=target_bundle_dir,
        comparison_run_id=comparison_run_id,
        variant_runs=variant_runs,
    )


__all__ = [
    "ComparisonRunResult",
    "run_variant_comparison",
    "_build_shared_cpcv_surface",
    "_verify_shared_artifact_hash",
]
