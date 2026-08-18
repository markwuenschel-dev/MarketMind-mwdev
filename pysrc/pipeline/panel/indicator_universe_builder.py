"""Discover, classify, and persist the full P2-PANEL indicator universe from artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.panel.feature_grain_audit import (
    DUPLICATE_KEY_SAMPLE_FILENAME,
    ROW_GRAIN,
    TICKER_DATE_INTERVAL_KEYS,
    SourceGrainReport,
    audit_panel_grain,
    build_duplicate_key_sample,
    normalize_panel_keys,
    validate_source_grain,
)
from pysrc.pipeline.panel.panel_feature_registry import (
    FeatureColumnRecord,
    FeatureExclusionReason,
    classify_column,
)
from pysrc.pipeline.panel.runtime import (
    resolve_supervision_path,
    supervision_artifact_exists,
)
from pysrc.pipeline.products import resolve_pipeline_indicator_features_path
from pysrc.pipeline.stages.preprocessing.indicators.fold_dedup import (
    collapse_slim_indicator_fold_duplicates,
)

PanelGrain = Literal["ticker_date_interval", "date_level"]

PANEL_TARGET_OPTIONS: tuple[str, ...] = (
    "forward_return",
    "cost_adjusted_forward_return",
    "cross_sectional_forward_return_rank",
    "positive_forward_return_label",
    "portfolio_adjusted_advantage",
    "abstain_or_trade_label",
    "allocation_score",
)

_DEFAULT_OUTPUT_DIR = Path("artifacts/phase_ii/p2_panel_model")
_MERGE_KEYS: tuple[str, ...] = TICKER_DATE_INTERVAL_KEYS
_W3B_SOURCE_ID = "w3_b_pandas_ta_indicators"
_CANONICAL_SOURCE_ID = "pipeline_full_indicator_feature_panel"
_W4A_SOURCE_ID = "w4_a_router_supervision"
_SYNTHETIC_SOURCE_ID = "synthetic_panel"
_W3B_KEY_COLUMNS: frozenset[str] = frozenset({"date", "instrument", "interval"})
_W4A_EXCLUSION_INCOMPATIBLE_ROUTER_GRAIN = "incompatible_router_grain"


@dataclass(frozen=True, slots=True)
class ColumnSource:
    source_id: str
    path: str | None
    frame: pd.DataFrame
    feature_family: str
    interval: str = "daily"
    load_report: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class DiscoveredSources:
    panel_sources: tuple[ColumnSource, ...]
    router_sources: tuple[ColumnSource, ...]


@dataclass(frozen=True, slots=True)
class PanelMergePlan:
    primary: ColumnSource
    optional_context: tuple[ColumnSource, ...]
    excluded_from_panel_merge: tuple[dict[str, object], ...]
    source_grain_reports: dict[str, SourceGrainReport]


@dataclass(frozen=True, slots=True)
class MergeSourcesResult:
    frame: pd.DataFrame
    provenance: dict[str, object]
    source_column_map: dict[str, tuple[str, ...]]
    collisions: tuple[dict[str, object], ...]
    w3_b_indicator_loaded: bool
    primary_feature_source: str
    merged_sources: tuple[str, ...]
    excluded_from_panel_merge: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class IndicatorUniverseResult:
    panel_frame: pd.DataFrame
    records: tuple[FeatureColumnRecord, ...]
    eligible_features: tuple[str, ...]
    grain_audit: Any
    sources: dict[str, object]
    source_column_map: dict[str, tuple[str, ...]]
    collisions: tuple[dict[str, object], ...]
    uses_full_discovered_feature_universe: bool
    uses_full_indicator_universe: bool
    sources_detail: dict[str, object]
    primary_feature_source: str | None
    merged_sources: tuple[str, ...]
    excluded_from_panel_merge: tuple[dict[str, object], ...]
    target_options: tuple[str, ...] = PANEL_TARGET_OPTIONS

    @property
    def eligible_feature_count(self) -> int:
        return len(self.eligible_features)

    @property
    def excluded_feature_count(self) -> int:
        return sum(
            1
            for record in self.records
            if record.exclusion_reason != FeatureExclusionReason.ELIGIBLE_FEATURE
        )


@dataclass(frozen=True, slots=True)
class PanelSupervisionFrame:
    """Ticker-level panel with full eligible indicator universe (not W4-B date-level)."""

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    target_columns: tuple[str, ...]
    source_paths: dict[str, str | None]
    grain: PanelGrain
    provenance: dict[str, object]
    universe: IndicatorUniverseResult


def default_panel_model_output_dir(config: P2Config | None = None) -> Path:
    if config is not None:
        return Path(config.panel_model_output_dir)
    return _DEFAULT_OUTPUT_DIR


def w3b_indicator_features_path(config: P2Config) -> Path:
    """Legacy W3-B artifact path (fallback when canonical preprocessing is absent)."""
    return (
        Path(config.w3_b_root)
        / "feature_panels"
        / config.supervision_surface
        / "indicator_features.parquet"
    )


def panel_indicator_features_path(config: P2Config) -> Path:
    """Preferred indicator panel path: canonical preprocessing, then W3-B fallback."""
    path, _ = resolve_pipeline_indicator_features_path(config)
    return path


def _prefixed_column_name(source_id: str, column: str) -> str:
    return f"{source_id}__{column}"


def _indicator_feature_count(source_id: str, frame: pd.DataFrame) -> int:
    if source_id in {_W3B_SOURCE_ID, _CANONICAL_SOURCE_ID}:
        return sum(1 for column in frame.columns if column not in _W3B_KEY_COLUMNS)
    if source_id == _W4A_SOURCE_ID:
        return sum(
            1
            for column in frame.columns
            if column not in _MERGE_KEYS
            and not str(column).startswith("child_")
            and str(column)
            not in {
                "bundle_id",
                "surface_id",
                "fold_id",
                "split",
                "hindsight_best_child",
                "hindsight_best_child_utility",
            }
        )
    return max(0, len(frame.columns) - len(_MERGE_KEYS))


def build_source_coverage_meta(
    config: P2Config,
    discovered: DiscoveredSources,
    *,
    merge_plan: PanelMergePlan | None = None,
) -> dict[str, object]:
    """Per-source existence, row counts, and indicator feature counts."""

    indicator_path, indicator_source_kind = resolve_pipeline_indicator_features_path(config)
    w3b_path = w3b_indicator_features_path(config)
    detail: dict[str, object] = {}
    indicator_counts: dict[str, int] = {}
    all_sources = [*discovered.panel_sources, *discovered.router_sources]
    for source in all_sources:
        exists = source.path is None or Path(source.path).is_file()
        if source.source_id == _SYNTHETIC_SOURCE_ID:
            exists = True
        row_count = int(len(source.frame))
        feature_count = int(len(source.frame.columns))
        indicator_count = _indicator_feature_count(source.source_id, source.frame)
        status = "router_only"
        if merge_plan is not None:
            if source.source_id == merge_plan.primary.source_id:
                status = "primary_panel"
            elif any(item.source_id == source.source_id for item in merge_plan.optional_context):
                status = "merged_context"
            elif any(
                item.get("source_id") == source.source_id
                for item in merge_plan.excluded_from_panel_merge
            ):
                status = "excluded_from_panel_merge"
        entry: dict[str, object] = {
            "path": source.path,
            "exists": exists,
            "row_count": row_count,
            "feature_count": feature_count,
            "indicator_feature_count": indicator_count,
            "panel_merge_status": status,
        }
        if source.load_report is not None:
            entry["load_report"] = dict(source.load_report)
        detail[source.source_id] = entry
        indicator_counts[source.source_id] = indicator_count

    return {
        "sources_detail": detail,
        "panel_indicator_features_path": str(indicator_path),
        "panel_indicator_source_kind": indicator_source_kind,
        "w3_b_indicator_features_path": str(w3b_path),
        "w3_b_indicator_features_exists": w3b_path.is_file(),
        "indicator_feature_count_by_source": indicator_counts,
    }


def discover_column_sources(
    config: P2Config,
    *,
    smoke_frame: pd.DataFrame | None = None,
) -> DiscoveredSources:
    """Load panel and router sources; W4-A is router-only unless explicitly enabled."""

    panel_sources: list[ColumnSource] = []
    router_sources: list[ColumnSource] = []
    if smoke_frame is not None:
        panel_sources.append(
            ColumnSource(
                source_id=_SYNTHETIC_SOURCE_ID,
                path=None,
                frame=smoke_frame,
                feature_family="synthetic",
            )
        )
        return DiscoveredSources(
            panel_sources=tuple(panel_sources),
            router_sources=tuple(router_sources),
        )

    indicator_path, indicator_source_kind = resolve_pipeline_indicator_features_path(config)
    if not indicator_path.is_file():
        w3b_fallback = w3b_indicator_features_path(config)
        raise FileNotFoundError(
            f"Indicator panel not found at canonical path {indicator_path} "
            f"or W3-B fallback {w3b_fallback}. "
            "Run `python -m pysrc.cli.marketmind dataprep run -c <config>` or build W3-B first."
        )
    indicators = pd.read_parquet(indicator_path, engine="pyarrow")
    if indicator_source_kind == "pipeline_preprocessing":
        collapsed = normalize_panel_keys(indicators)
        collapse_report: dict[str, object] = {"source_kind": indicator_source_kind}
        source_id = _CANONICAL_SOURCE_ID
        feature_family = "pipeline_preprocessing"
    else:
        collapsed, collapse_report = collapse_slim_indicator_fold_duplicates(indicators)
        source_id = _W3B_SOURCE_ID
        feature_family = "pandas_ta"
    panel_sources.append(
        ColumnSource(
            source_id=source_id,
            path=str(indicator_path),
            frame=collapsed,
            feature_family=feature_family,
            load_report=collapse_report,
        )
    )

    if supervision_artifact_exists(config):
        w4a_path = resolve_supervision_path(config)
        w4a = pd.read_parquet(w4a_path, engine="pyarrow")
        router_sources.append(
            ColumnSource(
                source_id=_W4A_SOURCE_ID,
                path=str(w4a_path),
                frame=w4a,
                feature_family="router_supervision",
            )
        )

    return DiscoveredSources(
        panel_sources=tuple(panel_sources),
        router_sources=tuple(router_sources),
    )


def partition_sources_for_panel_merge(
    config: P2Config,
    discovered: DiscoveredSources,
) -> PanelMergePlan:
    """Validate grains and choose W3-B primary panel merge sources."""

    reports: dict[str, SourceGrainReport] = {}
    excluded: list[dict[str, object]] = []
    optional_context: list[ColumnSource] = []

    if (
        len(discovered.panel_sources) == 1
        and discovered.panel_sources[0].source_id == _SYNTHETIC_SOURCE_ID
    ):
        primary = discovered.panel_sources[0]
        report = validate_source_grain(primary.frame, primary.source_id)
        reports[primary.source_id] = report
        if not report.valid:
            raise ValueError(
                f"Synthetic panel source failed grain validation: {report.exclusion_reason}"
            )
        return PanelMergePlan(
            primary=primary,
            optional_context=(),
            excluded_from_panel_merge=tuple(excluded),
            source_grain_reports=reports,
        )

    w3b_candidates = [
        source
        for source in discovered.panel_sources
        if source.source_id in {_W3B_SOURCE_ID, _CANONICAL_SOURCE_ID}
    ]
    if not w3b_candidates:
        raise FileNotFoundError(
            "Canonical or W3-B indicator source is required for non-smoke panel universe."
        )
    primary = w3b_candidates[0]
    w3b_report = validate_source_grain(primary.frame, primary.source_id)
    reports[primary.source_id] = w3b_report
    if not w3b_report.valid:
        raise ValueError(
            f"W3-B indicator source must be unique on {TICKER_DATE_INTERVAL_KEYS}; "
            f"duplicate_key_count={w3b_report.duplicate_key_count}"
        )

    for source in discovered.panel_sources:
        if source is primary or source.source_id == primary.source_id:
            continue
        report = validate_source_grain(source.frame, source.source_id)
        reports[source.source_id] = report
        if report.valid:
            optional_context.append(source)
        else:
            excluded.append(
                {
                    "source_id": source.source_id,
                    "reason": report.exclusion_reason or "grain_invalid",
                    "duplicate_key_count": report.duplicate_key_count,
                }
            )

    for source in discovered.router_sources:
        report = validate_source_grain(source.frame, source.source_id)
        reports[source.source_id] = report
        if config.panel_include_w4a_features and report.valid:
            optional_context.append(source)
        else:
            reason = (
                _W4A_EXCLUSION_INCOMPATIBLE_ROUTER_GRAIN
                if not report.valid
                else "w4a_opt_in_disabled"
            )
            excluded.append(
                {
                    "source_id": source.source_id,
                    "reason": reason,
                    "duplicate_key_count": report.duplicate_key_count,
                }
            )

    return PanelMergePlan(
        primary=primary,
        optional_context=tuple(optional_context),
        excluded_from_panel_merge=tuple(excluded),
        source_grain_reports=reports,
    )


def _collision_value_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
    merge_keys: tuple[str, ...],
) -> float | None:
    if column not in left.columns or column not in right.columns:
        return None
    keys = list(merge_keys)
    joined = left.loc[:, [*keys, column]].merge(
        right.loc[:, [*keys, column]],
        on=keys,
        how="inner",
        suffixes=("_left", "_right"),
    )
    if joined.empty:
        return None
    left_vals = pd.to_numeric(joined[f"{column}_left"], errors="coerce")
    right_vals = pd.to_numeric(joined[f"{column}_right"], errors="coerce")
    if left_vals.notna().sum() == 0 and right_vals.notna().sum() == 0:
        return None
    delta = (left_vals - right_vals).abs()
    if delta.notna().any():
        return float(delta.max())
    return 0.0


def _merge_sources(merge_plan: PanelMergePlan) -> MergeSourcesResult:
    merge_keys = _MERGE_KEYS
    provenance: dict[str, object] = {}
    collisions: list[dict[str, object]] = []
    source_column_map: dict[str, tuple[str, ...]] = {}

    base_source = merge_plan.primary
    merged = normalize_panel_keys(base_source.frame)
    base_columns = [column for column in base_source.frame.columns if column not in merge_keys]
    source_column_map[base_source.source_id] = tuple(base_columns)
    provenance[base_source.source_id] = {
        "path": base_source.path,
        "column_count": int(len(base_source.frame.columns)),
        "row_count": int(len(base_source.frame)),
        "merged_columns": list(base_columns),
    }

    w3_b_loaded = base_source.source_id in {_W3B_SOURCE_ID, _CANONICAL_SOURCE_ID}
    merged_sources: list[str] = [base_source.source_id]
    for source in merge_plan.optional_context:
        if source.source_id in {_W3B_SOURCE_ID, _CANONICAL_SOURCE_ID}:
            w3_b_loaded = True
        part = normalize_panel_keys(source.frame)
        rename_map: dict[str, str] = {}
        incoming_columns: list[str] = []
        for column in part.columns:
            if column in merge_keys:
                continue
            if column in merged.columns:
                prefixed = _prefixed_column_name(source.source_id, column)
                rename_map[column] = prefixed
                incoming_columns.append(prefixed)
                max_delta = _collision_value_delta(merged, part, column, merge_keys)
                collisions.append(
                    {
                        "column_name": column,
                        "existing_source": _infer_existing_source(
                            column, source_column_map, exclude=source.source_id
                        ),
                        "incoming_source": source.source_id,
                        "resolution": "prefixed",
                        "merged_name": prefixed,
                        "rows_compared": int(
                            merged.merge(
                                part.loc[:, [*merge_keys, column]], on=list(merge_keys), how="inner"
                            ).shape[0]
                        ),
                        "max_abs_delta": max_delta,
                    }
                )
            else:
                incoming_columns.append(column)
        part_renamed = part.rename(columns=rename_map)
        merged = merged.merge(part_renamed, on=list(merge_keys), how="left")
        source_column_map[source.source_id] = tuple(incoming_columns)
        merged_sources.append(source.source_id)
        provenance[source.source_id] = {
            "path": source.path,
            "column_count": int(len(source.frame.columns)),
            "row_count": int(len(source.frame)),
            "merged_columns": list(incoming_columns),
            "prefixed_columns": list(rename_map.values()),
        }

    merged = merged.sort_values(list(merge_keys), kind="mergesort").reset_index(drop=True)
    provenance["excluded_from_panel_merge"] = list(merge_plan.excluded_from_panel_merge)
    provenance["primary_feature_source"] = base_source.source_id
    provenance["merged_sources"] = list(merged_sources)
    return MergeSourcesResult(
        frame=merged,
        provenance=provenance,
        source_column_map=source_column_map,
        collisions=tuple(collisions),
        w3_b_indicator_loaded=w3_b_loaded,
        primary_feature_source=base_source.source_id,
        merged_sources=tuple(merged_sources),
        excluded_from_panel_merge=merge_plan.excluded_from_panel_merge,
    )


def _infer_existing_source(
    column: str,
    source_column_map: dict[str, tuple[str, ...]],
    *,
    exclude: str,
) -> str:
    for source_id, columns in source_column_map.items():
        if source_id == exclude:
            continue
        if column in columns:
            return source_id
        prefixed = _prefixed_column_name(source_id, column)
        if prefixed in columns:
            return source_id
    return "unknown"


def _train_coverage_mask(frame: pd.DataFrame) -> pd.Series:
    if "split" not in frame.columns:
        return pd.Series(True, index=frame.index)
    split = frame["split"].astype(str).str.lower()
    return split.eq("train") | split.eq("fit")


def _discovered_complete(
    source_column_map: dict[str, tuple[str, ...]],
    records: tuple[FeatureColumnRecord, ...],
) -> bool:
    classified = {record.feature_name for record in records}
    expected: set[str] = set()
    for columns in source_column_map.values():
        expected.update(columns)
    return expected.issubset(classified)


def build_indicator_universe(
    config: P2Config,
    *,
    smoke_frame: pd.DataFrame | None = None,
    coverage_threshold: float | None = None,
) -> IndicatorUniverseResult:
    """Discover every column from sources and classify eligibility with reasons."""

    threshold = (
        coverage_threshold
        if coverage_threshold is not None
        else float(config.panel_coverage_threshold)
    )
    discovered = discover_column_sources(config, smoke_frame=smoke_frame)
    merge_plan = partition_sources_for_panel_merge(config, discovered)
    coverage_meta = build_source_coverage_meta(config, discovered, merge_plan=merge_plan)
    merge_result = _merge_sources(merge_plan)
    panel_frame = merge_result.frame
    per_source_dupes = {
        source_id: report.duplicate_key_count
        for source_id, report in merge_plan.source_grain_reports.items()
    }
    grain_audit = audit_panel_grain(
        panel_frame,
        per_source_duplicate_key_count=per_source_dupes,
    )
    train_mask = _train_coverage_mask(panel_frame)

    merged_source_ids = {*merge_result.merged_sources}
    records: list[FeatureColumnRecord] = []
    merge_sources_list = (merge_plan.primary, *merge_plan.optional_context)
    for source in merge_sources_list:
        if source.source_id not in merged_source_ids:
            continue
        merged_columns = merge_result.source_column_map.get(source.source_id, ())
        for column in merged_columns:
            if column not in panel_frame.columns:
                continue
            series = (
                panel_frame.loc[train_mask, column] if train_mask.any() else panel_frame[column]
            )
            family = source.feature_family
            if not pd.api.types.is_numeric_dtype(series):
                family = "categorical_criterion"
            records.append(
                classify_column(
                    column,
                    series,
                    source=source.source_id,
                    feature_family=family,
                    interval=source.interval,
                    coverage_threshold=threshold,
                )
            )

    eligible = tuple(
        record.feature_name
        for record in records
        if record.exclusion_reason == FeatureExclusionReason.ELIGIBLE_FEATURE
    )
    w3b_exists = bool(
        coverage_meta.get("panel_indicator_features_path")
        and Path(str(coverage_meta["panel_indicator_features_path"])).is_file()
    )
    uses_full_indicator = (
        w3b_exists
        and merge_result.w3_b_indicator_loaded
        and smoke_frame is None
        and merge_result.primary_feature_source in {_W3B_SOURCE_ID, _CANONICAL_SOURCE_ID}
    )
    uses_full_discovered = _discovered_complete(merge_result.source_column_map, tuple(records))

    combined_sources = {**merge_result.provenance, **coverage_meta}

    return IndicatorUniverseResult(
        panel_frame=panel_frame,
        records=tuple(records),
        eligible_features=eligible,
        grain_audit=grain_audit,
        sources=combined_sources,
        source_column_map=merge_result.source_column_map,
        collisions=merge_result.collisions,
        uses_full_discovered_feature_universe=uses_full_discovered,
        uses_full_indicator_universe=uses_full_indicator,
        sources_detail=cast_sources_detail(coverage_meta),
        primary_feature_source=merge_result.primary_feature_source,
        merged_sources=merge_result.merged_sources,
        excluded_from_panel_merge=merge_result.excluded_from_panel_merge,
    )


def cast_sources_detail(coverage_meta: dict[str, object]) -> dict[str, object]:
    detail = coverage_meta.get("sources_detail")
    return detail if isinstance(detail, dict) else {}


def _exclusion_counts(records: tuple[FeatureColumnRecord, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.exclusion_reason.value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _categorical_exclusion_counts(records: tuple[FeatureColumnRecord, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.exclusion_reason != FeatureExclusionReason.NON_NUMERIC_UNSUPPORTED:
            continue
        key = str(record.exclusion_reason.value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _w3b_indicator_row_collapse_from_detail(
    sources_detail: dict[str, object],
) -> dict[str, object] | None:
    meta = sources_detail.get(_W3B_SOURCE_ID)
    if not isinstance(meta, dict):
        return None
    load_report = meta.get("load_report")
    return dict(load_report) if isinstance(load_report, dict) else None


def _source_column_counts(sources_detail: dict[str, object]) -> dict[str, int]:
    out: dict[str, int] = {}
    for source_id, meta in sources_detail.items():
        if isinstance(meta, dict):
            out[source_id] = int(meta.get("column_count", 0))
    return out


def write_feature_universe_artifacts(
    result: IndicatorUniverseResult,
    output_dir: Path,
    *,
    model_feature_usage_rows: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    """Write required audit artifacts (atomic JSON/CSV)."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible_rows = [
        {
            "feature_name": record.feature_name,
            "source": record.source,
            "feature_family": record.feature_family,
            "interval": record.interval,
            "coverage": record.coverage,
            "dtype": record.dtype,
            "used_by_default": record.used_by_default,
        }
        for record in result.records
        if record.is_eligible
    ]
    excluded_rows = [
        {
            "feature_name": record.feature_name,
            "source": record.source,
            "exclusion_reason": record.exclusion_reason.value,
            "coverage": record.coverage,
            "dtype": record.dtype,
        }
        for record in result.records
        if not record.is_eligible
    ]
    by_source_rows = [
        {
            "feature_name": record.feature_name,
            "source": record.source,
            "feature_family": record.feature_family,
            "interval": record.interval,
            "exclusion_reason": record.exclusion_reason.value,
        }
        for record in result.records
    ]
    categorical_rows = [
        {
            "feature_name": record.feature_name,
            "source": record.source,
            "dtype": record.dtype,
            "coverage": record.coverage,
            "unique_count": int(result.panel_frame[record.feature_name].nunique(dropna=True)),
            "exclusion_reason": record.exclusion_reason.value,
        }
        for record in result.records
        if record.exclusion_reason == FeatureExclusionReason.NON_NUMERIC_UNSUPPORTED
    ]

    grain = result.grain_audit
    sources_detail = result.sources_detail
    duplicate_sample_path: str | None = None
    paths = {
        "feature_universe_report": output_dir / "feature_universe_report.json",
        "eligible_features": output_dir / "eligible_features.csv",
        "excluded_features": output_dir / "excluded_features.csv",
        "feature_columns_by_source": output_dir / "feature_columns_by_source.csv",
        "unsupported_categorical_features": output_dir / "unsupported_categorical_features.csv",
        "model_feature_usage": output_dir / "model_feature_usage.csv",
        "source_column_collision_report": output_dir / "source_column_collision_report.json",
        "duplicate_key_sample": output_dir / DUPLICATE_KEY_SAMPLE_FILENAME,
    }
    if not grain.valid:
        duplicate_sample = build_duplicate_key_sample(
            result.panel_frame,
            grain.key_columns_used,
        )
        duplicate_sample.to_csv(paths["duplicate_key_sample"], index=False)
        duplicate_sample_path = DUPLICATE_KEY_SAMPLE_FILENAME

    report = {
        "total_discovered_columns": len(result.records),
        "eligible_feature_count": result.eligible_feature_count,
        "excluded_feature_count": result.excluded_feature_count,
        "sources": _source_column_counts(sources_detail),
        "sources_detail": sources_detail,
        "w3_b_indicator_features_path": result.sources.get("w3_b_indicator_features_path"),
        "w3_b_indicator_features_exists": result.sources.get("w3_b_indicator_features_exists"),
        "indicator_feature_count_by_source": result.sources.get(
            "indicator_feature_count_by_source"
        ),
        "exclusion_counts_by_reason": _exclusion_counts(result.records),
        "categorical_feature_count": len(categorical_rows),
        "categorical_exclusion_counts_by_reason": _categorical_exclusion_counts(result.records),
        "intervals_detected": list(grain.intervals_detected),
        "ticker_count": grain.ticker_count,
        "date_start": grain.date_start,
        "date_end": grain.date_end,
        "row_grain": ROW_GRAIN,
        "row_count": grain.row_count,
        "duplicate_key_count": grain.duplicate_key_count,
        "missing_key_columns": list(grain.missing_key_columns),
        "key_columns_used": list(grain.key_columns_used),
        "duplicate_key_sample_path": duplicate_sample_path,
        "per_source_duplicate_key_count": dict(grain.per_source_duplicate_key_count),
        "primary_feature_source": result.primary_feature_source,
        "merged_sources": list(result.merged_sources),
        "excluded_from_panel_merge": list(result.excluded_from_panel_merge),
        "uses_full_discovered_feature_universe": result.uses_full_discovered_feature_universe,
        "uses_full_indicator_universe": result.uses_full_indicator_universe,
        "grain_valid": grain.valid,
        "target_options": list(result.target_options),
        "source_provenance": {k: v for k, v in result.sources.items() if k != "sources_detail"},
        "collision_count": len(result.collisions),
        "w3b_indicator_row_collapse": _w3b_indicator_row_collapse_from_detail(sources_detail),
    }

    atomic_write_json(paths["feature_universe_report"], report)
    atomic_write_json(
        paths["source_column_collision_report"],
        {
            "collision_count": len(result.collisions),
            "collisions": list(result.collisions),
        },
    )
    pd.DataFrame(eligible_rows).to_csv(paths["eligible_features"], index=False)
    pd.DataFrame(excluded_rows).to_csv(paths["excluded_features"], index=False)
    pd.DataFrame(by_source_rows).to_csv(paths["feature_columns_by_source"], index=False)
    pd.DataFrame(categorical_rows).to_csv(paths["unsupported_categorical_features"], index=False)

    usage = model_feature_usage_rows or [
        {
            "model_id": "(pending)",
            "candidate_id": "(pending)",
            "feature_name": "(pending)",
            "used_as_input": False,
            "selection_stage": "universe_audit_only",
            "importance_if_available": np.nan,
        }
    ]
    pd.DataFrame(usage).to_csv(paths["model_feature_usage"], index=False)
    return paths


def generate_synthetic_panel_frame(
    *,
    n_rows: int = 200,
    n_indicators: int = 30,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Wide synthetic panel for smoke tests — discovers columns, not a fixed TA shortlist."""

    rng = np.random.default_rng(random_seed)
    dates = pd.date_range("2023-01-01", periods=n_rows // 5, freq="B")
    tickers = [f"T{i:03d}" for i in range(5)]
    rows: list[dict[str, object]] = []
    row_index = 0
    for date in dates:
        for ticker in tickers:
            if row_index % 10 < 7:
                split = "train"
            elif row_index % 10 < 9:
                split = "validation"
            else:
                split = "test"
            row: dict[str, object] = {
                "date": date.strftime("%Y-%m-%d"),
                "instrument": ticker,
                "fold_id": "fold_0",
                "split": split,
                "interval": "daily",
                "bundle_id": "w3b_ta",
                "surface_id": "surface_b",
                "regime_id": "trend_bull__vol_low",
                "forward_return": float(rng.normal(scale=0.01)),
                "future_return": float(rng.normal()),
                "hindsight_best_child": "alpha",
            }
            for idx in range(n_indicators):
                row[f"ta_feat_{idx:03d}"] = float(rng.normal())
            rows.append(row)
            row_index += 1
    return pd.DataFrame(rows)


def build_panel_supervision_frame(config: P2Config) -> PanelSupervisionFrame:
    """Build ticker-level supervision with full eligible indicator universe."""

    smoke = (
        generate_synthetic_panel_frame(random_seed=config.random_seed)
        if config.smoke_test
        else None
    )
    universe = build_indicator_universe(config, smoke_frame=smoke)
    if not universe.eligible_features:
        raise ValueError(
            "No eligible features after universe audit. See excluded_features.csv for reasons."
        )
    output_dir = default_panel_model_output_dir(config)
    write_feature_universe_artifacts(universe, output_dir)

    if not universe.grain_audit.valid and not config.smoke_test:
        sample_hint = (
            universe.grain_audit.duplicate_key_sample_path or DUPLICATE_KEY_SAMPLE_FILENAME
        )
        raise ValueError(
            "Panel grain audit failed: "
            f"missing={universe.grain_audit.missing_key_columns}, "
            f"duplicate_keys={universe.grain_audit.duplicate_key_count}. "
            f"Inspect {output_dir / sample_hint}"
        )

    target_columns = tuple(
        record.feature_name
        for record in universe.records
        if record.exclusion_reason == FeatureExclusionReason.TARGET_COLUMN
    )

    return PanelSupervisionFrame(
        frame=universe.panel_frame,
        feature_names=universe.eligible_features,
        target_columns=target_columns,
        source_paths={
            "w4_a": str(resolve_supervision_path(config))
            if supervision_artifact_exists(config)
            else None,
            "panel_indicators": str(panel_indicator_features_path(config))
            if panel_indicator_features_path(config).is_file()
            else None,
            "w3_b_indicators": str(w3b_indicator_features_path(config))
            if w3b_indicator_features_path(config).is_file()
            else None,
        },
        grain="ticker_date_interval",
        provenance={
            "feature_policy": "full_indicator_universe_v1",
            "eligible_feature_count": universe.eligible_feature_count,
            "excluded_feature_count": universe.excluded_feature_count,
            "audit_output_dir": str(output_dir),
            "uses_full_discovered_feature_universe": universe.uses_full_discovered_feature_universe,
            "uses_full_indicator_universe": universe.uses_full_indicator_universe,
        },
        universe=universe,
    )


def record_model_feature_usage(
    *,
    model_id: str,
    candidate_id: str,
    feature_names: tuple[str, ...],
    selection_stage: str = "fit",
    importance: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Build model_feature_usage.csv rows for a trained P2-PANEL model."""

    rows: list[dict[str, object]] = []
    for name in feature_names:
        rows.append(
            {
                "model_id": model_id,
                "candidate_id": candidate_id,
                "feature_name": name,
                "used_as_input": True,
                "selection_stage": selection_stage,
                "importance_if_available": (importance or {}).get(name, np.nan),
            }
        )
    return rows


def load_feature_universe_report(output_dir: Path) -> dict[str, object]:
    path = Path(output_dir) / "feature_universe_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"Feature universe report not found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Feature universe report must be a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def require_panel_grain_valid_for_training(config: P2Config) -> None:
    """Abort panel model training when the latest universe audit has grain_valid=false."""

    if config.smoke_test:
        return
    report_path = default_panel_model_output_dir(config) / "feature_universe_report.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("grain_valid") is True:
        return
    sample = report.get("duplicate_key_sample_path") or DUPLICATE_KEY_SAMPLE_FILENAME
    raise ValueError(
        "Panel grain_valid=false in feature_universe_report.json. "
        f"Rerun: python -m pysrc.cli.marketmind panel audit-features and inspect {sample} before training."
    )
