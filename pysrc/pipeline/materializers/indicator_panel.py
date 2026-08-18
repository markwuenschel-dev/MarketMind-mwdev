"""Materialize data/processed/full_indicator_feature_panel from pipeline output."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.products import resolve_pipeline_product_paths
from pysrc.pipeline.stages.market_data.sources.sip_panel import (
    SIP_ADJUSTED_PANEL_SOURCE,
    build_sip_base_panel,
    load_sip_adjusted_panel,
)
from pysrc.pipeline.stages.preprocessing.indicators.config import IndicatorLibraryConfig
from pysrc.pipeline.stages.preprocessing.indicators.engine import IndicatorEngine
from pysrc.pipeline.stages.preprocessing.indicators.schema import (
    W3B_INDICATOR_IDS,
    W3B_INDICATOR_SCHEMA_VERSION,
)

LOG = get_logger(__name__)

_DEFAULT_INTERVAL = "daily"
_PANEL_SUPERVISION_COLUMN_ALLOWLIST: tuple[str, ...] = (
    "forward_return",
    "forward_return_horizon",
    "adjusted_return_1d",
    "raw_return_1d",
    "split",
)
_PANEL_MERGE_KEYS: tuple[str, ...] = ("date", "instrument")


def _normalize_merge_keys(frame: pd.DataFrame, merge_keys: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    if "date" in merge_keys and "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=False).dt.strftime(
            "%Y-%m-%d"
        )
    if "instrument" in merge_keys and "instrument" in out.columns:
        out["instrument"] = out["instrument"].astype(str)
    return out


def attach_panel_supervision_columns(
    source: pd.DataFrame,
    features: pd.DataFrame,
    *,
    merge_keys: tuple[str, ...] = _PANEL_MERGE_KEYS,
) -> pd.DataFrame:
    """Join panel supervision targets from the cleaned frame onto indicator features."""

    carry_columns = [
        column
        for column in _PANEL_SUPERVISION_COLUMN_ALLOWLIST
        if column in source.columns and column not in features.columns
    ]
    if not carry_columns:
        return features

    key_columns = [
        column for column in merge_keys if column in source.columns and column in features.columns
    ]
    if len(key_columns) != len(merge_keys):
        return features

    source_norm = _normalize_merge_keys(source, merge_keys)
    features_norm = _normalize_merge_keys(features, merge_keys)
    supervision = source_norm.loc[:, [*key_columns, *carry_columns]].drop_duplicates(
        subset=key_columns
    )
    merged = features_norm.merge(supervision, on=key_columns, how="left")
    return merged


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    try:
        import polars as pl

        if isinstance(frame, pl.DataFrame):
            return frame.to_pandas()
    except ImportError:
        pass
    return pd.DataFrame(frame)


def materialize_indicator_panel_from_frame(
    frame: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Write pipeline preprocessing output to the canonical indicator product."""

    if not config.get("enabled", True):
        raise ValueError("indicator_panel materialization is disabled in config")

    processed_root = Path(config.get("processed_data_root", "data/processed"))
    pdf = _to_pandas(frame)
    if "interval" not in pdf.columns:
        pdf = pdf.copy()
        pdf["interval"] = _DEFAULT_INTERVAL

    paths = resolve_pipeline_product_paths(processed_root)
    paths.full_indicator_feature_panel.mkdir(parents=True, exist_ok=True)
    out_path = paths.indicator_panel_parquet
    tmp = out_path.parent / f".{out_path.name}.tmp"
    pdf.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)

    indicator_columns = [column for column in pdf.columns if column in W3B_INDICATOR_IDS]
    supervision_columns = [
        column for column in pdf.columns if column in _PANEL_SUPERVISION_COLUMN_ALLOWLIST
    ]
    target_metadata: dict[str, object] = {}
    if "forward_return_horizon" in supervision_columns:
        target_metadata["forward_return"] = {
            "column": "forward_return_horizon",
            "horizon_days": 1,
        }

    manifest_payload = {
        "schema_version": "processed_data_manifest.v1",
        "indicator_schema_version": W3B_INDICATOR_SCHEMA_VERSION,
        "producer": "pipeline.materializers.indicator_panel",
        "engine": "IndicatorEngineStep",
        "products": {"full_indicator_feature_panel": str(out_path)},
        "row_count": int(len(pdf)),
        "instrument_count": int(pdf["instrument"].nunique()) if "instrument" in pdf.columns else 0,
        "indicator_columns": indicator_columns,
        "supervision_columns": supervision_columns,
        "target_metadata": target_metadata,
        "row_grain": "ticker_date_interval",
        "key_columns": ["date", "instrument", "interval"],
    }
    atomic_write_json(paths.manifest, manifest_payload)

    sidecar = paths.full_indicator_feature_panel / "build_report.json"
    atomic_write_json(
        sidecar,
        {
            "schema_version": "full_indicator_feature_panel.v1",
            "source_kind": "pipeline_preprocessing",
            "row_count": int(len(pdf)),
            "column_count": int(len(pdf.columns)),
            "indicator_column_count": len(indicator_columns),
        },
    )

    LOG.info(
        "indicator_panel_materialized",
        path=str(out_path),
        rows=len(pdf),
        indicators=len(indicator_columns),
    )
    return {
        "path": str(out_path),
        "manifest": str(paths.manifest),
        "row_count": int(len(pdf)),
        "indicator_columns": indicator_columns,
    }


def materialize_full_indicator_panel(
    config: Mapping[str, Any],
    *,
    base_panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Legacy test helper: compute indicators without running full dataprep stages."""

    if base_panel is None:
        source = config.get("source") or {}
        kind = str(source.get("kind", "sip_adjusted_panel"))
        if kind != "sip_adjusted_panel":
            raise ValueError(f"Unsupported indicator panel source kind: {kind!r}")
        path = Path(source.get("path", SIP_ADJUSTED_PANEL_SOURCE))
        workers = max(1, int(config.get("workers", 1)))
        panel = load_sip_adjusted_panel(path, workers=workers)
        base_panel = build_sip_base_panel(panel, copy=False)

    processed_root = Path(config.get("processed_data_root", "data/processed"))
    workers = max(1, int(config.get("workers", 1)))
    scratch_dir = processed_root / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    engine = IndicatorEngine(IndicatorLibraryConfig())
    result = engine.compute(
        base_panel,
        workers=workers,
        copy_input=False,
        ta_scratch_path=scratch_dir / "indicator_engine_ta_input.parquet",
    )
    frame = attach_panel_supervision_columns(base_panel, result.features.copy())
    if "interval" not in frame.columns:
        frame["interval"] = _DEFAULT_INTERVAL
    return materialize_indicator_panel_from_frame(frame, config)


__all__ = [
    "attach_panel_supervision_columns",
    "materialize_full_indicator_panel",
    "materialize_indicator_panel_from_frame",
]
