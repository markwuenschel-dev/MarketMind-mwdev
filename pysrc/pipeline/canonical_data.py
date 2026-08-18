"""Canonical processed-product validation for pipeline lanes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pysrc.contracts.feature_channel import (
    channel_manifest_with_optional_macro,
)
from pysrc.pipeline.panel.feature_grain_audit import TICKER_DATE_INTERVAL_KEYS, audit_panel_grain
from pysrc.pipeline.products import require_pipeline_indicator_panel
from pysrc.pipeline.synthetic_panel import generate_synthetic_panel_frame


@dataclass(frozen=True, slots=True)
class CanonicalDataAudit:
    canonical_source_used: bool
    legacy_artifact_fallback_used: bool
    grain_valid: bool
    duplicate_key_count: int
    row_count: int
    ticker_count: int
    date_start: str | None
    date_end: str | None
    intervals_detected: tuple[str, ...]
    panel_path: str | None
    channel_manifest: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "canonical_data_audit.v1",
            "canonical_source_used": self.canonical_source_used,
            "legacy_artifact_fallback_used": self.legacy_artifact_fallback_used,
            "grain_valid": self.grain_valid,
            "duplicate_key_count": self.duplicate_key_count,
            "row_count": self.row_count,
            "ticker_count": self.ticker_count,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "intervals_detected": list(self.intervals_detected),
            "panel_path": self.panel_path,
            "channel_manifest": self.channel_manifest,
        }


@dataclass(frozen=True, slots=True)
class CanonicalPanelSource:
    """Canonical panel metadata resolved before choosing a training execution mode."""

    panel_path: Path | None
    manifest_path: Path | None
    manifest: dict[str, object]
    schema: dict[str, str]
    row_count: int
    expected_row_grain: str
    product_identity: str
    file_size_bytes: int | None
    file_mtime_ns: int | None
    config_hash: str | None
    target_metadata: dict[str, object]
    smoke_panel: pd.DataFrame | None = None
    audit: CanonicalDataAudit | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "canonical_panel_source.v1",
            "panel_path": str(self.panel_path) if self.panel_path is not None else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "row_count": self.row_count,
            "expected_row_grain": self.expected_row_grain,
            "product_identity": self.product_identity,
            "file_size_bytes": self.file_size_bytes,
            "file_mtime_ns": self.file_mtime_ns,
            "config_hash": self.config_hash,
            "schema_columns": list(self.schema),
            "target_metadata": self.target_metadata,
        }

    def load_pandas(self, *, columns: list[str] | None = None) -> pd.DataFrame:
        if self.smoke_panel is not None:
            if columns is None:
                return self.smoke_panel.copy()
            return self.smoke_panel.loc[
                :, [c for c in columns if c in self.smoke_panel.columns]
            ].copy()
        if self.panel_path is None:
            raise FileNotFoundError("Canonical panel source has no parquet path")
        return pd.read_parquet(self.panel_path, columns=columns)


def _pandas_schema(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Canonical manifest must be a mapping: {path}")
    return payload


def require_canonical_panel_source_for_real_run(
    config: Any,
    *,
    smoke_test: bool,
) -> CanonicalPanelSource:
    """Resolve canonical panel metadata without forcing pandas materialization."""

    macro_enabled = bool(getattr(config, "macro_state_channel_enabled", False))
    config_hash = getattr(config, "experiment_config_hash", None)
    target_metadata: dict[str, object] = {}
    horizon = getattr(config, "panel_target_horizon_days", None)
    target = str(getattr(config, "panel_target", "forward_return"))
    if horizon is not None:
        target_metadata[target] = {"horizon_days": int(horizon)}

    if smoke_test:
        panel = generate_synthetic_panel_frame(random_seed=getattr(config, "random_seed", 42))
        audit = validate_canonical_panel(
            panel,
            smoke_test=True,
            macro_channel_enabled=macro_enabled,
        )
        return CanonicalPanelSource(
            panel_path=None,
            manifest_path=None,
            manifest={},
            schema=_pandas_schema(panel),
            row_count=len(panel),
            expected_row_grain="ticker_date_interval",
            product_identity="synthetic_panel",
            file_size_bytes=None,
            file_mtime_ns=None,
            config_hash=config_hash,
            target_metadata=target_metadata,
            smoke_panel=panel,
            audit=audit,
        )

    panel_path = Path(require_pipeline_indicator_panel(config))
    if not panel_path.is_file():
        raise FileNotFoundError(
            f"Canonical panel required for real run but missing: {panel_path}. "
            "Run dataprep first or use --smoke-test."
        )

    from pysrc.pipeline.products import resolve_pipeline_product_paths

    paths = resolve_pipeline_product_paths(getattr(config, "processed_data_root", None))
    manifest_path = paths.manifest
    manifest = _load_manifest(manifest_path)

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(panel_path)
    stat = panel_path.stat()
    schema = {field.name: str(field.type) for field in parquet.schema_arrow}
    manifest_targets = manifest.get("target_metadata")
    if isinstance(manifest_targets, dict):
        target_metadata.update(manifest_targets)

    return CanonicalPanelSource(
        panel_path=panel_path,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        manifest=manifest,
        schema=schema,
        row_count=int(parquet.metadata.num_rows),
        expected_row_grain=str(manifest.get("row_grain", "ticker_date_interval")),
        product_identity="full_indicator_feature_panel",
        file_size_bytes=int(stat.st_size),
        file_mtime_ns=int(stat.st_mtime_ns),
        config_hash=config_hash,
        target_metadata=target_metadata,
    )


def validate_canonical_panel(
    panel: pd.DataFrame,
    *,
    panel_path: Path | str | None = None,
    smoke_test: bool = False,
    legacy_fallback: bool = False,
    macro_channel_enabled: bool = False,
) -> CanonicalDataAudit:
    if "interval" not in panel.columns:
        panel = panel.copy()
        panel["interval"] = "1d"
    if "instrument" not in panel.columns and "symbol" in panel.columns:
        panel = panel.rename(columns={"symbol": "instrument"})

    grain = audit_panel_grain(panel)
    manifest = channel_manifest_with_optional_macro(include_macro=macro_channel_enabled)
    return CanonicalDataAudit(
        canonical_source_used=not smoke_test and not legacy_fallback,
        legacy_artifact_fallback_used=legacy_fallback,
        grain_valid=grain.valid,
        duplicate_key_count=grain.duplicate_key_count,
        row_count=grain.row_count,
        ticker_count=grain.ticker_count,
        date_start=grain.date_start,
        date_end=grain.date_end,
        intervals_detected=grain.intervals_detected,
        panel_path=str(panel_path) if panel_path is not None else None,
        channel_manifest={
            "primary_channel_id": manifest.primary_channel_id,
            "channel_ids": list(manifest.channel_ids()),
        },
    )


def require_canonical_panel_for_real_run(
    config: Any,
    *,
    smoke_test: bool,
) -> tuple[pd.DataFrame, CanonicalDataAudit]:
    macro_enabled = bool(getattr(config, "macro_state_channel_enabled", False))
    if smoke_test:
        panel = generate_synthetic_panel_frame(random_seed=getattr(config, "random_seed", 42))
        audit = validate_canonical_panel(
            panel,
            smoke_test=True,
            macro_channel_enabled=macro_enabled,
        )
        return panel, audit

    panel_path = Path(require_pipeline_indicator_panel(config))
    if not panel_path.is_file():
        raise FileNotFoundError(
            f"Canonical panel required for real run but missing: {panel_path}. "
            "Run dataprep first or use --smoke-test."
        )
    panel = pd.read_parquet(panel_path)
    audit = validate_canonical_panel(
        panel,
        panel_path=panel_path,
        smoke_test=False,
        macro_channel_enabled=macro_enabled,
    )
    if not audit.grain_valid:
        raise ValueError(
            f"Canonical panel grain invalid: duplicate_key_count={audit.duplicate_key_count}"
        )
    if audit.duplicate_key_count > 0:
        raise ValueError(f"Canonical panel has duplicate keys at grain {TICKER_DATE_INTERVAL_KEYS}")
    return panel, audit


def assert_canonical_source_unchanged(source: CanonicalPanelSource) -> None:
    """Fail closed if the canonical parquet changed during a long train-matrix run."""
    if source.panel_path is None:
        return
    if source.file_size_bytes is None or source.file_mtime_ns is None:
        return
    path = Path(source.panel_path)
    if not path.is_file():
        raise RuntimeError(f"Canonical panel missing before artifact finalize: {path}")
    stat = path.stat()
    current_size = int(stat.st_size)
    current_mtime_ns = int(stat.st_mtime_ns)
    if current_size != int(source.file_size_bytes):
        raise RuntimeError(
            "Canonical panel file size changed during train-matrix: "
            f"path={path} expected_bytes={source.file_size_bytes} current_bytes={current_size}"
        )
    if current_mtime_ns != int(source.file_mtime_ns):
        raise RuntimeError(
            "Canonical panel modification time changed during train-matrix: "
            f"path={path} expected_mtime_ns={source.file_mtime_ns} current_mtime_ns={current_mtime_ns}"
        )


__all__ = [
    "CanonicalDataAudit",
    "CanonicalPanelSource",
    "assert_canonical_source_unchanged",
    "require_canonical_panel_for_real_run",
    "require_canonical_panel_source_for_real_run",
    "validate_canonical_panel",
]
