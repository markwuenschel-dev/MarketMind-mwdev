"""Resolve P2-PANEL supervision targets from merged panel frames."""

from __future__ import annotations

import pandas as pd

from pysrc.pipeline.contracts.p2 import P2Config, PanelModelTarget
from pysrc.pipeline.panel.indicator_universe_builder import PANEL_TARGET_OPTIONS

CANONICAL_PANEL_KEYS: tuple[str, ...] = ("date", "instrument", "interval")
PREDICTION_OUTPUT_KEYS: tuple[str, ...] = (*CANONICAL_PANEL_KEYS, "model_id", "fold_id")

_APPROVED_FORWARD_TARGET_COLUMNS: frozenset[str] = frozenset(
    {
        "forward_return",
        "forward_return_horizon",
        "cost_adjusted_forward_return",
        "forward_return_cost_adjusted",
    }
)
_CONTEMPORANEOUS_TARGET_COLUMNS: frozenset[str] = frozenset({"adjusted_return_1d", "raw_return_1d"})

_FORWARD_LOOKING_CONFIG_TARGETS: frozenset[str] = frozenset(
    {
        "forward_return",
        "cost_adjusted_forward_return",
        "cross_sectional_forward_return_rank",
        "positive_forward_return_label",
        "portfolio_adjusted_advantage",
        "abstain_or_trade_label",
        "allocation_score",
    }
)

_TARGET_ALIASES: dict[PanelModelTarget, tuple[str, ...]] = {
    "forward_return": (
        "forward_return",
        "forward_return_horizon",
    ),
    "cost_adjusted_forward_return": (
        "cost_adjusted_forward_return",
        "forward_return_cost_adjusted",
    ),
    "cross_sectional_forward_return_rank": ("cross_sectional_forward_return_rank",),
    "positive_forward_return_label": ("positive_forward_return_label",),
    "portfolio_adjusted_advantage": ("portfolio_adjusted_advantage",),
    "abstain_or_trade_label": ("abstain_or_trade_label",),
    "allocation_score": ("allocation_score",),
}


def panel_target_aliases(config: P2Config) -> tuple[str, ...]:
    """Return schema column aliases for the configured panel target."""
    return _TARGET_ALIASES.get(config.panel_target, (str(config.panel_target),))


def is_approved_forward_target_column(column: str) -> bool:
    return column in _APPROVED_FORWARD_TARGET_COLUMNS


def is_forward_looking_config_target(config: P2Config) -> bool:
    return str(config.panel_target) in _FORWARD_LOOKING_CONFIG_TARGETS


def target_metadata_for(config: P2Config, manifest: dict[str, object]) -> dict[str, object]:
    """Merge manifest and config target metadata for the configured panel target."""
    target = str(config.panel_target)
    metadata: dict[str, object] = {}
    manifest_targets = manifest.get("target_metadata")
    if isinstance(manifest_targets, dict):
        raw = manifest_targets.get(target)
        if isinstance(raw, dict):
            metadata.update(raw)
    if config.panel_target_horizon_days is not None:
        metadata.setdefault("horizon_days", int(config.panel_target_horizon_days))
    return metadata


def resolve_panel_target_column(frame: pd.DataFrame, config: P2Config) -> str:
    """Return the resolved target column name; fail loudly if unavailable."""
    target = config.panel_target
    if target not in PANEL_TARGET_OPTIONS:
        raise ValueError(f"Unsupported panel_target={target!r}")
    for candidate in panel_target_aliases(config):
        if candidate in frame.columns:
            return candidate
    available = sorted(frame.columns.astype(str).tolist())
    raise ValueError(
        f"Panel target {target!r} not found in merged panel frame. "
        f"Tried aliases={panel_target_aliases(config)}. Available columns sample={available[:20]}"
    )


def resolve_panel_target_from_schema(
    config: P2Config,
    schema: dict[str, str],
    manifest: dict[str, object],
) -> tuple[str, dict[str, object]]:
    """Resolve target column and metadata from parquet schema without loading the panel."""
    metadata = target_metadata_for(config, manifest)
    candidate = metadata.get("column")
    if isinstance(candidate, str) and candidate in schema:
        target_column = candidate
    else:
        target_column = ""
        for alias in panel_target_aliases(config):
            if alias in schema:
                target_column = alias
                break
        if not target_column:
            raise ValueError(
                f"Panel target {config.panel_target!r} not found in schema. "
                f"Tried aliases={panel_target_aliases(config)}."
            )
    metadata = {**metadata, "column": target_column}
    if is_forward_looking_config_target(config) and "horizon_days" not in metadata:
        raise ValueError(
            f"Forward-looking panel target {config.panel_target!r} requires target_horizon_days "
            "metadata to build leakage-safe purged folds."
        )
    return target_column, metadata


def validate_target_lineage(
    *,
    manifest: dict[str, object],
    target_column: str,
    target_metadata: dict[str, object],
) -> list[str]:
    """Return lineage failures when supervision provenance is incomplete."""
    failures: list[str] = []
    manifest_targets = manifest.get("target_metadata")
    supervision_columns = manifest.get("supervision_columns")
    if isinstance(supervision_columns, list) and target_column in supervision_columns:
        if "horizon_days" not in target_metadata:
            failures.append("target_horizon_days_missing_from_lineage")
        return failures
    if not isinstance(manifest_targets, dict) or not manifest_targets:
        failures.append("target_metadata_missing_from_manifest")
    else:
        documented = {
            str(value.get("column"))
            for value in manifest_targets.values()
            if isinstance(value, dict) and value.get("column")
        }
        if target_column not in documented:
            failures.append("resolved_target_not_documented_in_manifest_lineage")
    if "horizon_days" not in target_metadata:
        failures.append("target_horizon_days_missing_from_lineage")
    return failures


__all__ = [
    "CANONICAL_PANEL_KEYS",
    "PREDICTION_OUTPUT_KEYS",
    "_APPROVED_FORWARD_TARGET_COLUMNS",
    "is_approved_forward_target_column",
    "is_forward_looking_config_target",
    "panel_target_aliases",
    "resolve_panel_target_column",
    "resolve_panel_target_from_schema",
    "target_metadata_for",
    "validate_target_lineage",
]
