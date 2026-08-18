"""Diagnostics, orientation, and redundancy pruning for W3-B indicators."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import SupportsFloat, SupportsInt, cast

import numpy as np
import pandas as pd

from pysrc.pipeline.stages.preprocessing.indicators.schema import IndicatorClassification


@dataclass(frozen=True, slots=True)
class IndicatorDiagnosticsResult:
    diagnostics: dict[str, dict[str, object]]
    orientations: dict[str, int]
    active_indicators: tuple[str, ...]
    redundancy: dict[str, object]


def build_indicator_diagnostics(
    rows: pd.DataFrame,
    *,
    surface_id: str,
    indicator_columns: Sequence[str],
    redundancy_rank_corr_threshold: float,
    indicator_scales: Mapping[str, Mapping[str, object]] | None = None,
) -> IndicatorDiagnosticsResult:
    """Build train/validation-oriented diagnostics without test-set selection leakage."""

    diagnostics: dict[str, dict[str, object]] = {}
    orientations: dict[str, int] = {}
    tv_scores: dict[str, float] = {}
    for indicator_id in indicator_columns:
        scaling = indicator_scales.get(indicator_id) if indicator_scales is not None else None
        one = _diagnose_one(
            rows,
            indicator_id=indicator_id,
            surface_id=surface_id,
            scaling=scaling,
        )
        diagnostics[indicator_id] = one
        orientations[indicator_id] = _as_int(one["orientation"])
        tv_scores[indicator_id] = abs(_as_float(one["train_validation_net_utility"]))
    active_before_pruning = tuple(
        indicator_id
        for indicator_id, payload in diagnostics.items()
        if str(payload["classification"]) in {"KEEP", "INVERT", "REGIME_ONLY", "LIQUIDITY_ONLY"}
    )
    redundancy = prune_redundant_indicators(
        rows,
        indicator_columns=active_before_pruning,
        train_validation_scores=tv_scores,
        threshold=redundancy_rank_corr_threshold,
    )
    removed = set(cast_list(redundancy.get("removed_indicators")))
    active = tuple(
        indicator_id for indicator_id in active_before_pruning if indicator_id not in removed
    )
    return IndicatorDiagnosticsResult(
        diagnostics=diagnostics,
        orientations=orientations,
        active_indicators=active,
        redundancy=redundancy,
    )


def lag_outcome_diagnostic_by_horizon(
    rows: pd.DataFrame,
    *,
    value_column: str,
    horizon: int,
) -> pd.Series:
    """Lag outcome-derived diagnostics by the forecast horizon, not by a single bar."""

    if "instrument" not in rows.columns:
        raise ValueError("outcome diagnostic lag requires instrument column")
    if value_column not in rows.columns:
        raise ValueError(f"outcome diagnostic lag requires {value_column!r}")
    ordered = rows.sort_values(["instrument", "date"], kind="mergesort")
    lagged = ordered.groupby("instrument", sort=False)[value_column].shift(max(1, int(horizon)))
    return lagged.reindex(ordered.index).reindex(rows.index)


def prune_redundant_indicators(
    rows: pd.DataFrame,
    *,
    indicator_columns: Sequence[str],
    train_validation_scores: Mapping[str, float],
    threshold: float,
) -> dict[str, object]:
    tv = rows.loc[rows["split"].isin(["train", "validation"])] if "split" in rows.columns else rows
    decisions: list[dict[str, object]] = []
    removed: set[str] = set()
    columns = [column for column in indicator_columns if column in tv.columns]
    ranked = tv.loc[:, columns].rank(axis=0, method="average")
    corr = ranked.corr(method="pearson").fillna(0.0) if len(columns) >= 2 else pd.DataFrame()
    for left_index, left in enumerate(columns):
        if left in removed:
            continue
        for right in columns[left_index + 1 :]:
            if right in removed:
                continue
            value = _as_float(corr.at[left, right]) if not corr.empty else 0.0
            if abs(value) <= float(threshold):
                continue
            left_score = float(train_validation_scores.get(left, 0.0))
            right_score = float(train_validation_scores.get(right, 0.0))
            drop = right if left_score >= right_score else left
            keep = left if drop == right else right
            removed.add(drop)
            decisions.append(
                {
                    "indicator_a": left,
                    "indicator_b": right,
                    "rank_corr": value,
                    "kept_indicator": keep,
                    "dropped_indicator": drop,
                    "reason": "abs_train_validation_rank_corr_above_threshold",
                }
            )
    return {
        "schema_version": "w3_b.pandas_ta.indicator_redundancy.v1",
        "threshold": float(threshold),
        "decisions": decisions,
        "removed_indicators": sorted(removed),
    }


def diagnostics_summary_markdown(report: Mapping[str, object]) -> str:
    surfaces = cast_mapping(report.get("surfaces"))
    lines = [
        "# W3-B pandas-ta-classic Indicator Diagnostics",
        "",
        (
            "Indicator orientation, classification, and redundancy pruning use train+validation "
            "rows only."
        ),
        "",
    ]
    for surface_id, payload in surfaces.items():
        indicators = cast_mapping(cast_mapping(payload).get("indicators"))
        counts: dict[str, int] = {}
        for item in indicators.values():
            cls = str(cast_mapping(item).get("classification", "INCONCLUSIVE"))
            counts[cls] = counts.get(cls, 0) + 1
        lines.append(f"## {surface_id}")
        lines.append("")
        lines.append(
            ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
            or "No indicators."
        )
        lines.append("")
    return "\n".join(lines)


def _diagnose_one(
    rows: pd.DataFrame,
    *,
    indicator_id: str,
    surface_id: str,
    scaling: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if indicator_id not in rows.columns:
        return _empty_indicator_payload(indicator_id, surface_id, scaling=scaling)

    valid = rows[indicator_id].notna()
    valid_count = int(valid.sum())
    if valid_count == 0:
        return _empty_indicator_payload(indicator_id, surface_id, scaling=scaling)

    signal = pd.to_numeric(rows.loc[valid, indicator_id], errors="coerce").replace(
        [float("inf"), float("-inf")], np.nan
    )
    y = pd.to_numeric(rows.loc[valid, "forward_return_horizon"], errors="coerce")
    cost = pd.to_numeric(rows.loc[valid, "cost_estimate"], errors="coerce").fillna(0.0005)

    split: pd.Series | None = None
    if "split" in rows.columns:
        split = rows.loc[valid, "split"].astype(str)
        tv_mask = split.isin(["train", "validation"])
        test_count = int(split.eq("test").sum())
    else:
        tv_mask = pd.Series(True, index=signal.index)
        test_count = 0

    instrument = (
        rows.loc[valid, "instrument"].astype(str)
        if "instrument" in rows.columns
        else pd.Series(dtype=str, index=signal.index)
    )

    tv_signal = signal.loc[tv_mask]
    tv_y = y.loc[tv_mask]
    tv_cost = cost.loc[tv_mask]
    raw_tv = float((tv_signal * tv_y - tv_cost).mean()) if not tv_signal.empty else 0.0
    inv_tv = float(((-tv_signal) * tv_y - tv_cost).mean()) if not tv_signal.empty else 0.0
    orientation = -1 if inv_tv > raw_tv else 1
    oriented = signal * float(orientation)
    oriented_tv = oriented.loc[tv_mask]
    chosen_tv = max(raw_tv, inv_tv)
    utility = oriented * y - cost
    tv_utility = utility.loc[tv_mask]

    nonfinite_detected = False

    raw_tv_out, nf = _finite_metric(raw_tv)
    nonfinite_detected = nonfinite_detected or nf
    inv_tv_out, nf = _finite_metric(inv_tv)
    nonfinite_detected = nonfinite_detected or nf
    chosen_tv_out, nf = _finite_metric(chosen_tv)
    nonfinite_detected = nonfinite_detected or nf
    gross_sum = (oriented * y).sum()
    gross_out, nf = _finite_metric(gross_sum)
    nonfinite_detected = nonfinite_detected or nf
    net_sum = utility.sum()
    net_out, nf = _finite_metric(net_sum)
    nonfinite_detected = nonfinite_detected or nf

    regime_labels = (
        rows.loc[valid, "regime_id"].astype(str) if "regime_id" in rows.columns else None
    )
    price_labels = (
        rows.loc[valid, "price_bucket"].astype(str) if "price_bucket" in rows.columns else None
    )
    liquidity_labels = (
        rows.loc[valid, "liquidity_bucket"].astype(str)
        if "liquidity_bucket" in rows.columns
        else None
    )
    tv_regime_labels = regime_labels.loc[tv_mask] if regime_labels is not None else None
    tv_liquidity_labels = liquidity_labels.loc[tv_mask] if liquidity_labels is not None else None

    classification = _classify_indicator(
        chosen_train_validation_utility=chosen_tv_out,
        coverage=float(valid_count / len(rows)) if len(rows) else 0.0,
        utility_by_regime=_sum_by_labels(tv_regime_labels, tv_utility),
        utility_by_liquidity_bucket=_sum_by_labels(tv_liquidity_labels, tv_utility),
        orientation=orientation,
    )

    payload: dict[str, object] = {
        "surface_id": surface_id,
        "indicator_id": indicator_id,
        "classification": classification,
        "orientation": int(orientation),
        "orientation_scope": "train_validation_only",
        "raw_train_validation_net_utility": raw_tv_out,
        "inverted_train_validation_net_utility": inv_tv_out,
        "train_validation_net_utility": chosen_tv_out,
        "ic": _corr(oriented, y),
        "rank_ic": _corr(oriented.rank(method="average"), y.rank(method="average")),
        "hit_rate": float(((oriented * y) > 0.0).mean()),
        "gross_utility": gross_out,
        "net_utility": net_out,
        "turnover_proxy": float(oriented.groupby(instrument, sort=False).diff().abs().mean())
        if not instrument.empty
        else 0.0,
        "coverage": float(valid_count / len(rows)) if len(rows) else 0.0,
        "train_validation_test_degradation": _split_degradation(split, utility),
        "utility_by_regime": _sum_by_labels(regime_labels, utility),
        "utility_by_price_bucket": _sum_by_labels(price_labels, utility),
        "utility_by_liquidity_bucket": _sum_by_labels(liquidity_labels, utility),
        "top_1_instrument_utility_share": _top_share(instrument, utility, 1),
        "top_5_instrument_utility_share": _top_share(instrument, utility, 5),
        "train_validation_observation_count": int(tv_mask.sum()),
        "test_observation_count": test_count,
        "oriented_train_validation_mean_signal": float(oriented_tv.mean())
        if not oriented_tv.empty
        else 0.0,
        "nonfinite_metric_detected": nonfinite_detected,
    }
    if scaling is not None:
        payload["scaling"] = dict(scaling)
    return payload


def _classify_indicator(
    *,
    chosen_train_validation_utility: float,
    coverage: float,
    utility_by_regime: Mapping[str, float],
    utility_by_liquidity_bucket: Mapping[str, float],
    orientation: int,
) -> IndicatorClassification:
    if coverage <= 0.0:
        return "INCONCLUSIVE"
    if coverage < 0.10:
        return "TOO_UNSTABLE"
    if not math.isfinite(chosen_train_validation_utility) or chosen_train_validation_utility <= 0.0:
        return "DROP"
    if _dominant_share(utility_by_regime) > 0.85:
        return "REGIME_ONLY"
    if _dominant_share(utility_by_liquidity_bucket) > 0.85:
        return "LIQUIDITY_ONLY"
    return "INVERT" if orientation < 0 else "KEEP"


def _empty_indicator_payload(
    indicator_id: str,
    surface_id: str,
    *,
    scaling: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "surface_id": surface_id,
        "indicator_id": indicator_id,
        "classification": "INCONCLUSIVE",
        "orientation": 1,
        "orientation_scope": "train_validation_only",
        "raw_train_validation_net_utility": 0.0,
        "inverted_train_validation_net_utility": 0.0,
        "train_validation_net_utility": 0.0,
        "ic": 0.0,
        "rank_ic": 0.0,
        "hit_rate": 0.0,
        "gross_utility": 0.0,
        "net_utility": 0.0,
        "turnover_proxy": 0.0,
        "coverage": 0.0,
        "train_validation_test_degradation": 0.0,
        "utility_by_regime": {},
        "utility_by_price_bucket": {},
        "utility_by_liquidity_bucket": {},
        "top_1_instrument_utility_share": 0.0,
        "top_5_instrument_utility_share": 0.0,
        "train_validation_observation_count": 0,
        "test_observation_count": 0,
        "oriented_train_validation_mean_signal": 0.0,
        "nonfinite_metric_detected": False,
    }
    if scaling is not None:
        payload["scaling"] = dict(scaling)
    return payload


def _finite_metric(value: object) -> tuple[float, bool]:
    numeric = float(cast(SupportsFloat, value))
    if math.isfinite(numeric):
        return numeric, False
    return 0.0, True


def _split_degradation(split: pd.Series | None, utility: pd.Series) -> float:
    if split is None:
        return 0.0
    tv = utility.loc[split.isin(["train", "validation"])].mean()
    test = utility.loc[split.eq("test")].mean()
    if not math.isfinite(float(tv)) or not math.isfinite(float(test)):
        return 0.0
    return float(test - tv)


def _corr(left: pd.Series, right: pd.Series) -> float:
    mask = left.notna() & right.notna()
    if int(mask.sum()) < 3:
        return 0.0
    value = float(left.loc[mask].corr(right.loc[mask]))
    return value if math.isfinite(value) else 0.0


def _sum_by_labels(
    labels: pd.Series | None,
    values: pd.Series,
) -> dict[str, float]:
    if labels is None:
        return {}
    summed = values.groupby(labels.astype(str), sort=True).sum()
    return {str(key): float(value) for key, value in summed.items()}


def _top_share(instrument: pd.Series, values: pd.Series, count: int) -> float:
    if instrument.empty:
        return 0.0
    by_inst = values.groupby(instrument.astype(str), sort=True).sum().abs()
    total = float(by_inst.sum())
    if total <= 0.0 or not math.isfinite(total):
        return 0.0
    top_sum = float(by_inst.sort_values(ascending=False).head(count).sum())
    return top_sum / total


def _dominant_share(values: Mapping[str, float]) -> float:
    absolute = [abs(float(value)) for value in values.values() if math.isfinite(float(value))]
    total = float(sum(absolute))
    return float(max(absolute) / total) if total > 0.0 and absolute else 0.0


def cast_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def cast_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list | tuple) else []


def _as_float(value: object) -> float:
    return float(cast(SupportsFloat, value))


def _as_int(value: object) -> int:
    return int(cast(SupportsInt, value))
