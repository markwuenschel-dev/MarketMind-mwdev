"""Robust winsorization and scaling for W3-B indicator diagnostics and scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import SupportsFloat, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class IndicatorRobustScale:
    indicator_id: str
    median: float
    iqr: float
    mad: float
    lower_clip: float
    upper_clip: float
    scale_denominator: float
    train_validation_fit_only: bool
    iqr_multiplier: float

    def as_metadata(self) -> dict[str, object]:
        return {
            "indicator_id": self.indicator_id,
            "median": self.median,
            "iqr": self.iqr,
            "mad": self.mad,
            "lower_clip": self.lower_clip,
            "upper_clip": self.upper_clip,
            "scale_denominator": self.scale_denominator,
            "train_validation_fit_only": self.train_validation_fit_only,
            "iqr_multiplier": self.iqr_multiplier,
        }


def sanitize_indicator_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def fit_robust_indicator_scales(
    rows: pd.DataFrame,
    indicator_columns: Sequence[str],
    *,
    iqr_multiplier: float = 3.0,
    min_iqr: float = 1e-6,
) -> dict[str, IndicatorRobustScale]:
    if "split" in rows.columns:
        tv_mask = rows["split"].astype(str).isin(["train", "validation"]).to_numpy()
    else:
        tv_mask = np.ones(len(rows), dtype=bool)

    scales: dict[str, IndicatorRobustScale] = {}
    for indicator_id in indicator_columns:
        if indicator_id not in rows.columns:
            continue
        tv_values = sanitize_indicator_series(rows.loc[tv_mask, indicator_id]).to_numpy(
            dtype=np.float64
        )
        scales[indicator_id] = _fit_one_scale(
            indicator_id,
            tv_values,
            iqr_multiplier=float(iqr_multiplier),
            min_iqr=float(min_iqr),
        )
    return scales


def apply_robust_indicator_scales(
    rows: pd.DataFrame,
    scales: Mapping[str, IndicatorRobustScale],
    *,
    copy: bool = False,
) -> pd.DataFrame:
    out = rows.copy(deep=False) if copy else rows
    for indicator_id, scale in scales.items():
        if indicator_id not in out.columns:
            continue
        sanitized = sanitize_indicator_series(out[indicator_id])
        clipped = sanitized.clip(lower=scale.lower_clip, upper=scale.upper_clip)
        out[indicator_id] = ((clipped - scale.median) / scale.scale_denominator).astype("float32")
    return out


def robust_scale_metadata_payload(
    scales: Mapping[str, IndicatorRobustScale],
) -> dict[str, dict[str, object]]:
    return {indicator_id: scale.as_metadata() for indicator_id, scale in scales.items()}


def _fit_one_scale(
    indicator_id: str,
    tv_values: np.ndarray,
    *,
    iqr_multiplier: float,
    min_iqr: float,
) -> IndicatorRobustScale:
    finite = tv_values[np.isfinite(tv_values)]
    if finite.size == 0:
        return IndicatorRobustScale(
            indicator_id=indicator_id,
            median=0.0,
            iqr=1.0,
            mad=1.0,
            lower_clip=-1.0,
            upper_clip=1.0,
            scale_denominator=max(1.0, min_iqr),
            train_validation_fit_only=True,
            iqr_multiplier=iqr_multiplier,
        )

    median = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25.0, 75.0])
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(finite - median)))
    if not np.isfinite(iqr) or iqr <= 0.0:
        iqr = mad if np.isfinite(mad) and mad > 0.0 else 1.0
    if not np.isfinite(mad) or mad <= 0.0:
        mad = iqr

    half_width = float(iqr_multiplier) * iqr
    lower_clip = median - half_width
    upper_clip = median + half_width
    scale_denominator = max(float(iqr), float(min_iqr))

    return IndicatorRobustScale(
        indicator_id=indicator_id,
        median=median,
        iqr=iqr,
        mad=mad,
        lower_clip=lower_clip,
        upper_clip=upper_clip,
        scale_denominator=scale_denominator,
        train_validation_fit_only=True,
        iqr_multiplier=float(iqr_multiplier),
    )


def _as_float(value: object) -> float:
    return float(cast(SupportsFloat, value))
