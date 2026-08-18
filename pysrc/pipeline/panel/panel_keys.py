"""Canonical normalization for panel date/instrument/interval keys."""

from __future__ import annotations

import pandas as pd

CANONICAL_DAILY_INTERVAL = "1d"
_DAILY_INTERVAL_ALIASES = frozenset({"daily", "1d", "d", "day"})


def _normalize_date_labels(dates: object) -> pd.Series:
    from pysrc.pipeline.panel.train_model_matrix import normalize_date_labels

    return normalize_date_labels(dates)


def normalize_panel_date(value: object) -> str:
    """Normalize one panel date label to ``YYYY-MM-DD``."""
    normalized = _normalize_date_labels([value])
    return str(normalized.iloc[0])


def normalize_panel_interval(value: object) -> str:
    """Map daily interval aliases to the canonical ``1d`` token."""
    token = str(value).strip().lower()
    if token in _DAILY_INTERVAL_ALIASES:
        return CANONICAL_DAILY_INTERVAL
    return str(value).strip()


def normalize_key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize date, instrument, and interval columns for panel joins."""
    out = frame.copy()
    if "date" in out.columns:
        out["date"] = _normalize_date_labels(out["date"])
    if "instrument" in out.columns:
        out["instrument"] = out["instrument"].astype(str)
    if "interval" not in out.columns:
        out["interval"] = CANONICAL_DAILY_INTERVAL
    else:
        out["interval"] = out["interval"].map(normalize_panel_interval)
    return out


__all__ = [
    "CANONICAL_DAILY_INTERVAL",
    "normalize_key_columns",
    "normalize_panel_date",
    "normalize_panel_interval",
]
