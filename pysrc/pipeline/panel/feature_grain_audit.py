"""Row-grain validation for ticker x date x interval panel inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

ROW_GRAIN: Final[str] = "ticker_date_interval"
TICKER_DATE_INTERVAL_KEYS: Final[tuple[str, ...]] = ("date", "instrument", "interval")
REQUIRED_KEY_COLUMNS: Final[tuple[str, ...]] = ("date", "instrument")
DUPLICATE_KEY_SAMPLE_FILENAME: Final[str] = "duplicate_key_sample.csv"


@dataclass(frozen=True, slots=True)
class SourceGrainReport:
    source_id: str
    valid: bool
    row_count: int
    duplicate_key_count: int
    key_columns_used: tuple[str, ...]
    exclusion_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "valid": self.valid,
            "row_count": self.row_count,
            "duplicate_key_count": self.duplicate_key_count,
            "key_columns_used": list(self.key_columns_used),
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True, slots=True)
class PanelGrainAudit:
    row_grain: str
    valid: bool
    ticker_count: int
    date_start: str | None
    date_end: str | None
    intervals_detected: tuple[str, ...]
    duplicate_key_count: int
    missing_key_columns: tuple[str, ...]
    row_count: int
    key_columns_used: tuple[str, ...]
    per_source_duplicate_key_count: dict[str, int]
    duplicate_key_sample_path: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "row_grain": self.row_grain,
            "valid": self.valid,
            "ticker_count": self.ticker_count,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "intervals_detected": list(self.intervals_detected),
            "duplicate_key_count": self.duplicate_key_count,
            "missing_key_columns": list(self.missing_key_columns),
            "row_count": self.row_count,
            "key_columns_used": list(self.key_columns_used),
            "per_source_duplicate_key_count": dict(self.per_source_duplicate_key_count),
            "duplicate_key_sample_path": self.duplicate_key_sample_path,
        }


def normalize_panel_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize date/instrument/interval for panel grain checks and merges."""

    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=False).dt.strftime(
            "%Y-%m-%d"
        )
    if "instrument" not in out.columns and "ticker" in out.columns:
        out["instrument"] = out["ticker"].astype(str)
    elif "instrument" in out.columns:
        out["instrument"] = out["instrument"].astype(str)
    if "interval" not in out.columns:
        out["interval"] = "daily"
    else:
        out["interval"] = out["interval"].astype(str)
    return out


def _resolve_instrument_column(frame: pd.DataFrame) -> str:
    if "instrument" in frame.columns:
        return "instrument"
    if "ticker" in frame.columns:
        return "ticker"
    if "symbol" in frame.columns:
        return "symbol"
    raise ValueError("Panel frame must include instrument, ticker, or symbol column.")


def _missing_key_columns(frame: pd.DataFrame, key_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(column for column in key_columns if column not in frame.columns)


def _duplicate_key_count(frame: pd.DataFrame, key_columns: tuple[str, ...]) -> int:
    missing = _missing_key_columns(frame, key_columns)
    if missing:
        return 0
    return int(frame.duplicated(list(key_columns)).sum())


def validate_source_grain(frame: pd.DataFrame, source_id: str) -> SourceGrainReport:
    """Validate one source at canonical ticker-date-interval grain."""

    normalized = normalize_panel_keys(frame)
    key_columns = TICKER_DATE_INTERVAL_KEYS
    missing = _missing_key_columns(normalized, key_columns)
    duplicate_count = _duplicate_key_count(normalized, key_columns)
    valid = not missing and duplicate_count == 0 and len(normalized) > 0
    return SourceGrainReport(
        source_id=source_id,
        valid=valid,
        row_count=int(len(normalized)),
        duplicate_key_count=duplicate_count,
        key_columns_used=key_columns,
        exclusion_reason=None
        if valid
        else (f"missing_keys={list(missing)}" if missing else f"duplicate_keys={duplicate_count}"),
    )


def build_duplicate_key_sample(
    frame: pd.DataFrame,
    key_columns: tuple[str, ...],
    *,
    limit: int = 500,
) -> pd.DataFrame:
    """Sample rows participating in duplicate canonical keys."""

    normalized = normalize_panel_keys(frame)
    missing = _missing_key_columns(normalized, key_columns)
    if missing:
        return pd.DataFrame(columns=[*key_columns, "row_index"])

    keys = list(key_columns)
    duplicated_mask = normalized.duplicated(keys, keep=False)
    sample = normalized.loc[duplicated_mask, keys].copy()
    sample["row_index"] = sample.index.astype(int)
    return sample.sort_values(keys, kind="mergesort").head(limit).reset_index(drop=True)


def audit_panel_grain(
    frame: pd.DataFrame,
    *,
    per_source_duplicate_key_count: dict[str, int] | None = None,
    duplicate_key_sample_path: str | None = None,
) -> PanelGrainAudit:
    """Validate merged panel grain on fixed ticker-date-interval keys only."""

    key_columns = TICKER_DATE_INTERVAL_KEYS
    normalized = normalize_panel_keys(frame)
    missing = _missing_key_columns(normalized, key_columns)
    instrument_col = _resolve_instrument_column(normalized) if not missing else "instrument"
    duplicate_key_count = _duplicate_key_count(normalized, key_columns)

    dates = (
        pd.to_datetime(normalized["date"], errors="coerce", utc=False)
        if "date" in normalized.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    date_start = dates.min()
    date_end = dates.max()
    intervals = (
        tuple(sorted(normalized["interval"].astype(str).unique().tolist()))
        if "interval" in normalized.columns
        else ("daily",)
    )
    tickers = (
        int(normalized[instrument_col].astype(str).nunique())
        if instrument_col in normalized.columns
        else 0
    )

    valid = not missing and duplicate_key_count == 0 and tickers > 0
    return PanelGrainAudit(
        row_grain=ROW_GRAIN,
        valid=valid,
        ticker_count=tickers,
        date_start=date_start.strftime("%Y-%m-%d") if pd.notna(date_start) else None,
        date_end=date_end.strftime("%Y-%m-%d") if pd.notna(date_end) else None,
        intervals_detected=intervals,
        duplicate_key_count=duplicate_key_count,
        missing_key_columns=missing,
        row_count=int(len(normalized)),
        key_columns_used=key_columns,
        per_source_duplicate_key_count=dict(per_source_duplicate_key_count or {}),
        duplicate_key_sample_path=duplicate_key_sample_path,
    )
