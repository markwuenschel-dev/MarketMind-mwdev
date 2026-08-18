"""Massive US Stocks SIP adjusted day panel — registered market_data source."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from pysrc.pipeline.stages.market_data.exceptions import DataFetchError
from pysrc.pipeline.stages.market_data.sources.contracts import DataSource
from pysrc.pipeline.stages.market_data.sources.market_data import register_source
from pysrc.pipeline.stages.market_data.sources.sip_panel import (
    SIP_ADJUSTED_PANEL_SOURCE,
    SipPanelConfig,
    build_sip_base_panel,
    load_sip_adjusted_panel,
)

SOURCE_ID = "sip_adjusted_panel"
DEFAULT_PANEL_ROOT = SIP_ADJUSTED_PANEL_SOURCE


@dataclass(frozen=True, slots=True)
class SipPanelLoadConfig:
    path: Path
    workers: int = 1
    apply_base_panel: bool = True
    loader_backend: str = "polars"


def resolve_sip_panel_config(config: Mapping[str, Any] | None) -> SipPanelLoadConfig:
    cfg = dict(config or {})
    return SipPanelLoadConfig(
        path=Path(str(cfg.get("path", DEFAULT_PANEL_ROOT))),
        workers=max(1, int(cfg.get("workers", 1))),
        apply_base_panel=bool(cfg.get("apply_base_panel", True)),
        loader_backend=str(cfg.get("loader_backend", "polars")),
    )


def load_sip_adjusted_panel_frame(config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Load the SIP adjusted day panel (optionally through W2 base-panel prep)."""

    resolved = resolve_sip_panel_config(config)
    panel = load_sip_adjusted_panel(
        resolved.path,
        workers=resolved.workers,
        loader_backend=resolved.loader_backend,  # type: ignore[arg-type]
    )
    if resolved.apply_base_panel:
        # Canonical indicator panel manifest declares horizon_days=1; align base prep.
        return build_sip_base_panel(panel, SipPanelConfig(horizon=1), copy=False)
    return panel


def _filter_panel(
    panel: pd.DataFrame,
    *,
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    out = panel
    sym = str(symbol).strip()
    if sym and sym not in {"*", "__all__"}:
        key = "instrument" if "instrument" in out.columns else "symbol"
        if key in out.columns:
            out = out.loc[out[key].astype(str) == sym]

    date_col = "date" if "date" in out.columns else None
    if date_col is None:
        return out

    start_dt = pd.Timestamp(str(start)[:10])
    end_dt = pd.Timestamp(str(end)[:10])
    dates = pd.to_datetime(out[date_col], errors="coerce")
    mask = (dates >= start_dt) & (dates <= end_dt)
    return out.loc[mask].copy()


def _panel_to_polars(panel: pd.DataFrame, *, eager: bool) -> pl.LazyFrame | pl.DataFrame:
    frame = pl.from_pandas(panel)
    return frame if eager else frame.lazy()


@register_source(SOURCE_ID)
class SipAdjustedPanelSource(DataSource):
    """Panel-wide SIP source; symbol ``*`` returns the full filtered date range."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__(dict(config or {}))
        self._load_config = resolve_sip_panel_config(self.config)
        self._panel_cache: pd.DataFrame | None = None

    def _cached_panel(self) -> pd.DataFrame:
        if self._panel_cache is None:
            self._panel_cache = load_sip_adjusted_panel_frame(self.config)
        return self._panel_cache

    async def get_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        eager: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        panel = _filter_panel(self._cached_panel(), symbol=symbol, start=start, end=end)
        if panel.empty:
            raise DataFetchError(
                f"No SIP panel rows for symbol={symbol!r} between {start} and {end}"
            )
        return _panel_to_polars(panel, eager=eager)

    async def get_realtime(
        self,
        symbol: str,
        *,
        interval: float = 60.0,
    ) -> AsyncIterator[pl.DataFrame]:
        raise NotImplementedError("SIP adjusted panel is daily historical only")

    def panel_date_bounds(self) -> tuple[datetime, datetime]:
        panel = self._cached_panel()
        dates = pd.to_datetime(panel["date"], errors="coerce").dropna()
        if dates.empty:
            raise DataFetchError("SIP panel has no parseable dates")
        return dates.min().to_pydatetime(), dates.max().to_pydatetime()


__all__ = [
    "DEFAULT_PANEL_ROOT",
    "SOURCE_ID",
    "SipAdjustedPanelSource",
    "SipPanelLoadConfig",
    "load_sip_adjusted_panel_frame",
    "resolve_sip_panel_config",
]
