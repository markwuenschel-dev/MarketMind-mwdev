"""DashboardConfig: declarative configuration for Grafana/internal dashboard exports."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DashboardPanel:
    """A single panel definition within a dashboard."""

    title: str
    metric: str
    panel_type: str = "timeseries"
    unit: str = ""


@dataclass(frozen=True)
class DashboardConfig:
    """Declarative configuration for a monitoring dashboard."""

    job_id: str
    title: str
    panels: tuple[DashboardPanel, ...] = field(default_factory=tuple)
    refresh_seconds: int = 30
    grafana_url: str = ""


__all__ = ["DashboardPanel", "DashboardConfig"]
