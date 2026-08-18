"""Alerts: structured alert emission for drift, gate failure, and system events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    """An immutable structured alert."""

    severity: AlertSeverity
    title: str
    job_id: str
    details: dict[str, Any]
    emitted_at: datetime


def emit_alert(alert: Alert) -> None:
    """Emit an alert to the configured sink (stub: wire to mm_logkit or webhook)."""
    raise NotImplementedError(
        "emit_alert must be wired to mm_logkit structured logging or an alerting webhook"
    )


__all__ = ["AlertSeverity", "Alert", "emit_alert"]
