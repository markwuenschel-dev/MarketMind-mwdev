"""SlippageMonitor: tracks execution slippage between signal and fill prices."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SlippageReport:
    """Per-bar slippage statistics."""

    mean_bps: float
    max_bps: float
    p95_bps: float


class SlippageMonitor:
    """Computes slippage statistics from signal and fill price series."""

    def report(
        self,
        signal_prices: pd.Series,
        fill_prices: pd.Series,
    ) -> SlippageReport:
        """Return a SlippageReport comparing signal prices to actual fills."""
        slippage_bps = ((fill_prices - signal_prices) / signal_prices.abs()).abs() * 10_000
        return SlippageReport(
            mean_bps=float(slippage_bps.mean()),
            max_bps=float(slippage_bps.max()),
            p95_bps=float(slippage_bps.quantile(0.95)),
        )


__all__ = ["SlippageReport", "SlippageMonitor"]
