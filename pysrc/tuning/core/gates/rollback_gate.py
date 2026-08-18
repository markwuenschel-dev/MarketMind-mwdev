"""Rollback gate: determine whether a live strategy should be rolled back."""

from __future__ import annotations


def should_rollback(
    live_sharpe: float,
    baseline_sharpe: float,
    min_relative_ratio: float = 0.5,
    consecutive_drawdown_bars: int = 0,
    max_consecutive_drawdown: int = 10,
) -> tuple[bool, str]:
    """Return (should_rollback, reason)."""
    if live_sharpe < baseline_sharpe * min_relative_ratio:
        return True, (
            f"live_sharpe {live_sharpe:.3f} < {min_relative_ratio:.0%} "
            f"of baseline {baseline_sharpe:.3f}"
        )
    if consecutive_drawdown_bars > max_consecutive_drawdown:
        return True, (
            f"consecutive drawdown bars {consecutive_drawdown_bars} "
            f"> max {max_consecutive_drawdown}"
        )
    return False, "rollback_gate_ok"


__all__ = ["should_rollback"]
