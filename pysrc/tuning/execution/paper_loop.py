"""Paper trading loop skeleton: bar cycle with reconciliation and kill switch (no live IBKR by default)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from pysrc.artifact_registry._atomic import atomic_write_json
from pysrc.pipeline.candidate_portfolios.production_bridge import (
    load_model_predictions,
    predictions_to_trade_intent_envelope,
)
from pysrc.strategies.prediction_threshold import (
    load_promotion_bundle_runtime,
    runtime_to_portfolio_spec,
)
from pysrc.tuning.execution.kill_switch import KillSwitchState
from pysrc.tuning.execution.reconciliation import compare_ledger_to_broker
from pysrc.tuning.live.inference_adapter import InferenceAdapter

_MAX_PAPER_BARS = 4680  # 60 trading days × 78 five-minute bars
_SCHEMA_VERSION = "paper_loop_log.v1"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def paper_trading_enabled() -> bool:
    """Return True when ``PAPER_TRADING_ENABLED`` permits live broker submission."""

    raw = os.environ.get("PAPER_TRADING_ENABLED", "0").strip().lower()
    return raw in _TRUTHY


def _validate_n_bars(n_bars: int) -> int:
    if n_bars < 1:
        raise ValueError(f"n_bars must be >= 1, got {n_bars}")
    if n_bars > _MAX_PAPER_BARS:
        raise ValueError(f"n_bars must be <= {_MAX_PAPER_BARS}, got {n_bars}")
    return n_bars


def _bar_timestamp(bar_date: str, bar_index: int) -> str:
    """Synthetic 5-minute bar close timestamp for dry-run reconciliation keys."""

    return f"{bar_date}T16:{bar_index % 60:02d}:00Z"


def _ledger_from_intents(
    ledger: dict[str, float],
    cash: float,
    intents: pd.DataFrame,
    *,
    notional_per_intent: float = 100.0,
) -> tuple[dict[str, float], float]:
    """Apply dry-run fills: one unit per intent at fixed notional (internal ledger only)."""

    if intents.empty:
        return ledger, cash
    updated = dict(ledger)
    spent = 0.0
    for instrument in intents["instrument"].astype(str):
        updated[instrument] = updated.get(instrument, 0.0) + 1.0
        spent += notional_per_intent
    return updated, cash - spent


def _bar_record(
    *,
    bar_index: int,
    bar_date: str,
    intents: pd.DataFrame,
    reconciliation: Mapping[str, Any],
    kill_switch: KillSwitchState,
    orders_blocked: bool,
    submitted_count: int,
) -> dict[str, Any]:
    return {
        "bar_index": bar_index,
        "as_of_bar": _bar_timestamp(bar_date, bar_index),
        "date": bar_date,
        "intent_count": int(len(intents)),
        "orders_submitted": submitted_count,
        "orders_blocked": orders_blocked,
        "kill_switch": {
            "block_new_orders": kill_switch.block_new_orders,
            "reason": kill_switch.reason,
        },
        "reconciliation": dict(reconciliation),
        "lineage": {
            "source": "model_prediction_panel",
            "pit_front_door": "pysrc.data.dataview.DataView",
            "inference_adapter": "pysrc.tuning.live.inference_adapter",
        },
    }


def paper_loop_dry_run(
    *,
    bundle_dir: Path,
    run_dir: Path,
    n_bars: int = 5,
) -> dict[str, Any]:
    """Simulate the paper-loop bar cycle without IBKR submission when ``PAPER_TRADING_ENABLED=0``."""

    n_bars = _validate_n_bars(n_bars)
    bundle_dir = Path(bundle_dir)
    run_dir = Path(run_dir)
    live_submit = paper_trading_enabled()

    runtime = load_promotion_bundle_runtime(bundle_dir)
    portfolio_spec = runtime_to_portfolio_spec(runtime)
    adapter = InferenceAdapter()
    model_handle, feature_names = adapter.load_xgboost_from_run(run_dir)

    predictions = load_model_predictions(run_dir, runtime.model_id)
    dates = sorted(predictions["date"].astype(str).unique().tolist())[-n_bars:]
    window = predictions.loc[predictions["date"].astype(str).isin(dates)].copy()

    kill_switch = KillSwitchState()
    ledger_positions: dict[str, float] = {}
    ledger_cash = 10_000.0
    broker_positions: dict[str, float] = {}
    broker_cash = 10_000.0

    bars: list[dict[str, Any]] = []
    total_submitted = 0

    for bar_index, bar_date in enumerate(dates):
        bar_preds = window.loc[window["date"].astype(str) == bar_date]
        intents = predictions_to_trade_intent_envelope(
            bar_preds,
            strategy_id=runtime.strategy_id,
            source_product_id="model_prediction_panel",
            lineage={"run_id": run_dir.name, "model_id": runtime.model_id},
            top_k=portfolio_spec.top_k,
            source_model_id=runtime.model_id,
        )

        if feature_names and not bar_preds.empty:
            sample = bar_preds.iloc[0]
            feature_row = {
                name: float(sample[name]) if name in sample.index else 0.0 for name in feature_names
            }
            adapter.predict(feature_row, model_handle, feature_names=feature_names)

        diff = compare_ledger_to_broker(
            ledger_positions=ledger_positions,
            broker_positions=broker_positions,
            ledger_cash=ledger_cash,
            broker_cash=broker_cash,
            as_of_bar=_bar_timestamp(bar_date, bar_index),
        )
        if diff.has_mismatch:
            kill_switch.engage(f"reconciliation mismatch at {diff.as_of_bar}")

        orders_blocked = not kill_switch.allows_new_orders()
        submitted_count = 0
        if not orders_blocked and not intents.empty:
            if live_submit:
                raise RuntimeError(
                    "PAPER_TRADING_ENABLED=1 requires Kotlin execution service (ADR-006); "
                    "Python shell does not submit to IBKR"
                )
            ledger_positions, ledger_cash = _ledger_from_intents(
                ledger_positions, ledger_cash, intents
            )
            broker_positions = dict(ledger_positions)
            broker_cash = ledger_cash
            submitted_count = int(len(intents))
            total_submitted += submitted_count

        bars.append(
            _bar_record(
                bar_index=bar_index,
                bar_date=bar_date,
                intents=intents,
                reconciliation=diff.to_dict(),
                kill_switch=kill_switch,
                orders_blocked=orders_blocked,
                submitted_count=submitted_count,
            )
        )

    out_dir = run_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paper_loop_log.json"
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "bundle_dir": str(bundle_dir),
        "run_id": run_dir.name,
        "strategy_id": runtime.strategy_id,
        "model_id": runtime.model_id,
        "n_bars": len(dates),
        "paper_trading_enabled": live_submit,
        "ibkr_submit": False,
        "total_orders_submitted": total_submitted,
        "kill_switch_engaged": kill_switch.block_new_orders,
        "bars": bars,
    }
    atomic_write_json(out_path, payload)
    payload["report_path"] = str(out_path)
    return payload


__all__ = [
    "_MAX_PAPER_BARS",
    "paper_loop_dry_run",
    "paper_trading_enabled",
]
