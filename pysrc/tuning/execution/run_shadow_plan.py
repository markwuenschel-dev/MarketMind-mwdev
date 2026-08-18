"""Shadow replay plan: pseudo-live bar replay without broker submission."""

from __future__ import annotations

import json
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

_MAX_SHADOW_DAYS = 60
_SCHEMA_VERSION = "shadow_replay_log.v2"


def _validate_n_days(n_days: int) -> int:
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")
    if n_days > _MAX_SHADOW_DAYS:
        raise ValueError(f"n_days must be <= {_MAX_SHADOW_DAYS}, got {n_days}")
    return n_days


def _resolve_panel_path(bundle_dir: Path, panel_path: Path | None) -> Path:
    if panel_path is not None:
        return Path(panel_path)
    plan_path = Path(bundle_dir) / "plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        recorded = plan.get("panel_path")
        if recorded:
            return Path(str(recorded))
    return Path("data/processed/full_indicator_feature_panel/panel.parquet")


def _bar_provenance(
    *,
    bar_index: int,
    bar_date: str,
    predictions: pd.DataFrame,
    intents: pd.DataFrame,
) -> dict[str, Any]:
    bar_preds = predictions.loc[predictions["date"].astype(str) == bar_date]
    bar_intents = (
        intents.loc[intents["date"].astype(str) == bar_date] if not intents.empty else intents
    )
    instruments = sorted(bar_preds["instrument"].astype(str).unique().tolist())
    return {
        "bar_index": bar_index,
        "date": bar_date,
        "prediction_row_count": int(len(bar_preds)),
        "intent_count": int(len(bar_intents)),
        "instrument_count": len(instruments),
        "lineage": {
            "source": "model_prediction_panel",
            "pit_front_door": "pysrc.data.dataview.DataView",
        },
    }


def run_shadow_plan(
    *,
    bundle_dir: Path,
    run_dir: Path,
    panel_path: Path | None = None,
    n_days: int = 5,
) -> dict[str, Any]:
    """Replay the last ``n_days`` from the panel as pseudo-live bars; log intents only."""

    n_days = _validate_n_days(n_days)
    bundle_dir = Path(bundle_dir)
    run_dir = Path(run_dir)
    resolved_panel_path = _resolve_panel_path(bundle_dir, panel_path)

    runtime = load_promotion_bundle_runtime(bundle_dir)
    portfolio_spec = runtime_to_portfolio_spec(runtime)
    predictions = load_model_predictions(run_dir, runtime.model_id)
    dates = sorted(predictions["date"].astype(str).unique().tolist())[-n_days:]
    window = predictions.loc[predictions["date"].astype(str).isin(dates)].copy()
    intents = predictions_to_trade_intent_envelope(
        window,
        strategy_id=runtime.strategy_id,
        source_product_id="model_prediction_panel",
        lineage={"run_id": run_dir.name, "model_id": runtime.model_id},
        top_k=portfolio_spec.top_k,
        source_model_id=runtime.model_id,
    )
    bars = [
        _bar_provenance(
            bar_index=index,
            bar_date=bar_date,
            predictions=window,
            intents=intents,
        )
        for index, bar_date in enumerate(dates)
    ]
    out_dir = run_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shadow_replay_log.json"
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "bundle_dir": str(bundle_dir),
        "run_id": run_dir.name,
        "strategy_id": runtime.strategy_id,
        "model_id": runtime.model_id,
        "n_days": len(dates),
        "intent_row_count": int(len(intents)),
        "dates": dates,
        "bars": bars,
    }
    atomic_write_json(out_path, payload)
    payload["report_path"] = str(out_path)
    payload["panel_path"] = str(resolved_panel_path)
    return payload


__all__ = ["run_shadow_plan", "_MAX_SHADOW_DAYS"]
