"""Minimal 2-expert mixture-of-experts gate."""

from __future__ import annotations

import json

import pandas as pd

from pysrc.contracts.meta_router import META_ROUTER_DECISION_PANEL_COLUMNS
from pysrc.meta.gating_network import neural_gate_decisions


def mixture_of_experts_decisions(
    frame: pd.DataFrame,
    *,
    gate_id: str = "mixture_of_experts",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Blend two neural expert gates with fixed 0.5/0.5 gating for smoke."""

    expert_a = neural_gate_decisions(frame, gate_id=f"{gate_id}_expert_a", random_seed=random_seed)
    expert_b = neural_gate_decisions(
        frame, gate_id=f"{gate_id}_expert_b", random_seed=random_seed + 1
    )
    merged = expert_a.merge(
        expert_b,
        on=["date", "split"],
        suffixes=("_a", "_b"),
    )
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        wa = json.loads(str(row["model_weights_json_a"]))
        wb = json.loads(str(row["model_weights_json_b"]))
        keys = sorted(set(wa) | set(wb))
        blended = {k: 0.5 * wa.get(k, 0.0) + 0.5 * wb.get(k, 0.0) for k in keys}
        chosen = max(blended, key=lambda k: blended[k])
        rows.append(
            {
                "date": row["date"],
                "fold_id": "all",
                "split": row["split"],
                "gate_id": gate_id,
                "selected_candidate_id": chosen,
                "exposure_scale": 0.5 * float(row["exposure_scale_a"])
                + 0.5 * float(row["exposure_scale_b"]),
                "abstain_probability": 0.5 * float(row["abstain_probability_a"])
                + 0.5 * float(row["abstain_probability_b"]),
                "action": "route",
                "model_weights_json": json.dumps(blended),
            }
        )
    out = pd.DataFrame(rows)
    for col in META_ROUTER_DECISION_PANEL_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[list(META_ROUTER_DECISION_PANEL_COLUMNS)]


__all__ = ["mixture_of_experts_decisions"]
