from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from pysrc.ops.hashing import canonicalize_json_bytes
from pysrc.strategies.momentum.alpha_ir import AlphaIR
from pysrc.strategies.momentum.exceptions import SerializationError


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    strategy: str = "momentum"
    generated_at: str | None = None
    bundle_dir: str | None = None


def _safe_float(value: float) -> float | None:
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _asset_weights(signal: pd.Series) -> list[dict[str, Any]]:
    if isinstance(signal.index, pd.MultiIndex) and signal.index.nlevels >= 2:
        ordered = signal.groupby(level=signal.index.nlevels - 1, sort=False).last().sort_index()
    else:
        ordered = signal.sort_index()

    payload: list[dict[str, Any]] = []
    for asset_id, weight in ordered.items():
        payload.append(
            {
                "asset_id": str(asset_id),
                "weight": _safe_float(weight),
            }
        )
    return payload


def build_signal_card_payload(alpha_ir: AlphaIR, run_meta: RunMeta) -> dict[str, Any]:
    if alpha_ir.pit_provenance is None:
        raise SerializationError(
            "Momentum signal_card serialization requires non-null pit_provenance."
        )

    generated_at = run_meta.generated_at or datetime.now(UTC).isoformat()
    payload = {
        "schema_version": "v1",
        "strategy": run_meta.strategy,
        "run_meta": {
            **asdict(run_meta),
            "generated_at": generated_at,
        },
        "variant": alpha_ir.variant,
        "information_coefficient": alpha_ir.information_coefficient,
        "weighted_positions": _asset_weights(alpha_ir.signal),
        "realized_vol": None
        if alpha_ir.realized_vol is None
        else [_safe_float(value) for value in alpha_ir.realized_vol.tolist()],
        "task_embedding": {
            "status": "stub",
            "values": [float(value) for value in alpha_ir.task_embedding.tolist()],
        },
        "pit_provenance": asdict(alpha_ir.pit_provenance),
        "diagnostics": dict(alpha_ir.diagnostics),
    }
    content_hash = hashlib.sha256(canonicalize_json_bytes(payload)).hexdigest()
    payload["content_hash"] = f"attest.v1:jcs-sha256:{content_hash}"
    return payload
