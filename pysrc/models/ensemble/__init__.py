"""Thin ensemble stub emitting macro_state_panel columns (research fixture)."""

from __future__ import annotations

import pandas as pd

from pysrc.contracts.meta_router import MACRO_STATE_PANEL_COLUMNS
from pysrc.pipeline.products import load_macro_state_panel_fixture


def emit_macro_state_panel(*, n_days: int = 5) -> pd.DataFrame:
    """Return a synthetic macro state panel for channel tests."""

    return load_macro_state_panel_fixture(n_days=n_days)


def validate_macro_state_panel(frame: pd.DataFrame) -> None:
    missing = [c for c in MACRO_STATE_PANEL_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"macro_state_panel missing columns: {missing}")


__all__ = ["emit_macro_state_panel", "validate_macro_state_panel"]
