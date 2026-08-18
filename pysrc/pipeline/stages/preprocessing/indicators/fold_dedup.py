"""Collapse walk-forward fold duplicates in slim indicator artifacts.

Relocated from the retired W3-B allocator-benchmark lane; consumed by the panel
indicator-universe builder when reading slim ``indicator_features.parquet`` that
persisted one row per (date, instrument) per fold without a ``fold_id`` column.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def collapse_slim_indicator_fold_duplicates(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] = ("date", "instrument"),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Collapse walk-forward fold copies written without ``fold_id``.

    ``indicator_features.parquet`` is emitted from folded panel rows but only
    persists ``date``, ``instrument``, and indicators. Each calendar row appears
    once per fold, so the slim artifact has two rows per key. Keep the row with
    the most populated indicator values (fold warmup rows are typically all NaN).
    """

    keys = list(key_columns)
    rows_before = int(len(frame))
    duplicate_mask = frame.duplicated(keys, keep=False)
    if not bool(duplicate_mask.any()):
        return frame, {
            "applied": False,
            "rows_before": rows_before,
            "rows_after": rows_before,
            "rows_removed": 0,
            "reason": "no_duplicate_keys",
        }

    feature_columns = [column for column in frame.columns if column not in keys]
    scored = frame.assign(
        _fold_non_null_score=frame[feature_columns].notna().sum(axis=1).astype("int64")
    )
    ascending = [True] * len(keys) + [False]
    scored = scored.sort_values(
        [*keys, "_fold_non_null_score"], ascending=ascending, kind="mergesort"
    )
    collapsed = scored.drop(columns=["_fold_non_null_score"]).drop_duplicates(keys, keep="first")
    rows_after = int(len(collapsed))
    return collapsed, {
        "applied": True,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "reason": "walk_forward_fold_rows_missing_fold_id_in_slim_parquet",
    }
