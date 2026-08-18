from __future__ import annotations

from collections.abc import Iterable

from .types import PairsColumns


def pairs_zscore_column(cols: PairsColumns, window: int) -> str:
    """Keep the public feature-name contract explicit for the live pairs path."""

    return f"{cols.spread}_z{window}"


def require_pairs_feature_columns(
    available_columns: Iterable[str],
    *,
    cols: PairsColumns,
    zscore_window: int,
) -> None:
    """Fail closed when the live pairs slice is materialized with missing features."""

    available = set(available_columns)
    required = (
        cols.spread,
        pairs_zscore_column(cols, zscore_window),
        cols.half_life,
    )
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(
            "Required feature columns missing from stat-arb pairs frame: "
            + ", ".join(sorted(missing))
        )
