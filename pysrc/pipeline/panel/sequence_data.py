"""Panel-specific sequence window materialization for train-model-matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequence_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    *,
    sequence_length: int,
    date_col: str = "date",
    instrument_col: str = "instrument",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build (n_samples, seq_len, n_features) windows sorted by instrument/date."""

    ordered = frame.sort_values([instrument_col, date_col])
    xs: list[np.ndarray] = []
    ys: list[float] = []
    meta_rows: list[dict[str, object]] = []
    for _instrument, group in ordered.groupby(instrument_col, sort=False):
        values = group[feature_columns].to_numpy(dtype=np.float64)
        targets = group[target_column].to_numpy(dtype=np.float64)
        dates = group[date_col].astype(str).tolist()
        for idx in range(sequence_length, len(group)):
            xs.append(values[idx - sequence_length : idx])
            ys.append(float(targets[idx]))
            meta_rows.append(
                {
                    date_col: dates[idx],
                    instrument_col: str(_instrument),
                    "sequence_length": sequence_length,
                }
            )
    if not xs:
        return (
            np.empty((0, sequence_length, len(feature_columns))),
            np.empty(0),
            pd.DataFrame(columns=[date_col, instrument_col, "sequence_length"]),
        )
    return np.stack(xs), np.asarray(ys), pd.DataFrame(meta_rows)


__all__ = ["build_sequence_windows"]
