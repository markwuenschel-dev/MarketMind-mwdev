"""Train-only numeric encoding for full P2-PANEL indicator universes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class PanelFeatureEncoder:
    feature_names: tuple[str, ...]
    fill_values: dict[str, float]
    scaler: StandardScaler


def fit_panel_feature_encoder(
    train_frame: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> PanelFeatureEncoder:
    """Fit median imputation and standard scaling on train rows only."""

    fill_values: dict[str, float] = {}
    matrix = np.empty((len(train_frame), len(feature_names)), dtype=np.float64)
    for idx, name in enumerate(feature_names):
        series = pd.to_numeric(train_frame[name], errors="coerce")
        fill = float(series.median()) if series.notna().any() else 0.0
        fill_values[name] = fill
        matrix[:, idx] = series.fillna(fill).to_numpy(dtype=np.float64, copy=False)

    scaler = StandardScaler()
    scaler.fit(matrix)
    return PanelFeatureEncoder(
        feature_names=feature_names,
        fill_values=fill_values,
        scaler=scaler,
    )


def transform_panel_feature_matrix(
    encoder: PanelFeatureEncoder,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Transform panel rows to model input matrix using frozen train statistics."""

    matrix = np.empty((len(frame), len(encoder.feature_names)), dtype=np.float64)
    for idx, name in enumerate(encoder.feature_names):
        series = pd.to_numeric(frame[name], errors="coerce")
        matrix[:, idx] = series.fillna(encoder.fill_values[name]).to_numpy(
            dtype=np.float64,
            copy=False,
        )
    transformed = encoder.scaler.transform(matrix)
    return np.asarray(transformed, dtype=np.float64)
