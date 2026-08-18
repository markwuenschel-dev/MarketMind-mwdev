"""Walk-forward fold slices at ticker x date panel grain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.panel.indicator_universe_builder import PanelSupervisionFrame
from pysrc.pipeline.panel.panel_feature_encoder import (
    fit_panel_feature_encoder,
    transform_panel_feature_matrix,
)
from pysrc.pipeline.panel.panel_targets import resolve_panel_target_column

_META_COLUMNS = (
    "bundle_id",
    "surface_id",
    "fold_id",
    "split",
    "date",
    "instrument",
    "interval",
)


@dataclass(frozen=True, slots=True)
class PanelFoldSlice:
    fold_id: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_meta: pd.DataFrame


def build_panel_fold_slices(
    panel: PanelSupervisionFrame,
    config: P2Config,
) -> tuple[PanelFoldSlice, ...]:
    """Build fold slices using the full eligible feature universe."""

    frame = panel.frame
    feature_names = panel.feature_names
    if not feature_names:
        raise ValueError("Panel supervision frame has no eligible features.")

    target_column = resolve_panel_target_column(frame, config)
    if "split" not in frame.columns or "fold_id" not in frame.columns:
        train = frame
        encoder = fit_panel_feature_encoder(train, feature_names)
        y = (
            pd.to_numeric(train[target_column], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        meta_cols = [column for column in _META_COLUMNS if column in train.columns]
        return (
            PanelFoldSlice(
                fold_id="fold_0",
                x_train=transform_panel_feature_matrix(encoder, train),
                y_train=y.reshape(-1, 1),
                x_validation=np.empty((0, len(feature_names)), dtype=np.float64),
                y_validation=np.empty((0, 1), dtype=np.float64),
                x_test=np.empty((0, len(feature_names)), dtype=np.float64),
                y_test=np.empty((0, 1), dtype=np.float64),
                test_meta=train.loc[:, meta_cols].head(0).copy(),
            ),
        )

    split_col = frame["split"].astype(str)
    fold_col = frame["fold_id"].astype(str)
    slices: list[PanelFoldSlice] = []
    meta_cols = [column for column in _META_COLUMNS if column in frame.columns]

    for fold_id in sorted(fold_col.unique().tolist()):
        fold_mask = fold_col.eq(fold_id)
        train_mask = fold_mask & split_col.eq("train")
        validation_mask = fold_mask & split_col.eq("validation")
        test_mask = fold_mask & split_col.eq("test")
        if not train_mask.any() or not test_mask.any():
            continue

        train = frame.loc[train_mask]
        validation = frame.loc[validation_mask]
        test = frame.loc[test_mask]
        encoder = fit_panel_feature_encoder(train, feature_names)
        y_train = (
            pd.to_numeric(train[target_column], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            .reshape(-1, 1)
        )
        y_validation = (
            pd.to_numeric(validation[target_column], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            .reshape(-1, 1)
            if not validation.empty
            else np.empty((0, 1), dtype=np.float64)
        )
        y_test = (
            pd.to_numeric(test[target_column], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            .reshape(-1, 1)
        )
        slices.append(
            PanelFoldSlice(
                fold_id=fold_id,
                x_train=transform_panel_feature_matrix(encoder, train),
                y_train=y_train,
                x_validation=transform_panel_feature_matrix(encoder, validation)
                if not validation.empty
                else np.empty((0, len(feature_names)), dtype=np.float64),
                y_validation=y_validation,
                x_test=transform_panel_feature_matrix(encoder, test),
                y_test=y_test,
                test_meta=test.loc[:, meta_cols].copy(),
            )
        )
    return tuple(slices)
