"""Prep tests documenting LSTM PanelModel expectations before registry promotion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysrc.models.base import PanelModel
from pysrc.models.registry import EXECUTABLE_MODEL_FAMILIES, resolve_model_family
from pysrc.pipeline.panel.sequence_data import build_sequence_windows


@pytest.mark.determinism("d1")
def test_lstm_remains_planned_not_executable() -> None:
    assert "lstm" not in EXECUTABLE_MODEL_FAMILIES
    with pytest.raises(ValueError, match="planned but not yet executable"):
        resolve_model_family("lstm")


@pytest.mark.determinism("d1")
def test_sequence_windows_output_shape_matches_lstm_contract(deterministic_seed: int) -> None:
    """Windows materialized from the panel must be 3-D for LstmPanelModel.fit."""
    del deterministic_seed
    seq_len = 3
    frame = pd.DataFrame(
        {
            "date": [f"2020-01-{i:02d}" for i in range(1, 8)] * 2,
            "instrument": ["A"] * 7 + ["B"] * 7,
            "f0": np.arange(14, dtype=np.float64),
            "f1": np.arange(14, dtype=np.float64) * 0.1,
            "target": np.linspace(0.0, 1.0, 14),
        }
    )
    xs, ys, meta = build_sequence_windows(
        frame,
        ["f0", "f1"],
        "target",
        sequence_length=seq_len,
    )
    assert xs.ndim == 3
    assert xs.shape[1] == seq_len
    assert xs.shape[2] == 2
    assert ys.ndim == 1
    assert len(meta) == len(ys)


@pytest.mark.determinism("d1")
def test_regression_model_forward_accepts_3d_input(deterministic_seed: int) -> None:
    """Document that the panel backbone expects (batch, seq_len, features)."""
    torch = pytest.importorskip("torch")
    from pysrc.models.lstm import LSTMConfig, Model  # noqa: PLC0415

    del deterministic_seed
    cfg = LSTMConfig(input_dim=4, units=8, num_layers=1, bidirectional=False, seed=3)
    model = Model(cfg)
    x = torch.randn(5, 10, 4, dtype=torch.float32)
    out = model(x)
    assert out.shape == (5,)


@pytest.mark.determinism("d1")
def test_lstm_panel_model_is_panel_model_protocol(deterministic_seed: int) -> None:
    pytest.importorskip("torch")
    from pysrc.models.lstm import LstmPanelModel

    del deterministic_seed
    model = LstmPanelModel(model_id="lstm", sequence_length=5, random_seed=3)
    assert isinstance(model, PanelModel)


@pytest.mark.determinism("d1")
def test_lstm_panel_model_fit_predict_roundtrip(deterministic_seed: int) -> None:
    pytest.importorskip("torch")
    from pysrc.models.lstm import create_lstm_panel_model

    del deterministic_seed
    rng = np.random.default_rng(3)
    n, seq_len, n_feat = 32, 5, 4
    x = rng.standard_normal((n, seq_len, n_feat))
    y = rng.standard_normal(n)
    model = create_lstm_panel_model(
        model_id="lstm",
        sequence_length=seq_len,
        random_seed=3,
        params={"units": 8, "num_layers": 1, "epochs": 2, "batch_size": 8},
    )
    model.set_feature_names([f"f{i}" for i in range(n_feat)])
    model.fit(x, y, fold_id="fold_0")
    preds = model.predict(x)
    assert preds.shape == (n,)
    assert np.isfinite(preds).all()


@pytest.mark.determinism("d1")
def test_lstm_panel_model_predict_confidence_bounded(deterministic_seed: int) -> None:
    pytest.importorskip("torch")
    from pysrc.models.lstm import create_lstm_panel_model

    del deterministic_seed
    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, 4, 3))
    y = rng.standard_normal(8)
    model = create_lstm_panel_model(sequence_length=4, random_seed=5, params={"epochs": 1})
    model.fit(x, y, fold_id="fold_0")
    conf = model.predict_confidence(x)
    assert conf.shape == (8,)
    assert np.all(conf >= 0.0)
    assert np.all(conf <= 1.0)


@pytest.mark.determinism("d1")
def test_lstm_panel_model_save_load_roundtrip(tmp_path: Path, deterministic_seed: int) -> None:
    pytest.importorskip("torch")
    from pysrc.models.lstm import create_lstm_panel_model

    del deterministic_seed
    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, 3, 2))
    y = rng.standard_normal(16)
    model = create_lstm_panel_model(sequence_length=3, random_seed=7, params={"epochs": 1})
    model.fit(x, y, fold_id="fold_0")
    ckpt = tmp_path / "lstm_panel.pt"
    model.save(ckpt)
    loaded = create_lstm_panel_model(sequence_length=3, random_seed=7).load(ckpt)
    np.testing.assert_allclose(loaded.predict(x), model.predict(x), rtol=1e-5, atol=1e-5)


@pytest.mark.determinism("d1")
def test_lstm_panel_model_rejects_flat_input(deterministic_seed: int) -> None:
    pytest.importorskip("torch")
    from pysrc.models.lstm import create_lstm_panel_model

    del deterministic_seed
    model = create_lstm_panel_model(sequence_length=5, random_seed=3)
    x_flat = np.zeros((10, 4))
    y = np.zeros(10)
    with pytest.raises((ValueError, RuntimeError)):
        model.fit(x_flat, y, fold_id="fold_0")


@pytest.mark.determinism("d1")
def test_sequence_fold_masks_use_meta_dates_not_panel_indices(deterministic_seed: int) -> None:
    from pysrc.pipeline.panel.train_model_matrix import (
        _sequence_fold_masks,
        build_chronological_date_codes,
        build_walk_forward_boundaries,
    )

    del deterministic_seed
    seq_len = 2
    frame = pd.DataFrame(
        {
            "date": [f"2020-01-{i:02d}" for i in range(1, 9)] * 2,
            "instrument": ["A"] * 8 + ["B"] * 8,
            "f0": np.arange(16, dtype=np.float64),
            "target": np.linspace(0.0, 1.0, 16),
        }
    )
    _, meta = build_sequence_windows(frame, ["f0"], "target", sequence_length=seq_len)[1:]
    del _
    _, unique_dates = build_chronological_date_codes(frame["date"].astype(str).to_numpy())
    boundaries = build_walk_forward_boundaries(unique_dates, n_folds=2, target_horizon_days=0)
    boundary = boundaries[1]
    train_mask, test_mask = _sequence_fold_masks(meta, boundary, list(unique_dates))
    assert train_mask.any()
    assert test_mask.any()
    assert not np.any(train_mask & test_mask)


@pytest.mark.determinism("d1")
def test_sequence_windows_drop_leading_rows_per_instrument(deterministic_seed: int) -> None:
    del deterministic_seed
    seq_len = 4
    frame = pd.DataFrame(
        {
            "date": [f"2020-01-{i:02d}" for i in range(1, 11)] * 2,
            "instrument": ["A"] * 10 + ["B"] * 10,
            "f0": np.arange(20, dtype=np.float64),
            "target": np.arange(20, dtype=np.float64),
        }
    )
    xs, _, meta = build_sequence_windows(frame, ["f0"], "target", sequence_length=seq_len)
    assert len(meta) < len(frame)
    assert len(meta) == len(frame) - 2 * seq_len
    assert xs.shape[0] == len(meta)


@pytest.mark.determinism("d1")
def test_lstm_standard_scaler_preserves_sequence_shape(deterministic_seed: int) -> None:
    pytest.importorskip("torch")
    from sklearn.preprocessing import StandardScaler

    from pysrc.models.lstm import LstmPanelModel

    del deterministic_seed
    rng = np.random.default_rng(11)
    x = rng.standard_normal((12, 5, 3))
    model = LstmPanelModel(sequence_length=5, random_seed=11)
    scaled = model._scale(x, fit=True)
    assert scaled.shape == x.shape
    ref = StandardScaler().fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    np.testing.assert_allclose(scaled, ref.astype(np.float32), rtol=1e-5, atol=1e-5)
