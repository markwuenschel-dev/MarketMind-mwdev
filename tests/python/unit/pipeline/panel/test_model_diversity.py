"""Tests for model-matrix diversity diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysrc.pipeline.panel.model_diversity import (
    _cross_sectional_ic_by_group,
    _utility_by_group,
    build_low_memory_model_diversity_report,
    build_model_diversity_report,
    materialize_realized_panel_from_scratch,
)


@pytest.mark.determinism("d1")
def test_model_diversity_report_structure(deterministic_seed: int) -> None:
    _ = deterministic_seed
    rng = np.random.default_rng(5)
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    instruments = ["AAA", "BBB", "CCC"]
    rows: list[dict[str, object]] = []
    for model_id, scale in [("ridge", 1.0), ("random_forest", 1.2), ("mlp", 3.0)]:
        for date in dates:
            for instrument in instruments:
                rows.append(
                    {
                        "date": date,
                        "instrument": instrument,
                        "model_id": model_id,
                        "model_family": model_id,
                        "fold_id": "fold_0",
                        "split": "test",
                        "prediction": float(rng.normal(scale, 0.1)),
                        "confidence": 0.5,
                        "target_name": "adjusted_return_1d",
                        "run_id": "test",
                        "prediction_rank": 1,
                        "interval": "daily",
                    }
                )
    predictions = pd.DataFrame(rows)
    panel_rows: list[dict[str, object]] = []
    for date in dates:
        for instrument in instruments:
            panel_rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "adjusted_return_1d": float(rng.normal(0.0, 0.01)),
                }
            )
    panel = pd.DataFrame(panel_rows)

    report = build_model_diversity_report(predictions, panel, "adjusted_return_1d", top_k=2)

    assert report["schema_version"] == "model_diversity_report.v2"
    assert report["model_count"] == 3
    assert "prediction_correlation" in report
    assert "rank_correlation" in report
    assert "residual_correlation" in report
    assert "cross_sectional_correlation" in report
    assert "top_k_overlap" in report
    assert "fold_by_fold_spearman_ic" in report
    assert "ex_post_regime_spearman_ic" in report
    assert "ex_post_regime_documentation" in report
    assert "soft_redundancy_clusters" in report
    assert "nonredundant_child_count" in report
    assert "eligible_router_child_count" in report
    assert "prediction_dispersion" in report
    assert len(report["prediction_dispersion"]) == 3


@pytest.mark.determinism("d1")
def test_model_diversity_flags_redundant_pairs(deterministic_seed: int) -> None:
    _ = deterministic_seed
    dates = ["2024-01-02", "2024-01-03"]
    instruments = ["AAA", "BBB"]
    base = [
        {"date": d, "instrument": i, "prediction": 0.1 * idx}
        for idx, (d, i) in enumerate((d, i) for d in dates for i in instruments)
    ]
    rows: list[dict[str, object]] = []
    for model_id in ("ridge", "elastic_net"):
        for item in base:
            rows.append(
                {
                    **item,
                    "model_id": model_id,
                    "model_family": model_id,
                    "fold_id": "fold_0",
                    "split": "test",
                    "confidence": 0.5,
                    "target_name": "adjusted_return_1d",
                    "run_id": "test",
                    "prediction_rank": 1,
                    "interval": "daily",
                }
            )
    predictions = pd.DataFrame(rows)
    panel = pd.DataFrame(
        [
            {"date": d, "instrument": i, "adjusted_return_1d": 0.01}
            for d in dates
            for i in instruments
        ]
    )

    report = build_model_diversity_report(predictions, panel, "adjusted_return_1d", top_k=2)

    assert report["redundant_pairs"]
    assert report["nonredundant_child_count"] < report["model_count"]


@pytest.mark.determinism("d1")
def test_low_memory_diversity_report_matches_full_report(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    rng = np.random.default_rng(9)
    dates = ["2024-01-02", "2024-01-03"]
    instruments = ["AAA", "BBB"]
    n_rows = len(dates) * len(instruments)
    date_codes = np.array([0, 0, 1, 1], dtype=np.int32)
    instrument_codes = np.array([0, 1, 0, 1], dtype=np.int32)
    targets = rng.normal(size=n_rows).astype(np.float32)
    interval_codes = np.array([0, 0, 0, 0], dtype=np.int32)
    interval_code_path = tmp_path / "interval_codes.int32.memmap"
    np.memmap(interval_code_path, dtype=np.int32, mode="w+", shape=(n_rows,))[:] = interval_codes
    date_code_path = tmp_path / "date_codes.int32.memmap"
    instrument_code_path = tmp_path / "instrument_codes.int32.memmap"
    target_path = tmp_path / "target.float32.memmap"
    np.memmap(date_code_path, dtype=np.int32, mode="w+", shape=(n_rows,))[:] = date_codes
    np.memmap(instrument_code_path, dtype=np.int32, mode="w+", shape=(n_rows,))[:] = (
        instrument_codes
    )
    np.memmap(target_path, dtype=np.float32, mode="w+", shape=(n_rows,))[:] = targets

    rows: list[dict[str, object]] = []
    for model_id, scale in [("ridge", 1.0), ("mlp", 2.5)]:
        for idx in range(n_rows):
            rows.append(
                {
                    "date": dates[date_codes[idx]],
                    "instrument": instruments[instrument_codes[idx]],
                    "model_id": model_id,
                    "model_family": model_id,
                    "fold_id": "fold_0",
                    "split": "test",
                    "prediction": float(targets[idx] * scale + rng.normal(0.0, 0.01)),
                    "confidence": 0.5,
                    "target_name": "adjusted_return_1d",
                    "run_id": "test",
                    "prediction_rank": 1,
                    "interval": "1d",
                }
            )
    pred_path = tmp_path / "predictions.parquet"
    pd.DataFrame(rows).to_parquet(pred_path, index=False)

    panel = materialize_realized_panel_from_scratch(
        target_column="adjusted_return_1d",
        n_rows=n_rows,
        unique_dates=tuple(dates),
        unique_instruments=tuple(instruments),
        date_code_path=date_code_path,
        instrument_code_path=instrument_code_path,
        target_path=target_path,
    )
    full = build_model_diversity_report(
        pd.read_parquet(pred_path),
        panel,
        "adjusted_return_1d",
        top_k=2,
    )
    low_memory = build_low_memory_model_diversity_report(
        pred_path,
        target_column="adjusted_return_1d",
        n_rows=n_rows,
        unique_dates=tuple(dates),
        unique_instruments=tuple(instruments),
        unique_intervals=("1d",),
        date_code_path=date_code_path,
        instrument_code_path=instrument_code_path,
        interval_code_path=interval_code_path,
        target_path=target_path,
        top_k=2,
    )

    assert low_memory["schema_version"] == "model_diversity_report.v2"
    assert low_memory["model_count"] == full["model_count"]
    assert (
        low_memory["prediction_correlation"]["mean_pairwise"]
        == full["prediction_correlation"]["mean_pairwise"]
    )
    assert len(low_memory["prediction_dispersion"]) == len(full["prediction_dispersion"])


@pytest.mark.determinism("d1")
def test_fold_by_fold_spearman_ic_when_frame_has_realized(deterministic_seed: int) -> None:
    """Streaming path attaches realized before build_model_diversity_report."""
    _ = deterministic_seed
    rng = np.random.default_rng(11)
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    instruments = ["AAA", "BBB", "CCC"]
    folds = ["fold_0", "fold_1"]
    panel_rows: list[dict[str, object]] = []
    for date in dates:
        for instrument in instruments:
            panel_rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "interval": "1d",
                    "adjusted_return_1d": float(rng.normal(0.0, 0.01)),
                }
            )
    panel = pd.DataFrame(panel_rows)
    rows: list[dict[str, object]] = []
    for model_id, scale in [("ridge", 1.0), ("random_forest", 1.2)]:
        for fold_id in folds:
            for date in dates:
                for instrument in instruments:
                    target = float(
                        panel.loc[
                            (panel["date"] == date) & (panel["instrument"] == instrument),
                            "adjusted_return_1d",
                        ].iloc[0]
                    )
                    rows.append(
                        {
                            "date": date,
                            "instrument": instrument,
                            "interval": "1d",
                            "model_id": model_id,
                            "fold_id": fold_id,
                            "prediction": float(target * scale + rng.normal(0.0, 0.001)),
                            "realized": target,
                        }
                    )
    predictions = pd.DataFrame(rows)

    report = build_model_diversity_report(predictions, panel, "adjusted_return_1d", top_k=2)

    fold_ic = report["fold_by_fold_spearman_ic"]
    assert fold_ic
    assert set(fold_ic) == set(folds)
    for fold_id in folds:
        assert set(fold_ic[fold_id]) == {"ridge", "random_forest"}
        assert all(np.isfinite(value) for value in fold_ic[fold_id].values())


@pytest.mark.determinism("d1")
def test_cross_sectional_ic_beats_pooled_when_levels_drift(deterministic_seed: int) -> None:
    """Per-date rank IC should stay high when pooled fold IC washes out level shifts."""
    _ = deterministic_seed
    dates = [f"2024-01-{day:02d}" for day in range(2, 12)]
    instruments = ["AAA", "BBB", "CCC", "DDD"]
    rows: list[dict[str, object]] = []
    for day_idx, date in enumerate(dates):
        level = float(day_idx * 10)
        for rank, instrument in enumerate(instruments):
            realized = level + float(rank)
            rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "interval": "1d",
                    "model_id": "ridge",
                    "fold_id": "fold_0",
                    "prediction": float(rank),
                    "realized": realized,
                }
            )
    frame = pd.DataFrame(rows)
    pooled = _utility_by_group(frame, "fold_id")
    cross = _cross_sectional_ic_by_group(frame, "fold_id")
    assert cross["fold_0"]["ridge"] > 0.99
    assert pooled["fold_0"]["ridge"] < cross["fold_0"]["ridge"]
