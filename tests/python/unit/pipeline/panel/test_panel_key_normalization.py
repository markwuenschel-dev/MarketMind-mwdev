"""Tests for canonical panel key normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pysrc.pipeline.panel.model_diversity import (
    build_panel_target_lookup,
    build_streaming_model_diversity_report,
)
from pysrc.pipeline.panel.model_matrix_validation import (
    _load_price_frame,
    build_forward_target_numerical_audit,
    build_streaming_prediction_sanity_report,
)
from pysrc.pipeline.panel.panel_keys import (
    CANONICAL_DAILY_INTERVAL,
    normalize_key_columns,
    normalize_panel_date,
    normalize_panel_interval,
)


@pytest.mark.determinism("d0")
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-01-02", "2024-01-02"),
        ("2024-01-02T00:00:00Z", "2024-01-02"),
        (pd.Timestamp("2024-01-02", tz="UTC"), "2024-01-02"),
    ],
)
def test_normalize_panel_date_formats(raw: object, expected: str, deterministic_seed: int) -> None:
    _ = deterministic_seed
    assert normalize_panel_date(raw) == expected


@pytest.mark.determinism("d0")
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("daily", CANONICAL_DAILY_INTERVAL),
        ("1d", CANONICAL_DAILY_INTERVAL),
        ("D", CANONICAL_DAILY_INTERVAL),
        ("1h", "1h"),
    ],
)
def test_normalize_panel_interval_aliases(raw: str, expected: str, deterministic_seed: int) -> None:
    _ = deterministic_seed
    assert normalize_panel_interval(raw) == expected


@pytest.mark.determinism("d0")
def test_normalize_key_columns_frame(deterministic_seed: int) -> None:
    _ = deterministic_seed
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02T00:00:00Z", "2024-01-03"],
            "instrument": ["AAA", "BBB"],
            "interval": ["daily", "D"],
        }
    )
    normalized = normalize_key_columns(frame)
    assert normalized["date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert normalized["interval"].tolist() == [CANONICAL_DAILY_INTERVAL, CANONICAL_DAILY_INTERVAL]


@pytest.mark.determinism("d1")
def test_panel_target_lookup_matches_interval_aliases(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    panel_path = tmp_path / "panel.parquet"
    panel = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "instrument": ["AAA", "BBB"],
            "interval": ["1d", "1d"],
            "forward_return_horizon": [0.02, -0.01],
        }
    )
    pq.write_table(pa.Table.from_pandas(panel), panel_path)

    lookup = build_panel_target_lookup(panel_path, target_column="forward_return_horizon")
    predictions = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "instrument": ["AAA", "BBB"],
            "interval": ["daily", "D"],
        }
    )
    realized = lookup(predictions)
    assert realized.tolist() == pytest.approx([0.02, -0.01])


@pytest.mark.determinism("d1")
def test_streaming_diversity_attaches_realized_with_normalized_keys(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    panel_path = tmp_path / "panel.parquet"
    panel = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "instrument": ["AAA"],
            "interval": ["1d"],
            "forward_return_horizon": [0.03],
        }
    )
    pq.write_table(pa.Table.from_pandas(panel), panel_path)
    target_lookup = build_panel_target_lookup(panel_path, target_column="forward_return_horizon")

    pred_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "date": ["2024-01-02T00:00:00Z"],
            "instrument": ["AAA"],
            "interval": ["daily"],
            "model_id": ["ridge"],
            "fold_id": ["fold_0"],
            "prediction": [0.01],
        }
    ).to_parquet(pred_path, index=False)

    report = build_streaming_model_diversity_report(
        pred_path,
        target_column="forward_return_horizon",
        target_lookup=target_lookup,
        random_seed=3,
        max_sample_rows=10,
        expected_model_count=1,
        expected_fold_count=1,
    )
    assert report["streaming_rows_processed"] == 1


@pytest.mark.determinism("d1")
def test_streaming_prediction_sanity_uses_normalized_keys(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    panel_path = tmp_path / "panel.parquet"
    panel = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "instrument": ["AAA"],
            "interval": ["1d"],
            "forward_return_horizon": [0.04],
        }
    )
    pq.write_table(pa.Table.from_pandas(panel), panel_path)
    target_lookup = build_panel_target_lookup(panel_path, target_column="forward_return_horizon")

    pred_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "date": ["2024-01-02T00:00:00Z"],
            "instrument": ["AAA"],
            "interval": ["daily"],
            "model_id": ["ridge"],
            "fold_id": ["fold_0"],
            "prediction": [0.02],
        }
    ).to_parquet(pred_path, index=False)

    report = build_streaming_prediction_sanity_report(
        pred_path,
        target_column="forward_return_horizon",
        target_lookup=target_lookup,
    )
    ridge_fold = report["by_model_fold"][0]
    assert ridge_fold["target"]["mean"] == pytest.approx(0.04)


@pytest.mark.determinism("d1")
def test_load_price_frame_reads_hive_partitions(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    month_dir = tmp_path / "year=2024" / "month=01"
    month_dir.mkdir(parents=True)
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "instrument": ["AAA", "AAA"],
            "adj_close": [100.0, 101.0],
        }
    )
    prices.to_parquet(month_dir / "part-000.parquet", index=False)

    loaded = _load_price_frame(tmp_path, {"AAA"})
    assert len(loaded) == 2
    assert set(loaded["instrument"].astype(str)) == {"AAA"}


@pytest.mark.determinism("d1")
def test_load_price_frame_normalizes_ticker_column(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    month_dir = tmp_path / "year=2024" / "month=01"
    month_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "ticker": ["AAA", "BBB"],
            "adj_close": [100.0, 101.0],
        }
    ).to_parquet(month_dir / "part-000.parquet", index=False)

    loaded = _load_price_frame(tmp_path, {"AAA"})
    assert len(loaded) == 1
    assert loaded.iloc[0]["instrument"] == "AAA"


@pytest.mark.determinism("d1")
def test_forward_audit_uses_full_price_history_not_sample_rows(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    panel_path = tmp_path / "panel.parquet"
    panel = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "instrument": ["AAA", "AAA", "AAA"],
            "interval": ["1d", "1d", "1d"],
            "forward_return_horizon": [0.01, 102 / 101 - 1, np.nan],
            "adj_close": [100.0, 101.0, 102.0],
        }
    )
    pq.write_table(pa.Table.from_pandas(panel), panel_path)

    substrate = tmp_path / "prices"
    month_dir = substrate / "year=2024" / "month=01"
    month_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "instrument": ["AAA", "AAA", "AAA"],
            "adj_close": [100.0, 101.0, 102.0],
        }
    ).to_parquet(month_dir / "part-000.parquet", index=False)

    audit = build_forward_target_numerical_audit(
        panel_path=panel_path,
        target_column="forward_return_horizon",
        horizon_days=1,
        price_substrate_path=substrate,
        random_seed=11,
        sample_size=1,
    )
    assert audit["verification_status"] == "performed"
    assert int(audit["rows_checked"]) >= 1
    assert int(audit["mismatch_count"] or 0) == 0
