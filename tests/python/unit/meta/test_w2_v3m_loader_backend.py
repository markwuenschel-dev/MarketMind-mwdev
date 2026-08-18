from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pysrc.pipeline.stages.market_data.sources.sip_panel import load_sip_adjusted_panel

pytestmark = pytest.mark.determinism("d1")


def _write_partition(path: Path, *, date: str, instrument_seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "date": [date, date],
            "instrument": [f"SYM{instrument_seed}", f"SYM{instrument_seed + 1}"],
            "raw_close": [10.0 + instrument_seed, 11.0 + instrument_seed],
            "raw_volume": [1_000_000.0, 2_000_000.0],
            "adj_open": [9.9 + instrument_seed, 10.9 + instrument_seed],
            "adj_high": [10.2 + instrument_seed, 11.2 + instrument_seed],
            "adj_low": [9.7 + instrument_seed, 10.7 + instrument_seed],
            "adj_close": [10.1 + instrument_seed, 11.1 + instrument_seed],
            "adj_volume": [900_000.0, 1_900_000.0],
            "corporate_action_flag": [False, False],
            "extreme_raw_return_flag": [False, False],
            "extreme_adjusted_return_flag": [False, False],
        }
    )
    frame.to_parquet(path, index=False)


def test_w2_loader_polars_parity_with_pandas(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    root = tmp_path / "panel"
    _write_partition(
        root / "year=2024" / "month=01" / "2024-01-02.parquet", date="2024-01-02", instrument_seed=1
    )
    _write_partition(
        root / "year=2024" / "month=01" / "2024-01-03.parquet", date="2024-01-03", instrument_seed=3
    )

    pandas_loaded = load_sip_adjusted_panel(root, loader_backend="pandas", workers=2)
    polars_loaded = load_sip_adjusted_panel(root, loader_backend="polars", workers=2)

    columns = ["date", "instrument", "raw_close", "raw_volume", "adj_close", "adj_volume"]
    pd.testing.assert_frame_equal(
        pandas_loaded.loc[:, columns].sort_values(["date", "instrument"]).reset_index(drop=True),
        polars_loaded.loc[:, columns].sort_values(["date", "instrument"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_w2_loader_polars_coalesce_float32_matches_pandas(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    root = tmp_path / "panel"
    _write_partition(
        root / "year=2024" / "month=01" / "2024-01-02.parquet", date="2024-01-02", instrument_seed=1
    )

    pandas_loaded = load_sip_adjusted_panel(
        root,
        loader_backend="pandas",
        panel_coalesce_float32=True,
    )
    polars_loaded = load_sip_adjusted_panel(
        root,
        loader_backend="polars",
        panel_coalesce_float32=True,
    )

    assert str(pandas_loaded["raw_close"].dtype) == "float32"
    assert str(polars_loaded["raw_close"].dtype) == "float32"
    assert str(pandas_loaded["adj_close"].dtype) == "float32"
    assert str(polars_loaded["adj_close"].dtype) == "float32"


def test_w2_loader_backend_rejects_unknown_backend(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    root = tmp_path / "panel"
    _write_partition(
        root / "year=2024" / "month=01" / "2024-01-02.parquet", date="2024-01-02", instrument_seed=1
    )

    with pytest.raises(ValueError, match="Unsupported loader_backend"):
        load_sip_adjusted_panel(root, loader_backend="invalid_backend")
