from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pysrc.core.errors import PreprocessingError


@pytest.mark.determinism("d0")
def test_core_load_ohlcv_governed_path_rejects_direct_csv_fallback(
    deterministic_seed: int, tmp_path: Path
) -> None:
    from pysrc.preprocessor.core import load_ohlcv

    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("date,close\n2024-01-01,1.0\n", encoding="utf-8")

    with patch("pysrc.preprocessor.core.run", side_effect=RuntimeError("registry missing")):
        with pytest.raises(PreprocessingError, match="governed path"):
            load_ohlcv(csv_path)


@pytest.mark.determinism("d0")
def test_run_governed_path_rejects_oom_retry_fallback(deterministic_seed: int) -> None:
    import polars as pl

    from pysrc.preprocessor.api import Plan, run
    from pysrc.preprocessor.utils.errors import OOMRetry

    df = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    plan = Plan(ops=[], params={}, group_by=[])

    with patch("pysrc.preprocessor.api.get_executor") as mock_get:
        executor = mock_get.return_value
        executor.execution_history = []
        executor.execute.side_effect = OOMRetry("OOM")

        with pytest.raises(PreprocessingError, match="governed path"):
            run(df, plan, backend="gpu")


@pytest.mark.determinism("d0")
def test_columns_governed_path_rejects_return_original_frame_fallback(
    deterministic_seed: int,
) -> None:
    import polars as pl

    from pysrc.preprocessor.ops.common.columns import PromoteCategorical

    df = pl.DataFrame({"symbol": ["A", "B"]})
    op = PromoteCategorical()

    with patch("pysrc.preprocessor.ops.common.columns.capabilities") as mock_caps:
        mock_caps.return_value.has_cudf = False
        mock_caps.return_value.has_polars_gpu = False
        with patch.object(pl.DataFrame, "lazy", side_effect=RuntimeError("boom")):
            with pytest.raises(PreprocessingError, match="governed path"):
                op.apply(df, ["symbol"], governed=True)


@pytest.mark.determinism("d0")
def test_market_calendar_governed_path_rejects_approximate_calendar_fallback(
    deterministic_seed: int,
) -> None:
    import builtins

    from pysrc.preprocessor.domain.market_calendar import MarketCalendarFactory

    real_import = builtins.__import__

    def _import(name: str, *args, **kwargs):
        if name in {"exchange_calendars", "pandas_market_calendars"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import):
        with pytest.raises(PreprocessingError, match="governed path"):
            MarketCalendarFactory.get_calendar(governed=True)
