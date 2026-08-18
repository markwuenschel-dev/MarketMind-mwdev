from __future__ import annotations

import math
from typing import cast

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pysrc.preprocessor.graph.factory import register_builtin_ops, registry_snapshot
from pysrc.preprocessor.graph.ops_custom import (
    IndustryScore,
    ResidualOLS,
    VolScale,
    XSecRank,
    lower_industry_score_polars,
    lower_residual_kf_polars,
    lower_vol_scale_polars,
    lower_xsec_rank_polars,
)

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


pytestmark = pytest.mark.determinism("d1")
HAS_POLARS = pl is not None


def test_all_momentum_ops_in_registry() -> None:
    register_builtin_ops()
    registry, _ = registry_snapshot()
    for key in (
        "momentum.xsec_rank",
        "momentum.vol_scale",
        "momentum.residual_ols",
        "momentum.residual_kf",
        "momentum.industry_score",
    ):
        assert key in registry


def test_xsec_rank_to_ir_keys() -> None:
    ir = XSecRank(col="returns", window=252, out_col="mom_rank").to_ir()
    assert {"op", "kind", "params", "requires", "provides"} <= set(ir)


def test_vol_scale_to_ir_keys() -> None:
    ir = VolScale(
        signal_col="mom_rank",
        vol_col="realized_vol_60",
        target_vol=0.15,
        max_leverage=2.0,
        out_col="mom_scaled",
    ).to_ir()
    assert {"op", "kind", "params", "requires", "provides"} <= set(ir)


def test_residual_ols_to_ir_keys() -> None:
    ir = ResidualOLS(
        asset_ret_col="returns",
        factor_ret_cols=["market_return"],
        window=63,
        out_col="residual_ols",
    ).to_ir()
    assert {"op", "kind", "params", "requires", "provides"} <= set(ir)


def test_industry_score_to_ir_keys() -> None:
    ir = IndustryScore(
        ret_col="returns",
        sector_col="sector",
        window=20,
        out_col="industry",
    ).to_ir()
    assert {"op", "kind", "params", "requires", "provides"} <= set(ir)


@pytest.mark.property
@pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
@given(
    st.lists(
        st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=30,
    )
)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_xsec_rank_polars_without_panel_columns_returns_nan(values: list[float]) -> None:
    assert pl is not None
    frame = pl.DataFrame({"returns": values})
    out = cast(
        pl.LazyFrame,
        lower_xsec_rank_polars(
            {"params": {"col": "returns", "window": 5, "skip": 1, "out_col": "mom_rank"}},
            frame.lazy(),
        ),
    ).collect()
    assert all(value is None or math.isnan(float(value)) for value in out["mom_rank"].to_list())


@pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
def test_xsec_rank_polars_is_cross_sectional_by_date() -> None:
    assert pl is not None
    frame = pl.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-03",
            ],
            "symbol": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "returns": [0.01, 0.02, -0.01, 0.01, 0.02, -0.01, 0.01, 0.02, -0.01],
        }
    )
    out = cast(
        pl.LazyFrame,
        lower_xsec_rank_polars(
            {
                "params": {
                    "col": "returns",
                    "window": 2,
                    "skip": 1,
                    "date_col": "date",
                    "asset_col": "symbol",
                    "out_col": "mom_rank",
                }
            },
            frame.lazy(),
        ),
    ).collect()
    ranks = out.filter(pl.col("date") == "2024-01-03").sort("symbol")["mom_rank"].to_list()
    assert ranks == pytest.approx([2.0 / 3.0, 1.0, 1.0 / 3.0])


@pytest.mark.property
@pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
@given(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(deadline=None)
def test_vol_scale_denominator_guard_zero(signal: float) -> None:
    assert pl is not None
    frame = pl.DataFrame({"mom_rank": [signal], "realized_vol_60": [0.0]})
    out = cast(
        pl.LazyFrame,
        lower_vol_scale_polars(
            {
                "params": {
                    "signal_col": "mom_rank",
                    "vol_col": "realized_vol_60",
                    "target_vol": 0.15,
                    "max_leverage": 2.0,
                    "out_col": "mom_scaled",
                }
            },
            frame.lazy(),
        ),
    ).collect()
    assert out["mom_scaled"][0] == pytest.approx(2.0 * signal)


@pytest.mark.property
@pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
@given(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(deadline=None)
def test_vol_scale_denominator_guard_nan(signal: float) -> None:
    assert pl is not None
    frame = pl.DataFrame({"mom_rank": [signal], "realized_vol_60": [float("nan")]})
    out = cast(
        pl.LazyFrame,
        lower_vol_scale_polars(
            {
                "params": {
                    "signal_col": "mom_rank",
                    "vol_col": "realized_vol_60",
                    "target_vol": 0.15,
                    "max_leverage": 2.0,
                    "out_col": "mom_scaled",
                }
            },
            frame.lazy(),
        ),
    ).collect()
    assert out["mom_scaled"][0] == pytest.approx(2.0 * signal)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
def test_vol_scale_annualization() -> None:
    assert pl is not None
    signal = 0.5
    daily_std = 0.10
    frame = pl.DataFrame({"mom_rank": [signal], "realized_vol_60": [daily_std]})
    out = cast(
        pl.LazyFrame,
        lower_vol_scale_polars(
            {
                "params": {
                    "signal_col": "mom_rank",
                    "vol_col": "realized_vol_60",
                    "target_vol": 0.15,
                    "max_leverage": 2.0,
                    "out_col": "mom_scaled",
                }
            },
            frame.lazy(),
        ),
    ).collect()
    expected_scale = min(0.15 / (daily_std * math.sqrt(252.0)), 2.0)
    assert out["mom_scaled"][0] == pytest.approx(expected_scale * signal)


def test_residual_kf_lowering_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="OI-MOM-005"):
        lower_residual_kf_polars({}, None)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
def test_industry_score_lowering_raises_phase_stub() -> None:
    assert pl is not None
    with pytest.raises(NotImplementedError, match="OI-MOM-004"):
        lower_industry_score_polars(
            {
                "params": {
                    "ret_col": "returns",
                    "sector_col": "sector",
                    "window": 20,
                    "out_col": "industry_score",
                }
            },
            pl.DataFrame({"returns": [0.1], "sector": ["tech"]}).lazy(),
        )


def test_vol_scale_validate_params_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="target_vol"):
        VolScale(signal_col="sig", vol_col="vol", target_vol=0.0, max_leverage=2.0)
    with pytest.raises(ValueError, match="max_leverage"):
        VolScale(signal_col="sig", vol_col="vol", target_vol=0.1, max_leverage=-1.0)
