from __future__ import annotations

from datetime import UTC, date

import pandas as pd
import pytest

from pysrc.data.dataview import DataView
from pysrc.strategies.pipeline_strategy import (
    FeaturePlan,
    StrategyContext,
    materialize_features,
)
from pysrc.strategies.stat_arb.config import PAIRS_DEFAULT, HedgeEstimator, PairsConfig
from pysrc.strategies.stat_arb.pairs import StatArbPairsStrategy


def _synthetic_pair_history() -> pd.DataFrame:
    """Build a tiny SPY/QQQ PIT-safe history frame (long: one row per symbol per date)."""
    rows = []
    for i, (spy, qqq) in enumerate(
        [
            (100.0, 50.0),
            (101.0, 50.5),
            (102.0, 51.0),
            (103.0, 51.5),
            (104.0, 52.0),
        ],
        start=0,
    ):
        d = date(2024, 1, 1 + i)
        rows.append(
            {
                "symbol": "SPY",
                "valid_time": d,
                "knowledge_time": d,
                "SPY.close": spy,
                "QQQ.close": float("nan"),
            }
        )
        rows.append(
            {
                "symbol": "QQQ",
                "valid_time": d,
                "knowledge_time": d,
                "SPY.close": float("nan"),
                "QQQ.close": qqq,
            }
        )
    return pd.DataFrame(rows)


def test_features_plan_uses_expected_ops_and_order():
    cfg = PairsConfig()
    strat = StatArbPairsStrategy("SPY", "QQQ", config=cfg)
    plan = strat.features_plan()
    assert isinstance(plan, FeaturePlan)
    ops = [step.op for step in plan.steps]
    assert ops == ["pairs.beta", "pairs.spread", "scaling.zscore_roll", "stats.half_life"]


@pytest.mark.determinism("d1")
def test_generate_signal_produces_discrete_series(tmp_path):
    from datetime import datetime

    from pysrc.data.dataview_asof_adapter import DataViewAsOfAdapter

    history = _synthetic_pair_history()
    dataview = DataView()
    dataview.register_source(
        history, valid_time_col="valid_time", knowledge_time_col="knowledge_time"
    )
    adapter = DataViewAsOfAdapter(
        dataview=dataview,
        symbols=["SPY", "QQQ"],
        fields=["SPY.close", "QQQ.close"],
    )
    knowledge_dates = [
        datetime.combine(date(2024, 1, d), datetime.min.time()).replace(tzinfo=UTC)
        for d in range(1, 6)
    ]
    wide = adapter.as_wide_frame(knowledge_dates)
    ctx = StrategyContext(prices=wide, backend="pandas", cache_dir=tmp_path)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=PAIRS_DEFAULT)
    feats = materialize_features(ctx, strat.features_plan())
    if not isinstance(feats, pd.DataFrame):
        feats = feats.to_pandas()  # type: ignore[assignment]
    sig = strat.generate_signal(feats)
    assert set(sig.unique()) <= {-1, 0, 1}


def test_kalman_hedge_estimator_raises_not_implemented():
    cfg = PairsConfig(hedge_estimator=HedgeEstimator.KALMAN)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=cfg)
    with pytest.raises(NotImplementedError):
        strat.generate_signal(pd.DataFrame({"x": []}))


def _feat_frame(z_vals, hl_vals, n=None):
    """Build a minimal feature DataFrame for SPY/QQQ strategy (zscore_window=60)."""
    n = n or len(z_vals)
    import numpy as np

    spread = np.zeros(n)
    return pd.DataFrame(
        {
            "spread_SPY_QQQ": spread,
            "spread_SPY_QQQ_z60": z_vals,
            "hl_spread_SPY_QQQ": hl_vals,
        },
        index=pd.RangeIndex(0, n),
    )


def test_entry_on_low_zscore():
    """Z-score crosses below -entry_z → signal becomes +1 on that bar."""
    cfg = PairsConfig(entry_z=2.0, exit_z=0.5, max_hold_days=10, zscore_window=60)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=cfg)
    # Bars 0–2: z above -2; bar 3: z = -2.5 (crosses below -entry_z); hl valid
    z = [0.0, -1.0, -1.5, -2.5, -2.0]
    hl = [10.0] * 5
    feats = _feat_frame(z, hl)
    sig = strat.generate_signal(feats)
    assert sig.iloc[3] == 1
    assert (sig.iloc[:3] == 0).all()


def test_entry_on_high_zscore():
    """Z-score crosses above +entry_z → signal becomes -1 on that bar."""
    cfg = PairsConfig(entry_z=2.0, exit_z=0.5, max_hold_days=10, zscore_window=60)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=cfg)
    z = [0.0, 1.0, 1.5, 2.5, 2.0]
    hl = [10.0] * 5
    feats = _feat_frame(z, hl)
    sig = strat.generate_signal(feats)
    assert sig.iloc[3] == -1
    assert (sig.iloc[:3] == 0).all()


def test_exit_on_mean_reversion():
    """Position is open; z-score returns inside exit_z band → signal becomes 0."""
    cfg = PairsConfig(entry_z=2.0, exit_z=0.5, max_hold_days=10, zscore_window=60)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=cfg)
    # Enter long at bar 1 (z <= -2), exit at bar 3 when z in [-0.5, 0.5]
    z = [0.0, -2.2, -2.0, 0.0, 0.0]
    hl = [10.0] * 5
    feats = _feat_frame(z, hl)
    sig = strat.generate_signal(feats)
    assert sig.iloc[1] == 1
    assert sig.iloc[3] == 0


def test_max_hold_forces_exit():
    """Position held for max_hold_days bars without z-score exit → signal becomes 0."""
    cfg = PairsConfig(entry_z=2.0, exit_z=0.5, max_hold_days=3, zscore_window=60)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=cfg)
    # Enter at bar 0 (z <= -2); z stays outside exit band for 3 bars; force exit at bar 3
    z = [-2.5, -2.0, -1.8, -1.5, 0.0]
    hl = [10.0] * 5
    feats = _feat_frame(z, hl)
    sig = strat.generate_signal(feats)
    assert sig.iloc[0] == 1
    assert sig.iloc[3] == 0


def test_half_life_filter_blocks_entry():
    """Half-life is NaN → signal stays 0 even when z-score crosses entry threshold."""
    cfg = PairsConfig(entry_z=2.0, exit_z=0.5, max_hold_days=10, zscore_window=60)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=cfg)
    z = [0.0, -2.5, -2.5]
    hl = [float("nan"), float("nan"), float("nan")]
    feats = _feat_frame(z, hl)
    sig = strat.generate_signal(feats)
    assert (sig == 0).all()


def test_half_life_filter_allows_exit():
    """Position open; half_life becomes NaN → exit only on zscore/hold, not forced by invalid hl."""
    cfg = PairsConfig(entry_z=2.0, exit_z=0.5, max_hold_days=10, zscore_window=60)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=cfg)
    # Enter at bar 0 (valid hl), then hl becomes nan; exit when z in band at bar 2
    z = [-2.5, -1.0, 0.0]
    hl = [10.0, float("nan"), float("nan")]
    feats = _feat_frame(z, hl)
    sig = strat.generate_signal(feats)
    assert sig.iloc[0] == 1
    assert sig.iloc[2] == 0


def test_generate_signal_raises_on_non_dataframe():
    """generate_signal raises TypeError when features is not a DataFrame."""
    strat = StatArbPairsStrategy("SPY", "QQQ", config=PAIRS_DEFAULT)
    with pytest.raises(TypeError, match="expects a pandas DataFrame"):
        strat.generate_signal(None)
    with pytest.raises(TypeError, match="expects a pandas DataFrame"):
        strat.generate_signal([1, 2, 3])


def test_generate_signal_raises_on_missing_feature_columns():
    """generate_signal raises ValueError when required feature columns are missing."""
    strat = StatArbPairsStrategy("SPY", "QQQ", config=PairsConfig(zscore_window=60))
    feats = pd.DataFrame(
        {"spread_SPY_QQQ": [0.0], "spread_SPY_QQQ_z60": [0.0]}
    )  # missing hl_spread_SPY_QQQ
    with pytest.raises(ValueError, match="Required feature columns missing"):
        strat.generate_signal(feats)


def test_write_artifacts_with_none_signals_feats(tmp_path):
    """write_artifacts with signals=None, feats=None still writes both JSONs with defaults."""
    from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
    from pysrc.artifact_registry.bundle_writer import BundleWriter

    writer = BundleWriter(tmp_path)
    store = BundleBacktestArtifactStore(writer)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=PAIRS_DEFAULT)
    strat.write_artifacts(
        store, signals=None, feats=None, run_id="test", evaluation_window="2024-01-01:2024-01-04"
    )
    assert (tmp_path / "execution_assumptions.json").exists()
    assert (tmp_path / "stat_validity_report.json").exists()
    import json

    report = json.loads((tmp_path / "stat_validity_report.json").read_text())
    assert report.get("signal_count") == 0
    assert report.get("half_life_bars") is None


def test_write_artifacts_feats_missing_z_col(tmp_path):
    """write_artifacts when feats has no z column sets mean_spread_zscore to None."""
    from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
    from pysrc.artifact_registry.bundle_writer import BundleWriter

    writer = BundleWriter(tmp_path)
    store = BundleBacktestArtifactStore(writer)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=PairsConfig(zscore_window=60))
    feats = pd.DataFrame({"spread_SPY_QQQ": [0.0], "hl_spread_SPY_QQQ": [10.0]})  # no z column
    strat.write_artifacts(
        store, signals=pd.Series([0]), feats=feats, run_id="test", evaluation_window="w"
    )
    report = __import__("json").loads((tmp_path / "stat_validity_report.json").read_text())
    assert report.get("mean_spread_zscore") is None


def test_write_artifacts_feats_missing_hl_col(tmp_path):
    """write_artifacts when feats has no half_life column sets half_life_bars to None."""
    from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
    from pysrc.artifact_registry.bundle_writer import BundleWriter

    writer = BundleWriter(tmp_path)
    store = BundleBacktestArtifactStore(writer)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=PairsConfig(zscore_window=60))
    feats = pd.DataFrame({"spread_SPY_QQQ": [0.0], "spread_SPY_QQQ_z60": [0.0]})  # no hl column
    strat.write_artifacts(
        store, signals=pd.Series([0]), feats=feats, run_id="test", evaluation_window="w"
    )
    report = __import__("json").loads((tmp_path / "stat_validity_report.json").read_text())
    assert report.get("half_life_bars") is None


def test_write_artifacts_empty_valid_hl(tmp_path):
    """write_artifacts when hl column is all NaN sets half_life_bars to None."""
    from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
    from pysrc.artifact_registry.bundle_writer import BundleWriter

    writer = BundleWriter(tmp_path)
    store = BundleBacktestArtifactStore(writer)
    strat = StatArbPairsStrategy("SPY", "QQQ", config=PairsConfig(zscore_window=60))
    feats = pd.DataFrame(
        {
            "spread_SPY_QQQ": [0.0, 0.0],
            "spread_SPY_QQQ_z60": [0.0, 0.0],
            "hl_spread_SPY_QQQ": [float("nan"), float("nan")],
        }
    )
    strat.write_artifacts(
        store, signals=pd.Series([0, 0]), feats=feats, run_id="test", evaluation_window="w"
    )
    report = __import__("json").loads((tmp_path / "stat_validity_report.json").read_text())
    assert report.get("half_life_bars") is None
