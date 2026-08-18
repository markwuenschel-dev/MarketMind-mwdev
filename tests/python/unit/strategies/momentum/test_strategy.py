from __future__ import annotations

import importlib

import pandas as pd
import pytest

import pysrc.strategies.momentum.strategy as momentum_strategy_module
from pysrc.backtesting.contracts.types import PitMeta
from pysrc.preprocessor.graph.factory import register_builtin_ops, registry_snapshot
from pysrc.preprocessor.graph.ops_custom import (
    IndustryScore,
    ResidualKF,
    ResidualOLS,
    VolScale,
    XSecRank,
)
from pysrc.strategies.momentum.alpha_ir import AlphaIR
from pysrc.strategies.momentum.exceptions import FeatureFlagError
from pysrc.strategies.momentum.strategy import MomentumStrategy
from pysrc.strategies.pipeline_strategy import (
    FeaturePlan,
    FeatureStep,
    MaterializationError,
    PipelineError,
    StrategyContext,
    StrategyRegistry,
    TradeIntent,
    ValidationError,
)

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def _pit_meta() -> PitMeta:
    return PitMeta(
        as_of="2024-01-04T00:00:00",
        source="pysrc.data.dataview.DataView",
        knowledge_cutoff="2024-01-04",
    )


def _ctx(tmp_path, *, pit_provenance: PitMeta | None = None) -> StrategyContext:
    return StrategyContext(
        prices=pd.DataFrame({"close": [100.0 + idx for idx in range(20)]}),
        backend="pandas",
        cache_dir=tmp_path,
        pit_provenance=pit_provenance,
    )


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "returns": [0.01 * ((idx % 5) - 2) for idx in range(20)],
            "realized_vol_60": [0.2 + (idx * 0.001) for idx in range(20)],
            "mom_scaled": [0.05 * ((idx % 7) - 3) for idx in range(20)],
        }
    )


def _panel_features(asset_count: int = 12, periods: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    assets = [f"A{idx:02d}" for idx in range(asset_count)]
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    returns: list[float] = []
    signal: list[float] = []
    for _date in dates:
        for asset_idx in range(asset_count):
            centered = asset_idx - ((asset_count - 1) / 2.0)
            returns.append(centered / 10.0)
            signal.append(centered)
    return pd.DataFrame(
        {
            "returns": returns,
            "realized_vol_60": [0.2] * len(index),
            "mom_scaled": signal,
        },
        index=index,
    )


def test_package_import_and_registry_aliases() -> None:
    package = importlib.import_module("pysrc.strategies.momentum")
    assert package.MomentumStrategy is MomentumStrategy
    expected = [
        "momentum",
        "momentum_tsmom",
        "momentum_dual",
        "momentum_industry",
        "momentum_residual",
        "momentum_kalman",
        "momentum_ensemble",
        "momentum_ml",
    ]
    for key in expected:
        assert StrategyRegistry.get(key) is MomentumStrategy


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


def test_momentum_ops_to_ir_surface() -> None:
    ops = [
        XSecRank(col="returns", window=252, out_col="mom_rank").to_ir(),
        VolScale(
            signal_col="mom_rank",
            vol_col="realized_vol_60",
            target_vol=0.15,
            max_leverage=2.0,
            out_col="mom_scaled",
        ).to_ir(),
        ResidualOLS(
            asset_ret_col="returns",
            factor_ret_cols=["market_return"],
            window=63,
            out_col="residual_ols",
        ).to_ir(),
        ResidualKF(
            asset_ret_col="returns",
            factor_ret_cols=["market_return"],
            process_noise=1e-4,
            obs_noise=1e-3,
            out_col="residual_kf",
        ).to_ir(),
        IndustryScore(
            ret_col="returns",
            sector_col="sector",
            window=63,
            out_col="industry_score",
        ).to_ir(),
    ]
    for ir in ops:
        assert {"op", "kind", "params", "requires", "provides"} <= set(ir)


def test_invalid_variant_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported momentum variant"):
        MomentumStrategy(variant="garbage")


def test_kalman_without_flag_raises_feature_flag_error() -> None:
    with pytest.raises(FeatureFlagError):
        MomentumStrategy(variant="residual_kalman").features_plan()


def test_features_plan_signature_is_stable() -> None:
    strategy = MomentumStrategy(variant="xsec", target_vol=0.20)
    assert strategy.features_plan().signature() == strategy.features_plan().signature()


def test_generate_signal_returns_alpha_ir_for_live_variants() -> None:
    for variant in ("xsec", "tsmom", "dual", "industry", "residual_ols"):
        alpha_ir = MomentumStrategy(variant=variant).generate_signal(_features())
        assert isinstance(alpha_ir, AlphaIR)


def test_generate_signal_emits_zero_stub_task_embedding() -> None:
    alpha_ir = MomentumStrategy(variant="xsec").generate_signal(_features())
    assert alpha_ir.task_embedding.shape == (64,)
    assert alpha_ir.task_embedding.dtype == "float32"


def test_generate_signal_emits_book_membership_and_fail_closed_beta_reversal_status() -> None:
    alpha_ir = MomentumStrategy(variant="xsec").generate_signal(
        _panel_features(asset_count=3, periods=1)
    )
    membership = alpha_ir.diagnostics["book_membership"]
    beta_reversal = alpha_ir.diagnostics["beta_reversal_score"]
    assert membership["by_symbol"]["A00"] == "short"
    assert membership["by_symbol"]["A02"] == "long"
    assert membership["counts"] == {"long": 1, "short": 1, "flat": 1}
    assert beta_reversal["status"] == "UNAVAILABLE"


def test_generate_signal_requires_pit_metadata_in_governed_mode(tmp_path) -> None:
    strat = MomentumStrategy(variant="xsec", governed=True)
    strat._active_ctx = _ctx(tmp_path, pit_provenance=None)
    with pytest.raises(ValidationError, match="ctx.pit_provenance"):
        strat.generate_signal(_features())


def test_cross_sectional_ic_is_none_below_min_universe() -> None:
    alpha_ir = MomentumStrategy(variant="xsec").generate_signal(_panel_features(asset_count=9))
    assert alpha_ir.information_coefficient is None


def test_cross_sectional_ic_uses_panel_when_available() -> None:
    alpha_ir = MomentumStrategy(variant="xsec").generate_signal(_panel_features(asset_count=12))
    assert alpha_ir.information_coefficient == pytest.approx(1.0)


def test_governed_feature_ops_plan_raises_materialization_error(tmp_path) -> None:
    class LegacyMomentum(MomentumStrategy):
        def features_plan(self) -> FeaturePlan:
            return FeaturePlan.from_steps(
                [FeatureStep(op="PCT_CHANGE", inputs=("close",), kwargs={"out": "ret_1d"})]
            )

    with pytest.raises(MaterializationError, match="_FEATURE_OPS"):
        LegacyMomentum().generate_trade_intent(_ctx(tmp_path, pit_provenance=_pit_meta()))


def test_generate_signal_requires_final_signal_column() -> None:
    with pytest.raises(ValidationError, match="final signal column"):
        MomentumStrategy(variant="xsec").generate_signal(
            pd.DataFrame({"returns": [0.1], "realized_vol_60": [0.2]})
        )


def test_generate_signal_rejects_non_dataframe_inputs() -> None:
    with pytest.raises(TypeError, match="expects pandas or polars features"):
        MomentumStrategy(variant="xsec").generate_signal(object())  # type: ignore[arg-type]


def test_cross_sectional_ic_uses_date_and_asset_columns_when_present() -> None:
    rows: list[dict[str, object]] = []
    for date in pd.date_range("2024-01-01", periods=2, freq="D"):
        for asset_idx in range(10):
            centered = asset_idx - 4.5
            rows.append(
                {
                    "date": date,
                    "asset": f"A{asset_idx:02d}",
                    "returns": centered / 10.0,
                    "realized_vol_60": 0.2,
                    "mom_scaled": centered,
                }
            )
    alpha_ir = MomentumStrategy(variant="xsec").generate_signal(pd.DataFrame(rows))
    assert alpha_ir.information_coefficient == pytest.approx(1.0)


def test_cross_sectional_ic_returns_none_without_asset_keys() -> None:
    alpha_ir = MomentumStrategy(variant="xsec").generate_signal(
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5, freq="D"),
                "returns": [0.1, 0.2, 0.3, 0.4, 0.5],
                "realized_vol_60": [0.2] * 5,
                "mom_scaled": [1, 2, 3, 4, 5],
            }
        )
    )
    assert alpha_ir.information_coefficient is None


def test_book_membership_payload_handles_simple_index() -> None:
    alpha_ir = MomentumStrategy(variant="xsec").generate_signal(
        pd.DataFrame(
            {
                "returns": [0.1, -0.1, 0.0],
                "realized_vol_60": [0.2, 0.2, 0.2],
                "mom_scaled": [1.0, -1.0, 0.0],
            },
            index=["A", "B", "C"],
        )
    )
    assert alpha_ir.diagnostics["book_membership"]["counts"] == {
        "long": 1,
        "short": 1,
        "flat": 1,
    }


def test_generate_trade_intent_applies_regime_sizer_and_risk(monkeypatch, tmp_path) -> None:
    strategy = MomentumStrategy(variant="xsec")

    class _Regime:
        def gate(self, features):
            return 0.5

    class _Sizer:
        def size(self, signal):
            return signal * 2.0

    class _Risk:
        def clamp(self, *, weights, prices):
            return weights.clip(-0.25, 0.25)

    strategy.regime = _Regime()
    strategy.sizer = _Sizer()
    strategy.risk = _Risk()

    feats = pd.DataFrame(
        {
            "returns": [0.1, -0.1],
            "realized_vol_60": [0.2, 0.2],
            "mom_scaled": [1.0, -1.0],
        },
        index=["A", "B"],
    )
    monkeypatch.setattr(momentum_strategy_module, "materialize_features", lambda ctx, plan: feats)

    trade_intent = strategy.generate_trade_intent(_ctx(tmp_path, pit_provenance=_pit_meta()))
    assert isinstance(trade_intent, TradeIntent)
    assert list(trade_intent.weights.tolist()) == [0.25, -0.25]


@pytest.mark.parametrize(
    ("component_name", "exc", "message"),
    [
        ("regime", ValueError("bad gate"), "regime gating failed"),
        ("sizer", ValueError("bad size"), "position sizing failed"),
        ("risk", ValueError("bad clamp"), "risk clamp failed"),
    ],
)
def test_generate_trade_intent_wraps_component_failures(
    monkeypatch, tmp_path, component_name: str, exc: Exception, message: str
) -> None:
    strategy = MomentumStrategy(variant="xsec")

    class _Regime:
        def gate(self, features):
            raise exc

    class _Sizer:
        def size(self, signal):
            raise exc

    class _Risk:
        def clamp(self, *, weights, prices):
            raise exc

    strategy.regime = _Regime() if component_name == "regime" else None
    strategy.sizer = _Sizer() if component_name == "sizer" else None
    strategy.risk = _Risk() if component_name == "risk" else None

    feats = pd.DataFrame(
        {
            "returns": [0.1, -0.1],
            "realized_vol_60": [0.2, 0.2],
            "mom_scaled": [1.0, -1.0],
        },
        index=["A", "B"],
    )
    monkeypatch.setattr(momentum_strategy_module, "materialize_features", lambda ctx, plan: feats)

    with pytest.raises(PipelineError, match=message):
        strategy.generate_trade_intent(_ctx(tmp_path, pit_provenance=_pit_meta()))
