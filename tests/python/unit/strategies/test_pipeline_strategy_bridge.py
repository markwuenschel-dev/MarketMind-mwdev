# ADR-001 Gap 1 (Option A): two-stage registry lookup, graph executor path, cache, mutation guard.
from __future__ import annotations

import contextlib
import random
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from pysrc.backtesting.contracts.types import PitMeta
from pysrc.data.dataview import DataView
from pysrc.strategies.migrated_strategies import (
    BollingerBandsStrategy,
    MACDStrategy,
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    RSIStrategy,
)
from pysrc.strategies.pipeline_strategy import (
    FeaturePlan,
    FeatureStep,
    MaterializationError,
    PipelineStrategy,
    StrategyContext,
    _resolve_feature_op,
    materialize_features,
)

try:
    import pandas as pd
    import polars as pl
except Exception:
    pl = None
    pd = None


pytestmark = pytest.mark.determinism("d1")


@pytest.fixture
def registry_sandbox():
    import pysrc.strategies.pipeline_strategy as ps
    from pysrc.preprocessor.graph import factory as gf

    feature_ops_orig = dict(getattr(ps, "_FEATURE_OPS", {}))
    op_registry_orig = dict(getattr(gf, "_OP_REGISTRY", {}))
    alias_map_orig = dict(getattr(gf, "_ALIAS_MAP", {}))
    yield SimpleNamespace(ps=ps, gf=gf)
    ps._FEATURE_OPS.clear()
    ps._FEATURE_OPS.update(feature_ops_orig)
    gf._OP_REGISTRY.clear()
    gf._OP_REGISTRY.update(op_registry_orig)
    gf._ALIAS_MAP.clear()
    gf._ALIAS_MAP.update(alias_map_orig)


@pytest.fixture
def ctx_polars(tmp_path):
    if pl is None or pd is None:
        pytest.skip("polars/pandas not available")
    prices = pd.DataFrame({"close": [10.0, 11.0, 12.0]})
    return StrategyContext(prices=prices, backend="polars", cache_dir=tmp_path)


def _pit_meta() -> PitMeta:
    return PitMeta(
        as_of="2024-01-04T00:00:00",
        source="pysrc.data.dataview.DataView",
        knowledge_cutoff="2024-01-04",
    )


def _governed_history_frame() -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("pandas not available")
    source = pd.DataFrame(
        {
            "symbol": ["SPY"] * 5,
            "valid_time": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "knowledge_time": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 6),
            ],
            "close": [100.0, 102.0, 101.0, 103.0, 999.0],
            "high": [101.0, 103.0, 102.0, 104.0, 1000.0],
            "low": [99.0, 101.0, 100.0, 102.0, 998.0],
            "volume": [1000.0, 1100.0, 900.0, 1200.0, 999999.0],
        }
    )
    dataview = DataView()
    dataview.register_source(source)

    snapshots: list[pd.DataFrame] = []
    for knowledge_date in [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]:
        snapshot = dataview.as_of(["SPY"], ["close", "high", "low", "volume"], knowledge_date)
        assert not snapshot.empty
        row = snapshot.copy()
        row["price"] = row["close"].astype(float)
        row.index = pd.DatetimeIndex([pd.Timestamp(knowledge_date)])
        snapshots.append(row)

    history = pd.concat(snapshots, axis=0)
    history.attrs["source"] = {"kind": "raw-dataset-handle"}
    return history


def test_resolve_feature_op_prefers_op_registry_over_feature_ops(registry_sandbox):
    ps, gf = registry_sandbox.ps, registry_sandbox.gf
    op_name = "bridge.test_op"
    gf._OP_REGISTRY[op_name] = type("FakeOp", (), {"name": op_name})()
    ps._FEATURE_OPS[op_name] = lambda df, *a, **kw: df
    step = FeatureStep(op=op_name, inputs=("close",), kwargs={})
    kind, resolved = _resolve_feature_op(step)
    assert kind == "op_registry"
    assert resolved == op_name


def test_pipeline_strategy_init_does_not_mutate_global_rng(monkeypatch):
    def fail_seed(*args, **kwargs):
        pytest.fail("PipelineStrategy must not mutate process-global RNG state.")

    monkeypatch.setattr(random, "seed", fail_seed)
    monkeypatch.setattr(np.random, "seed", fail_seed)

    strategy = PipelineStrategy(random_state=123)

    assert strategy.random_state == 123


def test_resolve_feature_op_falls_back_to_feature_ops_when_not_in_op_registry(registry_sandbox):
    ps, gf = registry_sandbox.ps, registry_sandbox.gf
    op_name = "legacy.only"
    gf._OP_REGISTRY.pop(op_name, None)
    ps._FEATURE_OPS[op_name] = lambda df, *a, **kw: df
    step = FeatureStep(op=op_name, inputs=("close",), kwargs={})
    kind, resolved = _resolve_feature_op(step)
    assert kind == "feature_ops"
    assert resolved == op_name


def test_materialize_features_unknown_op_message_contract(registry_sandbox, ctx_polars):
    ps, gf = registry_sandbox.ps, registry_sandbox.gf
    gf._OP_REGISTRY.clear()
    ps._FEATURE_OPS.clear()
    gf._OP_REGISTRY["known.alpha"] = type("A", (), {"name": "known.alpha"})()
    ps._FEATURE_OPS["known.beta"] = lambda df, *a, **kw: df
    plan = FeaturePlan.from_steps([FeatureStep(op="does.not.exist", inputs=("close",), kwargs={})])
    with pytest.raises(MaterializationError) as exc_info:
        materialize_features(ctx_polars, plan, price_col="close")
    msg = str(exc_info.value)
    assert "does.not.exist" in msg
    assert "_OP_REGISTRY" in msg or "OP_REGISTRY" in msg
    assert "_FEATURE_OPS" in msg or "FEATURE_OPS" in msg
    assert "known.alpha" in msg or "known.beta" in msg


def test_materialize_features_graph_path_calls_build_graph_and_executor(
    registry_sandbox, ctx_polars, monkeypatch
):
    if pl is None or pd is None:
        pytest.skip("polars/pandas not available")
    _ps, gf = registry_sandbox.ps, registry_sandbox.gf
    op_name = "graph.simple"
    gf._OP_REGISTRY[op_name] = type("G", (), {"name": op_name})()

    build_calls = []
    exec_calls = []

    class FakeNode:
        def __init__(self, op_name: str):
            self.op = type("Op", (), {"name": op_name})()

        def to_ir(self):
            return {"op": self.op.name, "kind": "elementwise", "params": {}}

    def fake_build_graph(ops, params_map):
        build_calls.append((list(ops), dict(params_map)))
        other_op = "other.op"
        return SimpleNamespace(nodes=[FakeNode(other_op), FakeNode(op_name)])

    class FakeExecutor:
        def execute(self, plan_ir, data, group_by):
            exec_calls.append((plan_ir, data))
            return data.with_columns(pl.lit(999).alias("graph_out"))

    class FakeFactory:
        @staticmethod
        def create(backend: str):
            assert backend == "polars"
            return FakeExecutor()

    monkeypatch.setattr("pysrc.preprocessor.graph.factory.build_graph", fake_build_graph)
    monkeypatch.setattr("pysrc.preprocessor.graph.executor.ExecutorFactory", FakeFactory)

    plan = FeaturePlan.from_steps([FeatureStep(op=op_name, inputs=("close",), kwargs={"k": 2})])
    out = materialize_features(ctx_polars, plan, price_col="close")

    assert "graph_out" in out.columns
    assert len(build_calls) == 1
    assert build_calls[0][0] == [op_name]
    assert len(exec_calls) == 1
    plan_ir_arg = exec_calls[0][0]
    assert isinstance(plan_ir_arg, list)
    assert any(node.get("op") == op_name for node in plan_ir_arg)


def test_cache_invalidates_when_registry_fingerprint_changes(registry_sandbox, ctx_polars):
    if pl is None or pd is None:
        pytest.skip("polars/pandas not available")
    ps, gf = registry_sandbox.ps, registry_sandbox.gf
    ps._FEATURE_OPS.clear()
    gf._OP_REGISTRY.clear()
    counter = {"n": 0}

    def counting_op(df, *a, **kw):
        counter["n"] += 1
        if pl is not None and hasattr(df, "with_columns"):
            return df.with_columns(pl.lit(float(counter["n"])).alias("counted"))
        df["counted"] = float(counter["n"])
        return df

    op_name = "legacy.count"
    ps._FEATURE_OPS[op_name] = counting_op
    plan = FeaturePlan.from_steps(
        [FeatureStep(op=op_name, inputs=("close",), kwargs={}, out="counted")]
    )

    out1 = materialize_features(ctx_polars, plan, price_col="close")
    out2 = materialize_features(ctx_polars, plan, price_col="close")
    assert "counted" in out1.columns
    assert "counted" in out2.columns
    first_run_count = counter["n"]

    ps._FEATURE_OPS["legacy.new_key"] = lambda df, *a, **kw: df
    out3 = materialize_features(ctx_polars, plan, price_col="close")
    assert "counted" in out3.columns
    assert counter["n"] == first_run_count + 1


def test_legacy_mutation_guard_prevents_overwrite(registry_sandbox, ctx_polars):
    if pl is None or pd is None:
        pytest.skip("polars/pandas not available")
    ps = registry_sandbox.ps

    def bad_overwrite(df, *a, **kw):
        if pl is not None and hasattr(df, "with_columns"):
            return df.with_columns(pl.lit(123).alias("x"))
        df["x"] = 123
        return df

    ps._FEATURE_OPS["legacy.bad"] = bad_overwrite
    prices_with_x = pd.DataFrame({"close": [10.0, 11.0], "x": [1.0, 2.0]})
    ctx = StrategyContext(prices=prices_with_x, backend="polars", cache_dir=ctx_polars.cache_dir)
    plan = FeaturePlan.from_steps(
        [FeatureStep(op="legacy.bad", inputs=("close",), kwargs={}, out="x")]
    )
    with pytest.raises(MaterializationError) as exc_info:
        materialize_features(ctx, plan, price_col="close")
    msg = str(exc_info.value).lower()
    assert "x" in msg
    assert "overwrite" in msg or "collision" in msg or "already present" in msg


def test_pairs_ops_resolve_to_op_registry(registry_sandbox):
    gf = registry_sandbox.gf
    with contextlib.suppress(Exception):
        gf.register_builtin_ops()
    for op_name in ("pairs.beta", "pairs.spread"):
        step = FeatureStep(op=op_name, inputs=("close",), kwargs={})
        kind, _ = _resolve_feature_op(step)
        assert kind == "op_registry"


def test_pairs_beta_and_spread_polars_lowerings_not_implemented(registry_sandbox, ctx_polars):
    if pl is None or pd is None:
        pytest.skip("polars/pandas not available")
    gf = registry_sandbox.gf
    with contextlib.suppress(Exception):
        gf.register_builtin_ops()
    prices = pd.DataFrame({"A.close": [10.0, 11.0, 12.0], "B.close": [9.5, 10.5, 10.0]})
    ctx = StrategyContext(prices=prices, backend="polars", cache_dir=ctx_polars.cache_dir)
    plan = FeaturePlan.from_steps(
        [
            FeatureStep(op="pairs.beta", inputs=(), kwargs={"a": "A", "b": "B"}),
            FeatureStep(op="pairs.spread", inputs=(), kwargs={"a": "A", "b": "B"}),
        ]
    )
    try:
        out = materialize_features(ctx, plan, price_col=None)
    except NotImplementedError:
        pytest.xfail("pairs.* polars lowering not implemented yet")
    assert "beta_A_B" in out.columns
    assert "spread_A_B" in out.columns


def test_legacy_mutation_guard_blocks_unexpected_extra_outputs(registry_sandbox, ctx_polars):
    if pl is None or pd is None:
        pytest.skip("polars/pandas not available")
    ps = registry_sandbox.ps

    def multi_out(df, *a, **kw):
        df = df.with_columns(pl.lit(1).alias("one"))
        return df.with_columns(pl.lit(2).alias("two"))

    ps._FEATURE_OPS["legacy.multi"] = multi_out
    plan = FeaturePlan.from_steps(
        [FeatureStep(op="legacy.multi", inputs=("close",), kwargs={}, out="one")]
    )
    with pytest.raises(MaterializationError) as exc_info:
        materialize_features(ctx_polars, plan, price_col="close")
    msg = str(exc_info.value).lower()
    assert "unexpected" in msg or "two" in msg or "extra" in msg


def test_legacy_mutation_guard_blocks_column_drop(registry_sandbox, ctx_polars):
    if pl is None or pd is None:
        pytest.skip("polars/pandas not available")
    ps = registry_sandbox.ps

    def drop_y(df, *a, **kw):
        if pl is not None and hasattr(df, "select"):
            return df.select([c for c in df.columns if c != "y"])
        return df.drop("y", axis=1)

    ps._FEATURE_OPS["legacy.drop_y"] = drop_y
    prices = pd.DataFrame({"close": [10.0, 11.0], "y": [1.0, 2.0]})
    ctx = StrategyContext(prices=prices, backend="polars", cache_dir=ctx_polars.cache_dir)
    plan = FeaturePlan.from_steps([FeatureStep(op="legacy.drop_y", inputs=("close",), kwargs={})])
    with pytest.raises(MaterializationError) as exc_info:
        materialize_features(ctx, plan, price_col="close")
    msg = str(exc_info.value).lower()
    assert "dropped" in msg or "drop" in msg


def test_cache_invalidates_when_graph_op_version_changes(registry_sandbox, tmp_path, monkeypatch):
    ps, gf = registry_sandbox.ps, registry_sandbox.gf
    if pd is None:
        pytest.skip("pandas not available")
    ps._FEATURE_OPS.clear()
    gf._OP_REGISTRY.clear()

    class DummyOp:
        OP_VERSION = 1

    gf._OP_REGISTRY["graph.versioned"] = DummyOp

    calls = {"n": 0}

    def counting(df, *a, **kw):
        calls["n"] += 1
        df = df.copy()
        df["v"] = float(calls["n"])
        return df

    ps._FEATURE_OPS["graph.versioned"] = counting
    prices = pd.DataFrame({"close": [10.0, 11.0, 12.0]})
    ctx = StrategyContext(prices=prices, backend="pandas", cache_dir=tmp_path)
    plan = FeaturePlan.from_steps(
        [FeatureStep(op="graph.versioned", inputs=("close",), kwargs={"out": "v"})]
    )

    keys: list[str] = []

    def fake_get(self, key: str):
        keys.append(key)
        return None

    def fake_set(self, key: str, value: object) -> None:
        return None

    monkeypatch.setattr("pysrc.strategies.pipeline_strategy._Cache.get", fake_get)
    monkeypatch.setattr("pysrc.strategies.pipeline_strategy._Cache.set", fake_set)

    materialize_features(ctx, plan, price_col="close")
    first_key = keys[-1]

    DummyOp.OP_VERSION = 2
    materialize_features(ctx, plan, price_col="close")
    second_key = keys[-1]

    assert first_key != second_key


def test_supported_builtin_feature_plans_emit_only_canonical_ops() -> None:
    legacy_ops = {"RSI", "MACD", "ROLL_MEAN", "ROLL_STD", "Z_SCORE", "EMA"}
    strategies = [
        (RSIStrategy(), {"technical.RSI"}),
        (MACDStrategy(), {"technical.MACD_line_signal"}),
        (BollingerBandsStrategy(), {"technical.Bollinger"}),
        (MeanReversionStrategy(), {"scaling.zscore_roll"}),
        (MovingAverageCrossoverStrategy(ma_type="sma"), {"technical.SMA"}),
        (MovingAverageCrossoverStrategy(ma_type="ema"), {"technical.EMA"}),
    ]

    for strategy, expected_ops in strategies:
        emitted_ops = {step.op for step in strategy.features_plan().steps}
        assert emitted_ops == expected_ops
        assert emitted_ops.isdisjoint(legacy_ops)


def test_governed_context_blocks_direct_feature_ops_with_materialization_error(
    registry_sandbox, tmp_path, monkeypatch
) -> None:
    if pd is None:
        pytest.skip("pandas not available")
    ps = registry_sandbox.ps
    plan = FeaturePlan.from_steps(
        [FeatureStep(op="RSI", inputs=("price",), args=(2,), kwargs={"out": "rsi2"})]
    )
    ctx = StrategyContext(
        prices=pd.DataFrame({"price": [100.0, 101.0, 102.0]}),
        backend="polars",
        cache_dir=tmp_path,
        pit_provenance=_pit_meta(),
    )

    monkeypatch.setattr(ps, "_resolve_feature_op", lambda step: ("feature_ops", "RSI"))

    with pytest.raises(MaterializationError, match="_FEATURE_OPS|canonical graph lowering"):
        materialize_features(ctx, plan, price_col="price")


def test_governed_bridge_uses_dataview_snapshot_history_and_never_calls_legacy_runtime(
    registry_sandbox, tmp_path, monkeypatch
) -> None:
    if pl is None or pd is None:
        pytest.skip("polars/pandas not available")
    ps, gf = registry_sandbox.ps, registry_sandbox.gf
    gf.register_builtin_ops()

    history = _governed_history_frame()
    ctx = StrategyContext(
        prices=history[["price", "close", "high", "low", "volume"]],
        backend="polars",
        cache_dir=tmp_path,
        pit_provenance=_pit_meta(),
    )
    plan = FeaturePlan.from_steps(
        [FeatureStep(op="RSI", inputs=("price",), args=(2,), kwargs={"out": "rsi2"})]
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "direct _FEATURE_OPS execution should not be used for governed bridge execution"
        )

    monkeypatch.setitem(ps._FEATURE_OPS, "RSI", fail_if_called)
    out = materialize_features(ctx, plan, price_col="price")

    assert "rsi2" in out.columns
    assert out.height == 4
    assert float(out["price"][0]) == 100.0


def test_governed_graph_execution_never_exposes_raw_source_handle(
    registry_sandbox, tmp_path, monkeypatch
) -> None:
    if pl is None or pd is None:
        pytest.skip("polars/pandas not available")
    registry_sandbox.gf.register_builtin_ops()
    history = _governed_history_frame()
    ctx = StrategyContext(
        prices=history[["price", "close", "high", "low", "volume"]],
        backend="polars",
        cache_dir=tmp_path,
        pit_provenance=_pit_meta(),
    )
    plan = FeaturePlan.from_steps(
        [
            FeatureStep(
                "technical.SMA",
                inputs=("price",),
                kwargs={"input_col": "price", "window": 2, "out_col": "sma2"},
            )
        ]
    )

    import pysrc.strategies.pipeline_strategy as ps

    seen: dict[str, object] = {}
    real_execute_graph_step = ps._execute_graph_step

    def wrapped(step, resolved_key, feats):
        seen["has_source_attr"] = hasattr(feats, "source")
        seen["has_dataset_attr"] = hasattr(feats, "dataset")
        seen["attrs"] = getattr(feats, "attrs", None)
        return real_execute_graph_step(step, resolved_key, feats)

    monkeypatch.setattr(ps, "_execute_graph_step", wrapped)
    out = materialize_features(ctx, plan, price_col="price")

    assert "sma2" in out.columns
    assert seen["has_source_attr"] is False
    assert seen["has_dataset_attr"] is False
    assert seen["attrs"] in (None, {})


def test_pandas_backend_converts_only_after_graph_execution(
    registry_sandbox, tmp_path, monkeypatch
) -> None:
    if pl is None or pd is None:
        pytest.skip("polars/pandas not available")
    registry_sandbox.gf.register_builtin_ops()
    prices = pd.DataFrame({"price": [100.0, 101.0, 102.0, 103.0]})
    ctx = StrategyContext(
        prices=prices,
        backend="pandas",
        cache_dir=tmp_path,
        pit_provenance=_pit_meta(),
    )
    plan = FeaturePlan.from_steps(
        [
            FeatureStep(
                "technical.SMA",
                inputs=("price",),
                kwargs={"input_col": "price", "window": 2, "out_col": "ma_short"},
            ),
            FeatureStep(
                "technical.EMA",
                inputs=("price",),
                kwargs={"input_col": "price", "span": 3, "out_col": "ma_long"},
            ),
        ]
    )

    import pysrc.strategies.pipeline_strategy as ps

    graph_input_types: list[str] = []
    real_execute_graph_step = ps._execute_graph_step

    def wrapped(step, resolved_key, feats):
        graph_input_types.append(type(feats).__name__)
        assert isinstance(feats, pl.DataFrame)
        return real_execute_graph_step(step, resolved_key, feats)

    monkeypatch.setattr(ps, "_execute_graph_step", wrapped)
    out = materialize_features(ctx, plan, price_col="price")

    assert isinstance(out, pd.DataFrame)
    assert {"ma_short", "ma_long"}.issubset(out.columns)
    assert graph_input_types == ["DataFrame", "DataFrame"]


def test_governed_vwap_fails_closed_on_daily_history(registry_sandbox, tmp_path) -> None:
    if pd is None:
        pytest.skip("pandas not available")
    registry_sandbox.gf.register_builtin_ops()
    history = _governed_history_frame()
    ctx = StrategyContext(
        prices=history[["close", "volume"]],
        backend="polars",
        cache_dir=tmp_path,
        pit_provenance=_pit_meta(),
    )
    plan = FeaturePlan.from_steps(
        [
            FeatureStep(
                "technical.VWAP",
                inputs=("close",),
                kwargs={"price_col": "close", "volume_col": "volume", "out_col": "vwap"},
            )
        ]
    )

    with pytest.raises(MaterializationError, match="VWAP|daily approximation|session"):
        materialize_features(ctx, plan, price_col="close")
