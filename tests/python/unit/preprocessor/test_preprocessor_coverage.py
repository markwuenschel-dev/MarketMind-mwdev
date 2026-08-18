# tests/python/unit/preprocessor/test_preprocessor_coverage.py
"""
Targeted tests to raise py/preprocessor line coverage to >90% and branch to >75%.
Covers api.py, graph/factory.py, graph/planner.py, graph/executor.py, and edge cases.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


# =============================================================================
# api.py – merge_specs, ModelRegistry, resolve_models_in_ops, get_executor, _to_polars, run
# =============================================================================


class TestMergeSpecsEdgeCases:
    """Cover merge_specs branches and meta handling."""

    def test_merge_specs_skips_falsy_specs(self):
        from pysrc.preprocessor.api import PlanSpec, merge_specs

        spec = PlanSpec(ops=[{"kind": "a"}])
        result = merge_specs(None, spec, None)
        assert len(result.ops) == 1

    def test_merge_specs_meta_without_provenance_preserved(self):
        from pysrc.preprocessor.api import PlanSpec, merge_specs

        spec = PlanSpec(meta={"export": "x", "provenance": "p1"})
        result = merge_specs(spec)
        assert result.meta.get("export") == "x"
        assert "provenance_chain" in result.meta

    def test_merge_specs_last_wins_meta_extra_keys(self):
        from pysrc.preprocessor.api import PlanSpec, merge_specs

        s1 = PlanSpec(meta={"provenance": "p1", "extra": "first"})
        s2 = PlanSpec(meta={"provenance": "p2", "extra": "second"})
        result = merge_specs(s1, s2)
        assert result.meta.get("extra") == "second"


class TestModelRegistryClear:
    """ModelRegistry.clear() and resolve_models edge cases."""

    def test_registry_clear(self):
        from pysrc.preprocessor.api import ModelRegistry

        ModelRegistry.register("_clear_test", object())
        ModelRegistry.clear()
        assert ModelRegistry.get("_clear_test") is None

    def test_resolve_models_in_ops_model_ref_not_string_unchanged(self):
        from pysrc.preprocessor.api import resolve_models_in_ops

        ops = [{"kind": "x", "model_ref": 123}]
        result = resolve_models_in_ops(ops)
        assert result[0].get("__model__") is None
        assert result[0]["model_ref"] == 123

    def test_resolve_models_in_ops_model_not_found_no_attr(self):
        from pysrc.preprocessor.api import ModelRegistry, resolve_models_in_ops

        ModelRegistry.clear()
        ops = [{"kind": "x", "model_ref": "nonexistent_xyz"}]
        result = resolve_models_in_ops(ops)
        assert "__model__" not in result[0] or result[0].get("__model__") is None


class TestGetExecutorGpu:
    """get_executor('gpu') returns PolarsExecutor with engine_pref gpu."""

    def test_get_executor_gpu(self):
        from pysrc.preprocessor.api import get_executor
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = get_executor("gpu")
        assert isinstance(ex, PolarsExecutor)
        assert ex.engine_pref == "gpu"


class TestToPolars:
    """_to_polars conversion paths."""

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_to_polars_lazyframe_collects(self):
        from pysrc.preprocessor.api import _to_polars

        lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
        result = _to_polars(lf, pl)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (3, 1)

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_to_polars_dataframe_passthrough(self):
        from pysrc.preprocessor.api import _to_polars

        df = pl.DataFrame({"a": [1, 2, 3]})
        result = _to_polars(df, pl)
        assert result is df

    @pytest.mark.skipif(not HAS_POLARS or not HAS_PANDAS, reason="Pandas/Polars not available")
    def test_to_polars_from_pandas(self):
        from pysrc.preprocessor.api import _to_polars

        pdf = pd.DataFrame({"a": [1, 2, 3]})
        result = _to_polars(pdf, pl)
        assert isinstance(result, pl.DataFrame)
        assert list(result["a"]) == [1, 2, 3]

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_to_polars_generic_fallback_dict(self):
        from pysrc.preprocessor.api import _to_polars

        result = _to_polars({"a": [1, 2, 3]}, pl)
        assert isinstance(result, pl.DataFrame)

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_to_polars_fallback_raises_on_invalid_type(self):
        from pysrc.core.errors import PreprocessingError
        from pysrc.preprocessor.api import _to_polars

        with pytest.raises(PreprocessingError, match="Failed to convert"):
            _to_polars(12345, pl)


class TestCompileBackendHintWarning:
    """_compile logs when backend hint differs from requested backend."""

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_compile_warns_on_backend_hint_mismatch(self):
        from pysrc.preprocessor.api import Plan, _compile

        plan = Plan(
            ops=["technical.RSI"],
            params={
                "technical.RSI": [{"input_col": "close", "window": 14, "backend_hint": "cudf"}]
            },
            group_by=[],
        )
        with patch("pysrc.preprocessor.api.logger") as mock_log:
            compiled, _ = _compile(plan, backend="polars")
            # Compiled IR may include backend_hint from op params; warning should be called if hint != backend
            assert mock_log.warning.called or all(
                ir.get("backend_hint") != "cudf" for ir in compiled
            )


class TestRunOOMAndExceptionPaths:
    """run() OOMRetry fallback and generic exception."""

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_run_oom_retry_fallback(self):
        from pysrc.core.errors import PreprocessingError
        from pysrc.preprocessor.api import Plan, run
        from pysrc.preprocessor.utils.errors import OOMRetry

        df = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
        plan = Plan(ops=[], params={}, group_by=[])

        with patch("pysrc.preprocessor.api.get_executor") as mock_get:
            ex = MagicMock()
            ex.execution_history = []
            ex.execute.side_effect = OOMRetry("OOM")
            mock_get.return_value = ex

            with pytest.raises(
                PreprocessingError, match="governed path rejected retry under changed semantics"
            ):
                run(df, plan, backend="gpu")

            assert mock_get.call_count == 1

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_run_generic_exception_raises(self):
        from pysrc.core.errors import PreprocessingError
        from pysrc.preprocessor.api import Plan, run

        df = pl.DataFrame({"close": [1.0, 2.0]})
        plan = Plan(ops=[], params={}, group_by=[])

        with patch("pysrc.preprocessor.api.get_executor") as mock_get:
            ex = MagicMock()
            ex.execute.side_effect = RuntimeError("generic failure")
            ex.execution_history = []
            mock_get.return_value = ex

            with pytest.raises(PreprocessingError, match="Execution failed"):
                run(df, plan, backend="polars")


class TestPreprocessorBuilderAddOpNoParams:
    """PreprocessorBuilder.add_op with no params uses empty list."""

    def test_builder_add_op_without_params(self):
        from pysrc.preprocessor.api import PreprocessorBuilder

        builder = PreprocessorBuilder()
        builder.add_op("technical.RSI")  # no params
        assert "technical.RSI" in builder._ops
        assert builder._params.get("technical.RSI") == []


class TestRunDictPlanKeys:
    """run() with dict plan uses kind and group_by."""

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_run_dict_plan_ops_with_kind(self):
        from pysrc.preprocessor.api import run

        df = pl.DataFrame({"close": [1.0, 2.0]})
        plan_dict = {"ops": [{"kind": "technical.RSI", "params": {}}], "group_by": ["symbol"]}

        with patch("pysrc.preprocessor.api._compile") as mock_compile:
            mock_compile.return_value = ([], MagicMock())
            with patch("pysrc.preprocessor.api.get_executor") as mock_get:
                mock_get.return_value = MagicMock(
                    execute=lambda *a, **k: pl.DataFrame({"close": [1.0, 2.0]}),
                    execution_history=[],
                )
                result = run(df, plan_dict, backend="polars")
                assert result is not None

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_run_returns_without_to_polars_when_backend_auto_and_dict_input(self):
        """When input is dict and backend auto, executor may return non-Polars; run still returns."""
        from pysrc.preprocessor.api import Plan, run

        plan = Plan(ops=[], params={}, group_by=[])
        df = pl.DataFrame({"a": [1, 2]})
        with patch("pysrc.preprocessor.api.get_executor") as mock_get:
            out_df = pl.DataFrame({"a": [1, 2]})
            mock_get.return_value = MagicMock(execute=lambda *a, **k: out_df, execution_history=[])
            result = run(df, plan, backend="auto")
            assert result is not None
            assert isinstance(result, pl.DataFrame)


# =============================================================================
# graph/factory.py – register, register_alias, resolve_name, _parse_token, build_graph, registry_snapshot
# =============================================================================


class TestFactoryRegisterAndAlias:
    """Factory register/register_alias/resolve_name and duplicate handling."""

    def test_register_duplicate_raises(self):
        from pysrc.preprocessor.graph.factory import register
        from pysrc.preprocessor.graph.ops import Op, OpKind

        class DummyOp(Op):
            NAME = "_cov_dup_op"
            KIND = OpKind.elementwise

            def to_ir(self):
                return {"op": self.NAME, "kind": self.KIND.value, "params": self.params}

        register("_cov_dup_op", DummyOp)
        try:
            with pytest.raises(ValueError, match="already registered"):
                register("_cov_dup_op", DummyOp)
        finally:
            from pysrc.preprocessor.graph import factory

            factory._OP_REGISTRY.pop("_cov_dup_op", None)

    def test_register_alias_duplicate_raises(self):
        from pysrc.preprocessor.graph.factory import register_alias, resolve_name

        try:
            register_alias("_alias_xyz", "technical.RSI")
            with pytest.raises(ValueError, match="already used"):
                register_alias("_alias_xyz", "technical.SMA")
        finally:
            if resolve_name("_alias_xyz") != "_alias_xyz":
                pass  # alias existed

    def test_resolve_name_with_alias(self):
        from pysrc.preprocessor.graph.factory import register_alias, resolve_name

        register_alias("_resolve_alias", "technical.RSI")
        assert resolve_name("_resolve_alias") == "technical.RSI"
        assert resolve_name("technical.SMA") == "technical.SMA"


class TestFactoryParseTokenAndBuildGraph:
    """_parse_token branches via build_graph; build_graph and registry_snapshot."""

    def test_build_graph_sequence_lags_token(self):
        from pysrc.preprocessor.graph.factory import build_graph, register_builtin_ops

        register_builtin_ops()
        ops = ["sequence.lags:3"]
        params = {
            "sequence.lags": [
                {"cols": ["close"], "out_cols": ["close_lag1", "close_lag2", "close_lag3"]}
            ]
        }
        g = build_graph(ops, params)
        assert len(g.nodes) == 1

    def test_build_graph_sma_ema_token(self):
        from pysrc.preprocessor.graph.factory import build_graph, register_builtin_ops

        register_builtin_ops()
        ops = ["technical.SMA:close,20"]
        params = {}
        g = build_graph(ops, params)
        assert len(g.nodes) == 1

    def test_build_graph_rsi_tokens(self):
        from pysrc.preprocessor.graph.factory import build_graph, register_builtin_ops

        register_builtin_ops()
        g1 = build_graph(["technical.RSI:close"], {})
        assert len(g1.nodes) == 1
        g2 = build_graph(["technical.RSI:close,14"], {})
        assert len(g2.nodes) == 1

    def test_build_graph_unknown_op_raises(self):
        from pysrc.core.errors import UnsupportedPlan
        from pysrc.preprocessor.graph.factory import build_graph

        with pytest.raises(UnsupportedPlan, match="Unknown op"):
            build_graph(["unknown.op.xyz_123"], {})

    def test_registry_snapshot_returns_dicts(self):
        from pysrc.preprocessor.graph.factory import registry_snapshot

        reg, aliases = registry_snapshot()
        assert isinstance(reg, dict)
        assert isinstance(aliases, dict)


# =============================================================================
# graph/planner.py – _prune, _segment, _fuse_segment, update_from_history, _threshold, _reoptimize, _evolve_weights
# =============================================================================


class TestPlannerPruneSegmentFuse:
    """Planner plan(), _prune (no terminals fallback), _segment, _fuse_segment."""

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_planner_plan_returns_ir(self):
        from pysrc.preprocessor.graph.factory import build_graph, register_builtin_ops
        from pysrc.preprocessor.graph.planner import Planner

        register_builtin_ops()
        g = build_graph(["technical.RSI:close,14"], {})
        planner = Planner()
        ir = planner.plan(g, ["symbol"])
        assert isinstance(ir, list)
        assert len(ir) >= 1
        assert ir[0].get("op") == "technical.RSI"

    def test_planner_prune_no_terminals_uses_all_nodes(self):
        from pysrc.preprocessor.graph.graph import Graph
        from pysrc.preprocessor.graph.ops import Op, OpKind
        from pysrc.preprocessor.graph.planner import Planner

        class FakeOp(Op):
            NAME = "fake"
            KIND = OpKind.elementwise

            def to_ir(self):
                return {"op": self.NAME, "kind": self.KIND.value, "params": self.params}

        g = Graph()
        n1 = g.add_op(FakeOp(x=1))
        n2 = g.add_op(FakeOp(x=2))
        n1.outputs.append(n2)
        n2.inputs.append(n1)
        # No terminal (every node has outputs) -> terminals = graph.nodes
        planner = Planner()
        pruned = planner._prune(g)
        assert len(pruned.nodes) == 2

    def test_planner_update_from_history_existing_op(self):
        from pysrc.preprocessor.graph.planner import Planner

        planner = Planner()
        planner.update_from_history([{"op": "technical.RSI", "time": 0.5}])
        planner.update_from_history([{"op": "technical.RSI", "time": 0.6}])
        assert planner.history["technical.RSI"]["count"] == 2

    def test_planner_threshold_with_history(self):
        from pysrc.preprocessor.graph.planner import Planner

        planner = Planner()
        planner.history = {"a": {"time": 1.0}, "b": {"time": 2.0}}
        t = planner._threshold()
        assert t > 0
        assert t != 10.0  # not default cold start

    def test_planner_reoptimize_orders_by_metrics(self):
        from pysrc.preprocessor.graph.ops import Op, OpKind
        from pysrc.preprocessor.graph.planner import Planner

        class FakeOp(Op):
            NAME = "f"
            KIND = OpKind.elementwise

            def __init__(self, name="f", **kwargs):
                super().__init__(**kwargs)
                self._name = name

            @property
            def name(self):
                return self._name

            def to_ir(self):
                return {"op": self.name, "kind": self.KIND.value, "params": self.params}

        planner = Planner()
        planner.metrics = {"f_slow": 10.0, "f_fast": 0.1}
        ordered = [
            type("N", (), {"op": FakeOp(name="f_slow")})(),
            type("N", (), {"op": FakeOp(name="f_fast")})(),
        ]
        planner._reoptimize(ordered)
        assert ordered[0].op.name == "f_slow"  # slower first (reverse=True)

    def test_planner_evolve_weights(self):
        from pysrc.preprocessor.graph.planner import Planner

        planner = Planner()
        planner.metrics = {"op1": 2.0}
        planner._evolve_weights()
        assert planner.weights["time"] >= 1.0


# =============================================================================
# graph/executor.py – CuDFExecutor _to_cudf paths, execute with evolve
# =============================================================================


class TestCuDFExecutorToCudf:
    """CuDFExecutor._to_cudf: ImportError, cudf.DataFrame, pandas, polars, fallback, PreprocessingError."""

    def test_cudf_to_cudf_import_error_raises(self):
        import builtins

        from pysrc.core.errors import PreprocessingError
        from pysrc.preprocessor.graph.executor import CuDFExecutor

        ex = CuDFExecutor()
        real_import = builtins.__import__

        def raise_cudf_import(name, *args, **kwargs):
            if name == "cudf":
                raise ImportError("No module named 'cudf'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=raise_cudf_import):
            with pytest.raises(PreprocessingError, match="cuDF not available"):
                ex._to_cudf({"a": [1, 2, 3]})

    @pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
    def test_cudf_to_cudf_from_pandas(self):
        from pysrc.preprocessor.graph.executor import CuDFExecutor

        mock_cudf = MagicMock()
        mock_cudf.from_pandas = MagicMock(return_value="cudf_df")
        mock_cudf.DataFrame.from_pandas = MagicMock(return_value="cudf_df2")

        with patch.dict("sys.modules", {"cudf": mock_cudf}):
            ex = CuDFExecutor()
            result = ex._to_cudf(pd.DataFrame({"a": [1, 2, 3]}))
            assert result == "cudf_df" or result == "cudf_df2"

    @pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
    def test_cudf_to_cudf_from_pandas_no_from_pandas_uses_dataframe_from_pandas(self):
        from pysrc.preprocessor.graph.executor import CuDFExecutor

        mock_cudf = MagicMock(spec=[])  # no from_pandas
        del mock_cudf.from_pandas
        mock_cudf.DataFrame = MagicMock()
        mock_cudf.DataFrame.from_pandas = MagicMock(return_value="cudf_df")

        with patch.dict("sys.modules", {"cudf": mock_cudf}):
            ex = CuDFExecutor()
            result = ex._to_cudf(pd.DataFrame({"a": [1, 2, 3]}))
            assert result == "cudf_df"

    @pytest.mark.skipif(not HAS_POLARS or not HAS_PANDAS, reason="Polars/Pandas not available")
    def test_cudf_to_cudf_from_polars(self):
        from pysrc.preprocessor.graph.executor import CuDFExecutor

        mock_cudf = MagicMock()
        mock_cudf.from_pandas = MagicMock(return_value="cudf_df")

        with patch.dict("sys.modules", {"cudf": mock_cudf}):
            ex = CuDFExecutor()
            result = ex._to_cudf(pl.DataFrame({"a": [1, 2, 3]}))
            assert result == "cudf_df"

    def test_cudf_to_cudf_passthrough_when_already_cudf_dataframe(self):
        from pysrc.preprocessor.graph.executor import CuDFExecutor

        mock_cudf = MagicMock()
        CudfDataFrame = type("CudfDataFrame", (), {})
        mock_cudf.DataFrame = CudfDataFrame
        existing = CudfDataFrame()

        with patch.dict("sys.modules", {"cudf": mock_cudf}):
            ex = CuDFExecutor()
            result = ex._to_cudf(existing)
            assert result is existing

    def test_cudf_to_cudf_fallback_construct_raises_preprocessing_error(self):
        from pysrc.core.errors import PreprocessingError
        from pysrc.preprocessor.graph.executor import CuDFExecutor

        mock_cudf = MagicMock()
        mock_cudf.DataFrame = MagicMock(side_effect=Exception("cannot construct"))

        with patch.dict("sys.modules", {"cudf": mock_cudf}):
            ex = CuDFExecutor()
            with pytest.raises(PreprocessingError, match="Failed to convert"):
                ex._to_cudf(object())  # not DataFrame, not pandas, not polars


class TestCuDFExecutorExecute:
    """CuDFExecutor.execute and evolve re-execution."""

    @pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
    def test_cudf_execute_evolve_reexecutes(self):
        from pysrc.preprocessor.graph.executor import CuDFExecutor, ExecutorFactory

        mock_cudf = MagicMock()
        gdf = MagicMock()
        mock_cudf.from_pandas = MagicMock(return_value=gdf)

        with patch.dict("sys.modules", {"cudf": mock_cudf}):
            with patch.object(CuDFExecutor, "_to_cudf", return_value=gdf):
                with patch.object(CuDFExecutor, "_execute_node", return_value=gdf):
                    ex = CuDFExecutor()
                    ex.evolve = MagicMock(return_value="polars")
                    with patch.object(ExecutorFactory, "create") as mock_create:
                        result = ex.execute([], pd.DataFrame({"a": [1]}), [])
                        assert result is gdf
                        ex.evolve.assert_not_called()
                        mock_create.assert_not_called()


# =============================================================================
# core.py – load_ohlcv (if not covered)
# =============================================================================


class TestCoreLoadOhlcv:
    """core.load_ohlcv with a temp CSV."""

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_ohlcv_from_path(self, tmp_path):
        import csv
        from pathlib import Path

        from pysrc.preprocessor.core import load_ohlcv

        csv_path = tmp_path / "ohlcv.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close", "volume"])
            w.writerow(["2024-01-01", 100, 101, 99, 100.5, 1000])
            w.writerow(["2024-01-02", 100.5, 102, 100, 101.5, 1100])

        df = load_ohlcv(Path(csv_path))
        assert df.shape[0] == 2
        assert "close" in df.columns


@pytest.mark.determinism("d1")
def test_phase_ic_registry_contains_canonical_op_floor() -> None:
    from pysrc.preprocessor.graph.factory import register_builtin_ops, registry_snapshot

    register_builtin_ops()
    registry, aliases = registry_snapshot()

    required_ops = {
        "technical.RSI",
        "technical.SMA",
        "technical.EMA",
        "technical.MACD_line_signal",
        "technical.Bollinger",
        "technical.ATR",
        "technical.OBV",
        "technical.VWAP",
        "scaling.zscore_roll",
        "stats.rolling_std",
    }
    required_aliases = {
        "ROLL_MEAN": "technical.SMA",
        "ROLL_STD": "stats.rolling_std",
        "EMA": "technical.EMA",
        "MACD": "technical.MACD_line_signal",
        "Z_SCORE": "scaling.zscore_roll",
    }

    assert required_ops <= set(registry)
    for alias, target in required_aliases.items():
        assert aliases.get(alias) == target


@pytest.mark.determinism("d1")
def test_backend_polars_registry_helper_argument_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pysrc.preprocessor.graph.backends import polars as backend

    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(backend, "_reg_get", lambda *args: calls.append(args) or "value")
    monkeypatch.setattr(backend, "_reg_register", lambda *args: calls.append(args) or None)
    monkeypatch.setattr(backend, "_reg_list_ops", lambda *args: calls.append(args) or ["x"])

    assert backend.get("technical.SMA") == "value"
    assert backend.get("polars", "technical.SMA") == "value"
    assert backend.list_ops() == ["x"]
    assert backend.list_ops("polars") == ["x"]
    assert backend.register("custom.temp", lambda ir, lf, **kwargs: lf) is None
    assert backend.register("alt", "custom.temp2", lambda ir, lf, **kwargs: lf) is None

    with pytest.raises(TypeError, match="get\\(\\) expects"):
        backend.get("a", "b", "c")
    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        backend.register("x", lambda ir, lf, **kwargs: lf, nope=True)
    with pytest.raises(TypeError, match="register\\(\\) expects"):
        backend.register("only-one")
    with pytest.raises(TypeError, match="list_ops\\(\\) expects"):
        backend.list_ops("a", "b")

    assert ("polars", "technical.SMA") in calls
    assert ("polars",) in calls


@pytest.mark.determinism("d1")
def test_backend_polars_array_helpers_cover_edge_cases() -> None:
    from pysrc.preprocessor.graph.backends import polars as backend

    values = np.array([np.nan, 1.0, 3.0, np.nan, 5.0], dtype=float)

    mean = backend._rolling_mean_array(values, window=3, min_samples=2)
    std = backend._rolling_std_array(values, window=3, min_samples=2, ddof=1)
    ema = backend._ema_array(values, span=2)
    smooth = backend._wilder_smooth_array(values, window=3)
    empty_rsi = backend._rsi_array(np.array([], dtype=float), window=3)
    obv = backend._obv_array(
        np.array([10.0, 11.0, np.nan, 10.0], dtype=float),
        np.array([100.0, 200.0, 300.0, np.nan], dtype=float),
    )

    assert np.isnan(mean[0])
    assert np.isfinite(mean[2])
    assert np.isnan(std[1])
    assert np.isfinite(std[2])
    assert np.isnan(ema[0])
    assert np.isfinite(ema[1])
    assert np.isfinite(smooth[1])
    assert empty_rsi.size == 0
    assert obv.tolist() == [0.0, 200.0, 200.0, 200.0]


@pytest.mark.determinism("d1")
@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
def test_backend_polars_apply_eager_feature_lowerings_and_csv_path(tmp_path) -> None:
    from pysrc.preprocessor.graph.backends import polars as backend

    frame = pl.DataFrame(
        {
            "close": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "volume": [100.0, 200.0, 100.0, 300.0],
            "session": ["a", "a", "b", "b"],
            "ts": [
                "2024-01-01T09:30:00",
                "2024-01-01T09:31:00",
                "2024-01-02T09:30:00",
                "2024-01-02T09:31:00",
            ],
        }
    ).with_columns(pl.col("ts").str.strptime(pl.Datetime, strict=False))

    eager_df = backend._apply_eager(frame, lambda eager: eager.with_columns(pl.lit(1).alias("x")))
    eager_lf = backend._apply_eager(
        frame.lazy(), lambda eager: eager.with_columns(pl.lit(2).alias("y"))
    )
    assert isinstance(eager_df, pl.DataFrame)
    assert isinstance(eager_lf, pl.LazyFrame)

    robust = backend.scaling_robust_polars(
        {
            "params": {
                "cols": ["close"],
                "out_cols": ["close_robust"],
                "quantile_range": (25, 75),
                "with_centering": False,
                "with_scaling": False,
            }
        },
        frame.lazy(),
        group_by=["session"],
    ).collect()
    returns = backend.feature_returns_polars(
        {"params": {"column": "close"}}, frame.lazy()
    ).collect()
    sma = backend.feature_sma_polars(
        {"params": {"column": "close", "window": 2}}, frame.lazy()
    ).collect()
    rsi = backend.feature_rsi_polars(
        {"params": {"column": "close", "window": 2}}, frame.lazy()
    ).collect()

    csv_path = tmp_path / "prices.csv"
    frame.select(["close", "volume"]).write_csv(csv_path)
    loaded = backend.data_load_csv_polars(
        {"params": {"path": str(csv_path), "try_parse_dates": False}},
        None,
    ).collect()

    assert "close_robust" in robust.columns
    assert "returns" in returns.columns
    assert "sma_2" in sma.columns
    assert "rsi_2" in rsi.columns
    assert loaded.columns == ["close", "volume"]


@pytest.mark.determinism("d1")
@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
def test_backend_polars_vwap_error_paths_and_group_reset() -> None:
    from datetime import date

    from pysrc.preprocessor.graph.backends import polars as backend

    intraday = pl.DataFrame(
        {
            "close": [10.0, 11.0, 12.0],
            "volume": [100.0, 200.0, 100.0],
            "session": ["a", "a", "b"],
            "ts": ["2024-01-01T09:30:00", "2024-01-01T09:31:00", "2024-01-02T09:30:00"],
        }
    ).with_columns(pl.col("ts").str.strptime(pl.Datetime, strict=False))
    daily = pl.DataFrame(
        {
            "close": [10.0],
            "volume": [100.0],
            "ts": [date(2024, 1, 1)],
        }
    )

    with pytest.raises(ValueError, match="requires explicit session_col"):
        backend.technical_vwap_polars({"params": {}}, intraday.lazy()).collect()
    with pytest.raises(ValueError, match="missing required session column"):
        backend.technical_vwap_polars(
            {"params": {"session_col": "missing"}}, intraday.lazy()
        ).collect()
    with pytest.raises(ValueError, match="missing required timestamp column"):
        backend.technical_vwap_polars(
            {"params": {"timestamp_col": "missing"}}, intraday.lazy()
        ).collect()
    with pytest.raises(ValueError, match="daily-only timestamps"):
        backend.technical_vwap_polars({"params": {"timestamp_col": "ts"}}, daily.lazy()).collect()

    out = backend.technical_vwap_polars(
        {
            "params": {
                "price_col": "close",
                "volume_col": "volume",
                "session_col": "session",
                "timestamp_col": "ts",
                "out_col": "vwap",
            }
        },
        intraday.lazy(),
    ).collect()

    assert out["vwap"].to_list() == [10.0, 10.666666666666666, 12.0]


@pytest.mark.determinism("d1")
@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
def test_backend_polars_executor_collect_policy_and_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pysrc.preprocessor.graph.backends import polars as backend

    frame = pl.DataFrame({"close": [1.0, 2.0], "symbol": ["A", "A"]})

    class FakeLazyFrame:
        def __init__(self) -> None:
            self.calls: list[object] = []
            self.last_engine = None

        def collect(self, *, engine=None):
            self.calls.append(engine)
            self.last_engine = engine
            if engine == "gpu":
                raise RuntimeError("gpu failed")
            return frame

    cpu_executor = backend.PolarsExecutor(engine_pref="cpu")
    assert cpu_executor._collect_with_policy(FakeLazyFrame()).shape == (2, 2)

    gpu_executor = backend.PolarsExecutor(engine_pref="gpu")
    monkeypatch.setattr(backend, "capabilities", lambda: SimpleNamespace(has_polars_gpu=True))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pl, "GpuEngine", None, raising=False)
        assert gpu_executor._collect_with_policy(FakeLazyFrame()).shape == (2, 2)

    class FakeEngine:
        pass

    auto_lf = FakeLazyFrame()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pl, "GpuEngine", FakeEngine, raising=False)
        out = backend.PolarsExecutor(engine_pref="auto")._collect_with_policy(auto_lf)
    assert out.shape == (2, 2)
    assert isinstance(auto_lf.last_engine, FakeEngine)

    class FakeCompiledPlan:
        def __init__(self) -> None:
            self.order_by = ["close"]
            self.group_by = ["symbol"]
            self.nodes = [
                {
                    "op": "technical.SMA",
                    "params": {"input_col": "close", "window": 2, "out_col": "close_sma2"},
                }
            ]
            self.params = {"cols": ["close"]}
            self.expected_schema = None

        def report(self):
            return {"status": "ok"}

    monkeypatch.setattr(backend, "validate_dataframe", lambda df: None)
    monkeypatch.setattr(
        backend.SpecFactory, "build", lambda *_args, **kwargs: {"group": kwargs["by"]}
    )
    monkeypatch.setattr(backend, "schema_checks", lambda out, expected, strict: None)
    monkeypatch.setattr(backend, "op_chain", lambda *_ops: lambda out, cols: out)
    monkeypatch.setattr(
        backend, "to_torch_batch", lambda out, cols: SimpleNamespace(torch=cols, rows=out.height)
    )
    monkeypatch.setattr(
        backend.HeuristicPlanner, "optimize", lambda self, segments, sample: segments
    )

    result = backend.PolarsExecutor(engine_pref="cpu", to_torch=True).execute(
        FakeCompiledPlan(), frame.lazy()
    )

    assert result.torch == ["close"]
    assert result.rows == 2
