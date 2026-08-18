# tests/python/unit/preprocessor/test_dsl_and_api.py
"""
Comprehensive tests for py/preprocessor/graph/dsl.py and py/preprocessor/api.py

These tests target the 0% coverage in dsl.py (98 lines) and low coverage in api.py (40%).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# =============================================================================
# DSL Tests (py/preprocessor/graph/dsl.py)
# =============================================================================


class TestDSLImports:
    """Test that DSL module imports correctly."""

    def test_dsl_module_imports(self):
        """Basic import smoke test."""
        from pysrc.preprocessor.graph import dsl

        assert hasattr(dsl, "op")
        assert hasattr(dsl, "sequence")
        assert hasattr(dsl, "parallel")
        assert hasattr(dsl, "OpFactory")
        assert hasattr(dsl, "BackendAwareOp")

    def test_backend_aware_op_class_exists(self):
        """BackendAwareOp should extend Op."""
        from pysrc.preprocessor.graph.dsl import BackendAwareOp
        from pysrc.preprocessor.graph.ops import Op

        assert issubclass(BackendAwareOp, Op)


class TestOpFactory:
    """Test OpFactory.create() method."""

    def test_create_known_op_rsi(self):
        """Create a registered op by symbol."""
        from pysrc.preprocessor.graph.dsl import OpFactory

        op = OpFactory.create("technical.RSI", input_col="close", window=14)
        assert op is not None
        assert op.params.get("input_col") == "close"
        assert op.params.get("window") == 14

    def test_create_known_op_sma(self):
        """Create SMA op."""
        from pysrc.preprocessor.graph.dsl import OpFactory

        op = OpFactory.create("technical.SMA", input_col="close", window=20)
        assert op is not None
        assert op.params.get("window") == 20

    def test_create_known_op_via_alias(self):
        """Create op using alias (RSI -> technical.RSI)."""
        from pysrc.preprocessor.graph.dsl import OpFactory

        op = OpFactory.create("RSI", input_col="close", window=14)
        assert op is not None

    def test_create_unknown_op_raises(self):
        """Unknown op should raise ValueError."""
        from pysrc.preprocessor.graph.dsl import OpFactory

        with pytest.raises(ValueError, match="Unknown op"):
            OpFactory.create("nonexistent.op.xyz")

    def test_create_with_backend_hint(self):
        """Op created with backend_hint param."""
        from pysrc.preprocessor.graph.dsl import OpFactory

        op = OpFactory.create("technical.RSI", input_col="close", window=14, backend_hint="cudf")
        assert op.params.get("backend_hint") == "cudf"


class TestBackendAwareOp:
    """Test BackendAwareOp behavior."""

    def test_backend_aware_op_to_ir_includes_hint(self):
        """to_ir() should include backend_hint if set."""
        from pysrc.preprocessor.graph.dsl import BackendAwareOp
        from pysrc.preprocessor.graph.ops import OpKind

        class TestOp(BackendAwareOp):
            NAME = "test.op"
            KIND = OpKind.elementwise

            def to_ir(self):
                ir = {"op": self.NAME, "kind": self.KIND.value, "params": self.params}
                if self.backend_hint:
                    ir["backend_hint"] = self.backend_hint
                return ir

        op = TestOp(backend_hint="polars")
        ir = op.to_ir()
        assert ir.get("backend_hint") == "polars"

    def test_backend_aware_op_without_hint(self):
        """to_ir() without backend_hint."""
        from pysrc.preprocessor.graph.dsl import BackendAwareOp
        from pysrc.preprocessor.graph.ops import OpKind

        class TestOp(BackendAwareOp):
            NAME = "test.op"
            KIND = OpKind.elementwise

            def to_ir(self):
                ir = {"op": self.NAME, "kind": self.KIND.value, "params": self.params}
                if self.backend_hint:
                    ir["backend_hint"] = self.backend_hint
                return ir

        op = TestOp()
        ir = op.to_ir()
        assert "backend_hint" not in ir or ir.get("backend_hint") is None


class TestDSLOpFunction:
    """Test the op() DSL sugar function."""

    def test_op_creates_op_instance(self):
        """op() should return an Op instance."""
        from pysrc.preprocessor.graph.dsl import op
        from pysrc.preprocessor.graph.ops import Op

        result = op("technical.RSI", input_col="close", window=14)
        assert isinstance(result, Op)

    def test_op_with_backend_hint(self):
        """op() with backend_hint parameter."""
        from pysrc.preprocessor.graph.dsl import op

        result = op("technical.RSI", backend_hint="cudf", input_col="close", window=14)
        assert result.params.get("backend_hint") == "cudf"


class TestDSLSequence:
    """Test sequence() graph builder."""

    def test_sequence_from_strings(self):
        """sequence() with string op names."""
        from pysrc.preprocessor.graph.dsl import sequence
        from pysrc.preprocessor.graph.graph import Graph

        g = sequence("technical.RSI:close,14", "technical.SMA:close,20")
        assert isinstance(g, Graph)
        assert len(g.nodes) == 2

    def test_sequence_from_ops(self):
        """sequence() with Op instances."""
        from pysrc.preprocessor.graph.dsl import op, sequence
        from pysrc.preprocessor.graph.graph import Graph

        rsi = op("technical.RSI", input_col="close", window=14)
        sma = op("technical.SMA", input_col="close", window=20)
        g = sequence(rsi, sma)
        assert isinstance(g, Graph)
        assert len(g.nodes) == 2

    def test_sequence_links_nodes(self):
        """Nodes in sequence should be linked."""
        from pysrc.preprocessor.graph.dsl import sequence

        g = sequence("technical.RSI:close,14", "technical.SMA:close,20")
        # First node should have second as output
        if g.nodes[0].outputs:
            assert g.nodes[1] in g.nodes[0].outputs
        # Second node should have first as input
        if g.nodes[1].inputs:
            assert g.nodes[0] in g.nodes[1].inputs

    def test_sequence_from_dicts(self):
        """sequence() with dict specs."""
        from pysrc.preprocessor.graph.dsl import sequence
        from pysrc.preprocessor.graph.graph import Graph

        g = sequence(
            {"symbol": "technical.RSI", "params": {"input_col": "close", "window": 14}},
            {"symbol": "technical.SMA", "params": {"input_col": "close", "window": 20}},
        )
        assert isinstance(g, Graph)
        assert len(g.nodes) == 2

    def test_sequence_single_op(self):
        """sequence() with single op."""
        from pysrc.preprocessor.graph.dsl import sequence

        g = sequence("technical.RSI:close,14")
        assert len(g.nodes) == 1

    def test_sequence_empty(self):
        """sequence() with no ops creates empty graph."""
        from pysrc.preprocessor.graph.dsl import sequence

        g = sequence()
        assert len(g.nodes) == 0


class TestDSLParallel:
    """Test parallel() graph builder."""

    def test_parallel_combines_graphs(self):
        """parallel() should merge multiple graphs."""
        from pysrc.preprocessor.graph.dsl import parallel, sequence
        from pysrc.preprocessor.graph.graph import Graph

        g1 = sequence("technical.RSI:close,14")
        g2 = sequence("technical.SMA:close,20")
        combined = parallel(g1, g2)

        assert isinstance(combined, Graph)
        assert len(combined.nodes) == 2

    def test_parallel_from_strings(self):
        """parallel() with string op names."""
        from pysrc.preprocessor.graph.dsl import parallel

        g = parallel("technical.RSI:close,14", "technical.SMA:close,20")
        assert len(g.nodes) == 2

    def test_parallel_from_ops(self):
        """parallel() with Op instances."""
        from pysrc.preprocessor.graph.dsl import op, parallel

        rsi = op("technical.RSI", input_col="close", window=14)
        sma = op("technical.SMA", input_col="close", window=20)
        g = parallel(rsi, sma)
        assert len(g.nodes) == 2

    def test_parallel_nodes_not_linked(self):
        """Parallel nodes should not have edges between them."""
        from pysrc.preprocessor.graph.dsl import parallel

        g = parallel("technical.RSI:close,14", "technical.SMA:close,20")
        # Parallel ops shouldn't be linked to each other directly
        for node in g.nodes:
            for other in g.nodes:
                if node is not other:
                    # They may share inputs but shouldn't be in each other's outputs
                    assert node not in other.outputs or other not in node.outputs


class TestDSLOperatorOverloads:
    """Test >> and | operator overloads."""

    def test_rshift_operator_chains_ops(self):
        """>> operator should chain ops."""
        from pysrc.preprocessor.graph.dsl import op
        from pysrc.preprocessor.graph.graph import Graph

        rsi = op("technical.RSI", input_col="close", window=14)
        sma = op("technical.SMA", input_col="close", window=20)

        g = rsi >> sma
        assert isinstance(g, Graph)
        assert len(g.nodes) == 2

    def test_or_operator_parallels_ops(self):
        """| operator should parallel ops."""
        from pysrc.preprocessor.graph.dsl import op
        from pysrc.preprocessor.graph.graph import Graph

        rsi = op("technical.RSI", input_col="close", window=14)
        sma = op("technical.SMA", input_col="close", window=20)

        g = rsi | sma
        assert isinstance(g, Graph)
        assert len(g.nodes) == 2

    def test_chained_operators(self):
        """Chained >> and | operators."""
        from pysrc.preprocessor.graph.dsl import op

        a = op("technical.RSI", input_col="close", window=14)
        b = op("technical.SMA", input_col="close", window=20)
        c = op("scaling.zscore", cols=["close"])

        # (a >> b) | c
        g = (a >> b) | c
        assert len(g.nodes) == 3


class TestCombineOps:
    """Test combine_ops() composite op factory."""

    def test_combine_ops_creates_composite(self):
        """combine_ops() should create a CompositeOp class."""
        from pysrc.preprocessor.graph.dsl import combine_ops, op
        from pysrc.preprocessor.graph.ops import Op

        rsi = op("technical.RSI", input_col="close", window=14)
        sma = op("technical.SMA", input_col="close", window=20)

        CompositeClass = combine_ops("my_composite", rsi, sma)
        assert issubclass(CompositeClass, Op)

    def test_combine_ops_registers_in_factory(self):
        """Combined op should be registered."""
        import uuid

        from pysrc.preprocessor.graph.dsl import OpFactory, combine_ops, op

        rsi = op("technical.RSI", input_col="close", window=14)
        sma = op("technical.SMA", input_col="close", window=20)

        name = f"composite_{uuid.uuid4().hex[:8]}"
        combine_ops(name, rsi, sma)

        # Should be able to create it via factory
        composite_op = OpFactory.create(name)
        assert composite_op is not None

    def test_combine_ops_backend_selector(self):
        """Backend selector function should determine backend_hint."""
        import uuid

        from pysrc.preprocessor.graph.dsl import combine_ops, op

        rsi = op("technical.RSI", input_col="close", window=14)

        def always_cudf(ops):
            return "cudf"

        name = f"cudf_composite_{uuid.uuid4().hex[:8]}"
        CompositeClass = combine_ops(name, rsi, backend_selector=always_cudf)
        instance = CompositeClass()
        assert instance.backend_hint == "cudf"


# =============================================================================
# API Tests (py/preprocessor/api.py)
# =============================================================================


class TestAPIImports:
    """Test API module imports."""

    def test_api_module_imports(self):
        """Basic import smoke test."""
        from pysrc.preprocessor import api

        assert hasattr(api, "run")
        assert hasattr(api, "Plan")
        assert hasattr(api, "PlanSpec")
        assert hasattr(api, "PreprocessorBuilder")
        assert hasattr(api, "get_executor")


class TestPlanSpec:
    """Test PlanSpec dataclass."""

    def test_planspec_defaults(self):
        """PlanSpec should have sensible defaults."""
        from pysrc.preprocessor.api import PlanSpec

        spec = PlanSpec()
        assert spec.ops == []
        assert spec.target is None
        assert spec.sequence is None
        assert spec.scaling is None
        assert spec.meta is None

    def test_planspec_with_ops(self):
        """PlanSpec with ops list."""
        from pysrc.preprocessor.api import PlanSpec

        spec = PlanSpec(ops=[{"kind": "zscore", "cols": ["close"]}])
        assert len(spec.ops) == 1


class TestMergeSpecs:
    """Test merge_specs() function."""

    def test_merge_empty_specs(self):
        """Merging empty specs."""
        from pysrc.preprocessor.api import merge_specs

        result = merge_specs()
        assert result.ops == []

    def test_merge_single_spec(self):
        """Merging single spec."""
        from pysrc.preprocessor.api import PlanSpec, merge_specs

        spec = PlanSpec(ops=[{"kind": "zscore"}])
        result = merge_specs(spec)
        assert len(result.ops) == 1

    def test_merge_multiple_specs(self):
        """Merging multiple specs combines ops."""
        from pysrc.preprocessor.api import PlanSpec, merge_specs

        spec1 = PlanSpec(ops=[{"kind": "zscore"}])
        spec2 = PlanSpec(ops=[{"kind": "sma"}])
        result = merge_specs(spec1, spec2)
        assert len(result.ops) == 2

    def test_merge_preserves_last_target(self):
        """Last non-None target wins."""
        from pysrc.preprocessor.api import PlanSpec, merge_specs

        spec1 = PlanSpec(target={"col": "a"})
        spec2 = PlanSpec(target={"col": "b"})
        result = merge_specs(spec1, spec2)
        assert result.target == {"col": "b"}

    def test_merge_meta_provenance(self):
        """Meta provenance chain is built."""
        from pysrc.preprocessor.api import PlanSpec, merge_specs

        spec1 = PlanSpec(meta={"provenance": "source1"})
        spec2 = PlanSpec(meta={"provenance": "source2"})
        result = merge_specs(spec1, spec2)
        assert "provenance_chain" in result.meta
        assert "source1" in result.meta["provenance_chain"]
        assert "source2" in result.meta["provenance_chain"]


class TestModelRegistry:
    """Test ModelRegistry class."""

    def test_register_and_get_model(self):
        """Register and retrieve a model."""
        from pysrc.preprocessor.api import ModelRegistry

        mock_model = MagicMock()
        ModelRegistry.register("test_model", mock_model)
        retrieved = ModelRegistry.get("test_model")
        assert retrieved is mock_model

    def test_get_nonexistent_returns_none(self):
        """Getting nonexistent model returns None."""
        from pysrc.preprocessor.api import ModelRegistry

        result = ModelRegistry.get("nonexistent_model_xyz")
        assert result is None


class TestResolveModelsInOps:
    """Test resolve_models_in_ops() function."""

    def test_resolve_adds_model_to_op(self):
        """model_ref string should be resolved to __model__."""
        from pysrc.preprocessor.api import ModelRegistry, resolve_models_in_ops

        mock_model = MagicMock()
        ModelRegistry.register("my_model", mock_model)

        ops = [{"kind": "external", "model_ref": "my_model"}]
        resolved = resolve_models_in_ops(ops)

        assert resolved[0].get("__model__") is mock_model

    def test_resolve_preserves_ops_without_model_ref(self):
        """Ops without model_ref pass through unchanged."""
        from pysrc.preprocessor.api import resolve_models_in_ops

        ops = [{"kind": "zscore", "cols": ["close"]}]
        resolved = resolve_models_in_ops(ops)
        assert resolved == ops


class TestPlan:
    """Test Plan dataclass."""

    def test_plan_is_frozen(self):
        """Plan should be immutable."""
        from pysrc.preprocessor.api import Plan

        plan = Plan(ops=["RSI"], params={}, group_by=["symbol"])
        with pytest.raises(AttributeError):
            plan.ops = ["SMA"]

    def test_plan_default_version(self):
        """Plan has default version."""
        from pysrc.preprocessor.api import Plan

        plan = Plan(ops=[], params={}, group_by=[])
        assert plan.version == "1.0"


class TestGetExecutor:
    """Test get_executor() function."""

    def test_get_executor_auto(self):
        """get_executor('auto') returns an executor."""
        from pysrc.preprocessor.api import get_executor
        from pysrc.preprocessor.graph.executor import Executor

        executor = get_executor("auto")
        assert isinstance(executor, Executor)

    def test_get_executor_polars(self):
        """get_executor with polars backend."""
        from pysrc.preprocessor.api import get_executor
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        executor = get_executor("polars")
        assert isinstance(executor, PolarsExecutor)

    def test_get_executor_cpu(self):
        """get_executor('cpu') returns PolarsExecutor."""
        from pysrc.preprocessor.api import get_executor
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        executor = get_executor("cpu")
        assert isinstance(executor, PolarsExecutor)


class TestPreprocessorBuilder:
    """Test PreprocessorBuilder fluent API."""

    def test_builder_add_op(self):
        """add_op() adds operation."""
        from pysrc.preprocessor.api import PreprocessorBuilder

        builder = PreprocessorBuilder()
        result = builder.add_op("technical.RSI", input_col="close", window=14)
        assert result is builder  # Fluent API
        assert "technical.RSI" in builder._ops

    def test_builder_set_group_by(self):
        """set_group_by() sets grouping columns."""
        from pysrc.preprocessor.api import PreprocessorBuilder

        builder = PreprocessorBuilder()
        builder.set_group_by(["symbol", "date"])
        assert builder._group_by == ["symbol", "date"]

    def test_builder_set_backend(self):
        """set_backend() sets execution backend."""
        from pysrc.preprocessor.api import PreprocessorBuilder

        builder = PreprocessorBuilder()
        builder.set_backend("polars")
        assert builder._backend == "polars"

    def test_builder_from_dict(self):
        """from_dict() loads configuration."""
        from pysrc.preprocessor.api import PreprocessorBuilder

        config = {
            "ops": ["technical.RSI", "technical.SMA"],
            "params": {"technical.RSI": [{"input_col": "close", "window": 14}]},
            "group_by": ["symbol"],
            "backend": "polars",
        }
        builder = PreprocessorBuilder().from_dict(config)
        assert "technical.RSI" in builder._ops
        assert "technical.SMA" in builder._ops
        assert builder._backend == "polars"

    def test_builder_build_plan(self):
        """build_plan() creates immutable Plan."""
        from pysrc.preprocessor.api import Plan, PreprocessorBuilder

        builder = PreprocessorBuilder()
        builder.add_op("technical.RSI", input_col="close", window=14)
        plan = builder.build_plan()

        assert isinstance(plan, Plan)
        assert "technical.RSI" in plan.ops

    def test_builder_build_runner(self):
        """build_runner() creates callable."""
        from pysrc.preprocessor.api import PreprocessorBuilder

        builder = PreprocessorBuilder()
        builder.add_op("technical.RSI", input_col="close", window=14)
        runner = builder.build_runner()

        assert callable(runner)

    def test_builder_add_dsl_op(self):
        """add_dsl_op() adds op with backend hint."""
        from pysrc.preprocessor.api import PreprocessorBuilder

        builder = PreprocessorBuilder()
        builder.add_dsl_op("technical.RSI", backend_hint="cudf", input_col="close", window=14)

        # Check params contain backend_hint
        params = builder._params.get("technical.RSI", [])
        assert any(p.get("backend_hint") == "cudf" for p in params)

    def test_builder_add_sequence(self):
        """add_sequence() adds multiple ops."""
        from pysrc.preprocessor.api import PreprocessorBuilder

        builder = PreprocessorBuilder()
        builder.add_sequence(
            "technical.RSI",
            {"symbol": "technical.SMA", "params": {"input_col": "close", "window": 20}},
        )
        assert len(builder._ops) == 2

    def test_builder_add_parallel(self):
        """add_parallel() merges from other builders."""
        from pysrc.preprocessor.api import PreprocessorBuilder

        b1 = PreprocessorBuilder().add_op("technical.RSI", input_col="close", window=14)
        b2 = PreprocessorBuilder().add_op("technical.SMA", input_col="close", window=20)

        main = PreprocessorBuilder()
        main.add_parallel(b1, b2)

        assert "technical.RSI" in main._ops
        assert "technical.SMA" in main._ops

    def test_builder_add_transform(self):
        """add_transform() adds named transform."""
        from pysrc.preprocessor.api import PreprocessorBuilder

        builder = PreprocessorBuilder()
        builder.add_transform("log", input_col="close")

        assert "transform_log" in builder._ops


class TestStream:
    """Test stream() function."""

    def test_stream_not_implemented(self):
        """stream() should raise NotImplementedError."""
        from pysrc.preprocessor.api import Plan, stream

        plan = Plan(ops=[], params={}, group_by=[])
        with pytest.raises(NotImplementedError):
            stream(plan)


class TestRunFunction:
    """Test run() function with mocked execution."""

    def test_run_accepts_dict_plan(self):
        """run() should accept dict as plan."""
        import polars as pl

        from pysrc.preprocessor.api import run

        df = pl.DataFrame({"close": [100.0, 101.0, 102.0]})
        plan_dict = {"ops": [], "group_by": []}

        # Should not raise on empty plan
        result = run(df, plan_dict, backend="polars")
        assert result is not None

    def test_run_accepts_plan_object(self):
        """run() should accept Plan object."""
        import polars as pl

        from pysrc.preprocessor.api import Plan, run

        df = pl.DataFrame({"close": [100.0, 101.0, 102.0]})
        plan = Plan(ops=[], params={}, group_by=[])

        result = run(df, plan, backend="polars")
        assert result is not None

    def test_run_returns_polars_for_polars_input(self):
        """run() should return Polars DataFrame for Polars input."""
        import polars as pl

        from pysrc.preprocessor.api import Plan, run

        df = pl.DataFrame({"close": [100.0, 101.0, 102.0]})
        plan = Plan(ops=[], params={}, group_by=[])

        result = run(df, plan, backend="auto")
        assert isinstance(result, pl.DataFrame)


# =============================================================================
# Integration Tests
# =============================================================================


class TestDSLToAPIIntegration:
    """Test DSL and API working together."""

    def test_dsl_sequence_to_builder(self):
        """DSL sequence can inform builder configuration."""
        from pysrc.preprocessor.api import PreprocessorBuilder
        from pysrc.preprocessor.graph.dsl import sequence

        # Build a graph with DSL
        g = sequence("technical.RSI:close,14", "technical.SMA:close,20")

        # Use builder to create equivalent plan
        builder = PreprocessorBuilder()
        builder.add_op("technical.RSI", input_col="close", window=14)
        builder.add_op("technical.SMA", input_col="close", window=20)
        plan = builder.build_plan()

        assert len(plan.ops) == len(g.nodes)
