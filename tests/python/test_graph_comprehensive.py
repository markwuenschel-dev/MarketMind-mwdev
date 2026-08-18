# tests/unit/test_graph_comprehensive.py
from __future__ import annotations

import contextlib
import random
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from tests.python.infra.matrix import matrix

try:
    from pysrc.preprocessor.graph.graph import (
        FusedNode,
        Graph,
        Node,
        SimpleNode,
        _cols,
        deserialize,
        register_node_factory,
        serialize,
    )
    from pysrc.preprocessor.graph.ops import Op, OpKind
except (ImportError, ModuleNotFoundError) as _imp_err:
    pytest.skip(f"graph module not importable: {_imp_err}", allow_module_level=True)

try:
    from pysrc.core.errors import UnsupportedPlan
except (ImportError, ModuleNotFoundError):

    class UnsupportedPlan(Exception):  # type: ignore[no-redef]
        pass


# === Helpers ===


class _TestOp(Op):
    # Minimal Op implementation for testing.
    NAME: str
    KIND: Any

    def __init__(
        self,
        name: str,
        kind: Any,
        provides: Iterable[str] = (),
        requires: Iterable[str] = (),
        params: dict[str, Any] | None = None,
        fittable: bool = False,
        state: dict[str, Any] | None = None,
    ):
        with contextlib.suppress(TypeError):
            super().__init__()
        self.NAME = name
        self.KIND = kind if hasattr(kind, "value") else type("K", (), {"value": str(kind)})()
        self.params = dict(params or {})
        self._provides = set(provides)
        self._requires = set(requires)
        self._fittable = fittable
        self._state = dict(state or {})

    @property
    def name(self) -> str:
        return self.NAME

    def validate_params(self) -> None:
        pass

    def provides(self) -> set[str]:
        return set(self._provides)

    def requires(self) -> set[str]:
        return set(self._requires)

    def is_fittable(self) -> bool:
        return self._fittable

    def state_dict(self) -> dict[str, Any]:
        return dict(self._state)

    def to_ir(self) -> dict[str, Any]:
        ir = {
            "op": self.NAME,
            "kind": getattr(self.KIND, "value", str(self.KIND)),
            "params": dict(self.params),
            "requires": sorted(self._requires),
            "provides": sorted(self._provides),
        }
        if self._fittable:
            ir["fittable"] = True
            if self._state:
                ir["state"] = self._state
        return ir


def _kind(label: str) -> Any:
    # Get OpKind or fallback to string.
    if OpKind is None:
        return label
    return getattr(OpKind, label, label)


def _link(a: Node, b: Node) -> None:
    # Create edge between nodes.
    if b not in a.outputs:
        a.outputs.append(b)
    if a not in b.inputs:
        b.inputs.append(a)


# === Fixtures ===


@pytest.fixture(autouse=True)
def _deterministic_graph_tests():
    # Ensure deterministic behavior for graph tests.
    np.random.seed(42)
    random.seed(42)
    return


@pytest.fixture
def empty_graph() -> Graph:
    return Graph()


@pytest.fixture
def single_node_graph() -> Graph:
    g = Graph()
    g.add_op(_TestOp("single", _kind("elementwise"), provides=["x"]))
    return g


@pytest.fixture
def chain_graph() -> Graph:
    g = Graph()
    nodes = [g.add_op(_TestOp(f"n{i}", _kind("elementwise"), provides=[f"c{i}"])) for i in range(4)]
    for i in range(3):
        _link(nodes[i], nodes[i + 1])
    return g


@pytest.fixture
def diamond_graph() -> Graph:
    g = Graph()
    a = g.add_op(_TestOp("a", _kind("elementwise"), provides=["a"]))
    b = g.add_op(_TestOp("b", _kind("elementwise"), provides=["b"]))
    c = g.add_op(_TestOp("c", _kind("elementwise"), provides=["c"]))
    d = g.add_op(_TestOp("d", _kind("elementwise"), provides=["d"]))
    _link(a, b)
    _link(a, c)
    _link(b, d)
    _link(c, d)
    return g


@pytest.fixture
def two_component_graph() -> Graph:
    g = Graph()
    # Component 1
    a = g.add_op(_TestOp("a", _kind("elementwise"), provides=["a"]))
    b = g.add_op(_TestOp("b", _kind("elementwise"), provides=["b"]))
    _link(a, b)

    # Component 2
    c = g.add_op(_TestOp("c", _kind("elementwise"), provides=["c"]))
    d = g.add_op(_TestOp("d", _kind("elementwise"), provides=["d"]))
    _link(c, d)

    return g


# === TESTS ===

# --- Unit tests ---


@matrix(
    input_type=[
        "none",
        "callable_none",
        "callable_set",
        "callable_list",
        "callable_tuple",
        "callable_string",
        "set",
        "list",
        "tuple",
        "string",
    ],
)
class TestColsHelper:
    # Test _cols() normalization function with various input types.

    def test_cols_input_types(self):
        inputs = {
            "none": (None, []),
            "callable_none": (lambda: None, []),
            "callable_set": (lambda: {"a", "b", "c"}, ["a", "b", "c"]),
            "callable_list": (lambda: ["x", "y"], ["x", "y"]),
            "callable_tuple": (lambda: ("p", "q"), ["p", "q"]),
            "callable_string": (lambda: "single", ["single"]),
            "set": ({"a", "b"}, ["a", "b"]),
            "list": (["x", "y", "z"], ["x", "y", "z"]),
            "tuple": (("m", "n"), ["m", "n"]),
            "string": ("column_name", ["column_name"]),
        }

        # Iterate through every scenario instead of relying on fragile @matrix
        # fixture injection. We assert semantic invariants of _cols(), not the
        # exact ordering for inherently unordered inputs.
        for input_val, expected in inputs.values():
            result = _cols(input_val)

            # _cols() contract: always returns a list of column labels
            assert isinstance(result, list)

            # Some inputs are sets (unordered) or callables returning sets.
            unordered = False
            if isinstance(input_val, set):
                unordered = True
                runtime_val = input_val
            elif callable(input_val):
                runtime_val = input_val()
                if isinstance(runtime_val, set):
                    unordered = True
            else:
                runtime_val = input_val

            if unordered:
                # Compare as sets to allow any ordering.
                assert set(result) == set(expected)
            else:
                assert result == expected


class TestNodeClasses:
    # Test Node, SimpleNode, and FusedNode behavior.

    def test_node_hash_based_on_identity(self):
        op1 = _TestOp("op1", _kind("elementwise"), provides=["a"])
        op2 = _TestOp("op2", _kind("elementwise"), provides=["b"])

        node1 = SimpleNode(op1)
        node2 = SimpleNode(op2)
        node3 = node1

        assert hash(node1) == hash(node3)
        assert hash(node1) != hash(node2)
        assert node1 == node3
        assert node1 != node2

    def test_fused_node_synthesized_op_aggregates_requires_and_provides(self):
        op1 = _TestOp("op1", _kind("rolling"), provides=["a", "b"], requires=["x", "y"])
        op2 = _TestOp("op2", _kind("rolling"), provides=["b", "c"], requires=["y", "z"])

        fused = FusedNode([op1, op2], _kind("rolling"))

        assert fused.op.requires() == {"x", "y", "z"}
        assert fused.op.provides() == {"a", "b", "c"}

    def test_fused_node_to_ir_contains_sub_irs(self):
        op1 = _TestOp("roll1", _kind("rolling"), provides=["a"], requires=["x"])
        op2 = _TestOp("roll2", _kind("rolling"), provides=["b"], requires=["y"])

        fused = FusedNode([op1, op2], _kind("rolling"))
        ir = fused.to_ir()

        assert "sub_irs" in ir
        assert len(ir["sub_irs"]) == 2


class TestHasCycle:
    # Test has_cycle() detection using fixtures.

    def test_has_cycle_empty_graph(self, empty_graph):
        assert empty_graph.has_cycle() is False

    def test_has_cycle_single_node(self, single_node_graph):
        assert single_node_graph.has_cycle() is False

    def test_has_cycle_chain(self, chain_graph):
        assert chain_graph.has_cycle() is False

    def test_has_cycle_diamond(self, diamond_graph):
        assert diamond_graph.has_cycle() is False

    @matrix(
        cycle_type=["simple_cycle", "self_loop", "complex_cycle"],
    )
    def test_has_cycle_returns_true_for_cycles(self, cycle_type):
        g = Graph()

        # Unpack tuple from @matrix decorator (fragile fixture injection)
        cycle_name = cycle_type[0] if isinstance(cycle_type, (tuple, list)) else cycle_type

        if cycle_name == "simple_cycle":
            a = g.add_op(_TestOp("a", _kind("elementwise"), provides=["a"]))
            b = g.add_op(_TestOp("b", _kind("elementwise"), provides=["b"]))
            _link(a, b)
            _link(b, a)
        elif cycle_name == "self_loop":
            a = g.add_op(_TestOp("a", _kind("elementwise"), provides=["a"]))
            _link(a, a)
        elif cycle_name == "complex_cycle":
            nodes = [
                g.add_op(_TestOp(f"n{i}", _kind("elementwise"), provides=[f"c{i}"]))
                for i in range(4)
            ]
            _link(nodes[0], nodes[1])
            _link(nodes[1], nodes[2])
            _link(nodes[2], nodes[3])
            _link(nodes[3], nodes[1])  # Back edge

        assert g.has_cycle() is True


class TestShortestPath:
    # Test shortest_path() algorithm.

    def test_shortest_path_same_node(self, single_node_graph):
        node = single_node_graph.nodes[0]
        path = single_node_graph.shortest_path(node, node)
        assert path == [node]

    def test_shortest_path_chain(self, chain_graph):
        nodes = chain_graph.nodes
        path = chain_graph.shortest_path(nodes[0], nodes[3])
        assert len(path) == 4
        assert path[0] == nodes[0]
        assert path[-1] == nodes[3]

    def test_shortest_path_diamond_finds_shortest(self, diamond_graph):
        nodes = diamond_graph.nodes
        a = next(n for n in nodes if n.op.name == "a")
        d = next(n for n in nodes if n.op.name == "d")

        path = diamond_graph.shortest_path(a, d)
        assert len(path) == 3  # a -> (b or c) -> d


class TestConnectedComponents:
    # Test connected_components() algorithm.

    def test_connected_components_empty(self, empty_graph):
        assert empty_graph.connected_components() == []

    def test_connected_components_single_node(self, single_node_graph):
        comps = single_node_graph.connected_components()
        assert len(comps) == 1
        assert comps[0] == set(single_node_graph.nodes)

    def test_connected_components_single_component(self, chain_graph):
        comps = chain_graph.connected_components()
        assert len(comps) == 1
        assert comps[0] == set(chain_graph.nodes)

    def test_connected_components_two_components(self, two_component_graph):
        comps = two_component_graph.connected_components()
        assert len(comps) == 2

        # Each component should have 2 nodes
        assert all(len(comp) == 2 for comp in comps)


class TestSerialization:
    # Test serialize() and deserialize() functions.

    def test_serialize_empty_graph(self, empty_graph):
        data = serialize(empty_graph)

        assert data["version"] == "1.0"
        assert data["nodes"] == []
        assert data["input_requires"] == []
        assert data["topology"]["node_count"] == 0

    def test_serialize_single_node(self, single_node_graph):
        data = serialize(single_node_graph)

        assert len(data["nodes"]) == 1
        assert data["topology"]["node_count"] == 1

    @matrix(
        graph_fixture=["chain_graph", "diamond_graph", "two_component_graph"],
    )
    def test_serialize_various_structures(self, graph_fixture, request):
        name = (
            graph_fixture[0]
            if isinstance(graph_fixture, (list, tuple)) and len(graph_fixture) == 1
            else graph_fixture
        )
        g = request.getfixturevalue(name)
        data = serialize(g)

        assert "version" in data
        assert "nodes" in data
        assert len(data["nodes"]) == len(g.nodes)
        assert data["topology"]["node_count"] == len(g.nodes)

    @matrix(
        graph_fixture=["chain_graph", "diamond_graph"],
    )
    def test_serialize_deserialize_roundtrip(self, graph_fixture, request):
        name = (
            graph_fixture[0]
            if isinstance(graph_fixture, (list, tuple)) and len(graph_fixture) == 1
            else graph_fixture
        )
        g1 = request.getfixturevalue(name)

        data = serialize(g1)
        g2 = deserialize(data)

        assert len(g2.nodes) == len(g1.nodes)
        order1 = g1.topological_sort()
        order2 = g2.topological_sort()
        assert len(order1) == len(order2)


# --- Pairwise tests ---


class TestNodePairwise:
    @matrix(
        has_fittable=[False, True],
        has_state=[False, True],
    )
    def test_simple_node_to_ir_variations(self, has_fittable: bool, has_state: bool):
        state = {"mean": 10.5, "std": 2.3} if has_state else {}
        op = _TestOp(
            "test_op",
            _kind("transform"),
            provides=["out1", "out2"],
            requires=["in1"],
            params={"param1": "value1"},
            fittable=has_fittable,
            state=state,
        )
        node = SimpleNode(op)

        ir = node.to_ir()

        assert ir["op"] == "test_op"
        assert ir["params"] == {"param1": "value1"}

        if has_fittable:
            assert ir["fittable"] is True
            if has_state:
                assert ir["state"] == state
            else:
                assert "state" not in ir
        else:
            assert "fittable" not in ir


# --- Property tests ---


class TestGraphProperties:
    # Test graph invariants using fixtures.

    @matrix(
        graph_fixture=["chain_graph", "diamond_graph", "two_component_graph"],
    )
    def test_edge_count_invariant(self, graph_fixture, request):
        # Some parametrization providers (like @matrix) hand us a 1-tuple
        # e.g. ('chain_graph',) instead of the bare fixture name. pytest then
        # treats that tuple as a phantom fixture and collection fails.
        fixture_name = (
            graph_fixture[0]
            if isinstance(graph_fixture, (list, tuple)) and len(graph_fixture) == 1
            else graph_fixture
        )
        g = request.getfixturevalue(fixture_name)

        in_edges = sum(len(n.inputs) for n in g.nodes)
        out_edges = sum(len(n.outputs) for n in g.nodes)

        # For any directed graph, total in-edges must equal total out-edges.
        assert in_edges == out_edges

    def test_topological_order_respects_edges(self, diamond_graph):
        order = diamond_graph.topological_sort()
        pos = {order[i]: i for i in range(len(order))}

        for node in diamond_graph.nodes:
            for output in node.outputs:
                assert pos[node] < pos[output]

    def test_after_merge_components_combine(self):
        g1 = Graph()
        a = g1.add_op(_TestOp("a", _kind("elementwise"), provides=["a"]))
        b = g1.add_op(_TestOp("b", _kind("elementwise"), provides=["b"]))
        _link(a, b)

        g2 = Graph()
        c = g2.add_op(_TestOp("c", _kind("elementwise"), provides=["c"]))
        d = g2.add_op(_TestOp("d", _kind("elementwise"), provides=["d"]))
        _link(c, d)

        comps1 = g1.connected_components()
        comps2 = g2.connected_components()

        g1.merge(g2)
        comps_merged = g1.connected_components()

        assert len(comps_merged) == len(comps1) + len(comps2)


# --- Perf tests ---


class TestPerformance:
    # Test performance characteristics.

    @pytest.mark.perf
    @pytest.mark.skip(reason="Performance tests skipped by default")
    def test_topological_sort_scales(self, benchmark):
        g = Graph()
        n = 500
        nodes = [
            g.add_op(_TestOp(f"n{i}", _kind("elementwise"), provides=[f"c{i}"])) for i in range(n)
        ]

        for i in range(n - 1):
            _link(nodes[i], nodes[i + 1])

        result = benchmark(g.topological_sort)
        assert len(result) == n

    @pytest.mark.perf
    @pytest.mark.skip(reason="Performance tests skipped by default")
    def test_connected_components_scales(self, benchmark):
        g = Graph()
        n = 300
        nodes = [
            g.add_op(_TestOp(f"n{i}", _kind("elementwise"), provides=[f"c{i}"])) for i in range(n)
        ]

        # Create multiple small components
        for i in range(0, n - 1, 10):
            for j in range(i, min(i + 9, n - 1)):
                _link(nodes[j], nodes[j + 1])

        result = benchmark(g.connected_components)
        assert len(result) > 1


# --- Error tests ---


class TestErrorConditions:
    # Test error handling and validation.

    def test_fused_node_with_empty_sub_ops_raises_error(self):
        with pytest.raises(ValueError, match="at least one"):
            FusedNode([], _kind("rolling"))

    def test_fused_node_with_non_op_types_raises_error(self):
        op1 = _TestOp("op1", _kind("rolling"), provides=["a"])
        with pytest.raises(TypeError, match="All sub_ops must be Op instances"):
            FusedNode([op1, "not_an_op", 123], _kind("rolling"))

    @matrix(
        error_case=["start_not_in_graph", "end_not_in_graph", "no_path_exists"],
    )
    def test_shortest_path_error_cases(self, error_case: str, chain_graph):
        external = SimpleNode(_TestOp("external", _kind("elementwise"), provides=["x"]))
        nodes = chain_graph.nodes

        if error_case == "start_not_in_graph":
            with pytest.raises(ValueError, match="not in graph"):
                chain_graph.shortest_path(external, nodes[0])
        elif error_case == "end_not_in_graph":
            with pytest.raises(ValueError, match="not in graph"):
                chain_graph.shortest_path(nodes[0], external)
        elif error_case == "no_path_exists":
            # Use two_component graph instead
            g = Graph()
            a = g.add_op(_TestOp("a", _kind("elementwise"), provides=["a"]))
            b = g.add_op(_TestOp("b", _kind("elementwise"), provides=["b"]))
            # No connection between a and b
            with pytest.raises(ValueError, match="No path"):
                g.shortest_path(a, b)

    @matrix(
        version=["2.0", "0.5", None],
    )
    def test_deserialize_invalid_version_raises_error(self, version):
        data = {"version": version, "nodes": []}

        with pytest.raises(ValueError, match="Unsupported serialization version"):
            deserialize(data)

    @matrix(
        error_type=["non_string_key", "non_callable_factory"],
    )
    def test_register_node_factory_type_errors(self, error_type: str):
        if error_type == "non_string_key":
            with pytest.raises(TypeError, match="Factory key must be str"):
                register_node_factory(123, lambda op: SimpleNode(op))
        elif error_type == "non_callable_factory":
            with pytest.raises(TypeError, match="Factory must be callable"):
                register_node_factory("test_factory", "not_callable")

    def test_register_node_factory_duplicate_key_raises_error(self):
        key = f"unique_test_{id(self)}_{random.randint(0, 1000000)}"

        def factory(op):
            return SimpleNode(op)

        register_node_factory(key, factory)

        with pytest.raises(ValueError, match="already registered"):
            register_node_factory(key, factory)
