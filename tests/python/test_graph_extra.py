# tests/python/test_graph_extra.py
from __future__ import annotations

import contextlib
import random
from collections.abc import Iterable
from typing import Any

import pytest

try:
    from pysrc.preprocessor.graph.graph import (
        FusedNode,
        Graph,
        Node,
        SimpleNode,
        deserialize,
        register_node_factory,
        serialize,
    )
    from pysrc.preprocessor.graph.ops import Op, OpKind
except (ImportError, ModuleNotFoundError) as _imp_err:  # pragma: no cover
    pytest.skip(f"graph module not importable: {_imp_err}", allow_module_level=True)


# Constants (optional)


# Helpers (deterministic; typed returns)


class _TestOp(Op):
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
        return None

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
                ir["state"] = dict(self._state)
        return ir


def _kind(label: str) -> Any:
    try:
        return getattr(OpKind, label)
    except Exception:
        return label


def _link(a: Node, b: Node) -> None:
    if b not in a.outputs:
        a.outputs.append(b)
    if a not in b.inputs:
        b.inputs.append(a)


def _names(nodes: Iterable[Node]) -> list[str]:
    return [n.op.name for n in nodes]


def _adj(graph: Graph) -> list[tuple[str, list[str]]]:
    return sorted((n.op.name, sorted(o.op.name for o in n.outputs)) for n in graph.nodes)


# Fixtures (tmp_path resources/setup/teardown)


@pytest.fixture(autouse=True)
def _stable_seed() -> None:
    random.seed(1337)


@pytest.fixture
def sibling_elementwise_graph() -> Graph:
    g = Graph()
    k = _kind("elementwise")

    root = g.add_op(_TestOp("root", k, provides=["r"]))

    z = g.add_op(_TestOp("z_scale", k, provides=["z"], requires=["r"]))
    a = g.add_op(_TestOp("a_fill", k, provides=["a"], requires=["r"]))
    m = g.add_op(_TestOp("m_clip", k, provides=["m"], requires=["r"]))

    _link(root, z)
    _link(root, a)
    _link(root, m)

    return g


@pytest.fixture
def live_and_dead_graph() -> Graph:
    g = Graph()
    k = _kind("elementwise")

    src = g.add_op(_TestOp("src", k, provides=["s"]))
    mid = g.add_op(_TestOp("mid", k, provides=["m"], requires=["s"]))
    sink = g.add_op(_TestOp("sink", k, provides=["t"], requires=["m"]))
    _link(src, mid)
    _link(mid, sink)

    c1 = g.add_op(_TestOp("c1", k, provides=["c1"]))
    c2 = g.add_op(_TestOp("c2", k, provides=["c2"]))
    _link(c1, c2)
    _link(c2, c1)

    return g


@pytest.fixture
def branching_fit_graph() -> Graph:
    g = Graph()
    k = _kind("elementwise")

    a = g.add_op(_TestOp("a", k, provides=["a"]))
    b = g.add_op(_TestOp("b", k, provides=["b"], requires=["a"]))
    c = g.add_op(_TestOp("c", k, provides=["c"], requires=["a"]))

    d = g.add_op(
        _TestOp(
            "d_fit",
            k,
            provides=["d"],
            requires=["b", "c"],
            params={"p": 1},
            fittable=True,
            state={"coef": [42]},
        )
    )

    _link(a, b)
    _link(a, c)
    _link(b, d)
    _link(c, d)

    return g


# === TESTS (gated) ===


# Branch-targeted tests (iff BRANCHES)
class TestOptimizeBehavior:
    def test_reorder_commutative_ops_reorders_siblings_but_not_edges(
        self,
        sibling_elementwise_graph: Graph,
    ):
        # before optimize(): node list preserves insertion order
        g = sibling_elementwise_graph
        before_nodes = _names(g.nodes)
        assert before_nodes == ["root", "z_scale", "a_fill", "m_clip"]
        before_children = _names(g.nodes[0].outputs)
        assert before_children == ["z_scale", "a_fill", "m_clip"]

        # branch: Graph.optimize() triggers _reorder_commutative_ops on parallel elementwise siblings
        g.optimize()

        after_nodes = _names(g.nodes)
        # root should still lead, siblings alphabetized in g.nodes
        assert after_nodes[0] == "root"
        assert after_nodes[1:] == ["a_fill", "m_clip", "z_scale"]

        # outputs list on root is not mutated by reorder
        after_children = _names(g.nodes[0].outputs)
        assert after_children == before_children

        # no loss / dup of nodes
        assert sorted(after_nodes) == sorted(before_nodes)

    @pytest.mark.skipif(
        not hasattr(OpKind, "rolling"),
        reason="OpKind.rolling not available / fusion not supported in this build",
    )
    def test_fuse_compatible_rollings_updates_graph_and_col_providers(self):
        # branch: Graph.optimize() triggers _fuse_compatible_rollings on two-chain rolling ops w/ same window
        g = Graph()
        k = OpKind.rolling

        r1 = g.add_op(_TestOp("r1", k, provides=["r"], params={"window": 5}))
        r2 = g.add_op(_TestOp("r2", k, provides=["m"], requires=["r"], params={"window": 5}))
        _link(r1, r2)

        # precondition
        assert len(g.nodes) == 2
        assert g.col_providers["r"] == [g.nodes[0]]
        assert g.col_providers["m"] == [g.nodes[1]]

        g.optimize()

        # postcondition: fused node replaces both rolling ops
        assert len(g.nodes) == 1
        fused = g.nodes[0]
        assert isinstance(fused, FusedNode)
        assert set(fused.op.provides()) == {"r", "m"}

        # providers remapped to fused node
        assert set(g.col_providers["r"]) == {fused}
        assert set(g.col_providers["m"]) == {fused}

        # in this minimal chain fused should be isolated
        assert fused.inputs == []
        assert fused.outputs == []


class TestPruneDeadIslands:
    def test_prune_drops_unreachable_cycle_and_recomputes_requires(
        self,
        live_and_dead_graph: Graph,
    ):
        # branch: _prune keeps only nodes that reach a sink; drops closed SCC islands
        g = live_and_dead_graph
        pruned = g._prune(g)

        kept_names = set(_names(pruned.nodes))
        assert kept_names == {"src", "mid", "sink"}

        # recomputed input_requires comes only from surviving requires()
        assert pruned.input_requires == {"s", "m"}


# Unit tests (always)
class TestRegisterNodeFactoryIntegration:
    def test_add_op_uses_custom_factory(self):
        # branch: Graph.add_op consults register_node_factory dispatch table
        unique_key = f"custom_kind_{id(object())}"

        class SpecialKind:
            value = unique_key

        class MarkerNode(SimpleNode):
            pass

        seen: dict[str, Op] = {}

        def custom_factory(op: Op):
            seen["op"] = op
            return MarkerNode(op)

        register_node_factory(unique_key, custom_factory)

        g = Graph()
        made = g.add_op(_TestOp("special", SpecialKind, provides=["s"]))

        assert isinstance(made, MarkerNode)
        assert seen["op"].name == "special"


class TestSerializationDeepRoundtrip:
    def test_roundtrip_preserves_state_adjacency_and_requires(
        self,
        branching_fit_graph: Graph,
    ):
        # branch: serialize captures fittable state, components, adjacency; deserialize restores
        g1 = branching_fit_graph

        data = serialize(g1)

        # fitted node IR includes fittable + state
        fit_node_ir = next(nd for nd in data["nodes"] if nd["op"] == "d_fit")
        assert fit_node_ir.get("fittable") is True
        assert fit_node_ir.get("state") == {"coef": [42]}

        # topology metadata should include components == 1 for this connected DAG
        assert data["topology"]["components"] == 1

        adj_before = _adj(g1)
        requires_before = set(g1.input_requires)

        g2 = deserialize(data)

        # adjacency should match by op name
        assert _adj(g2) == adj_before

        # required inputs should survive roundtrip
        assert set(g2.input_requires) == requires_before
