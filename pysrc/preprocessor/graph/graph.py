# py/preprocessor/graph/graph.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from pysrc.core.errors import UnsupportedPlan
from pysrc.preprocessor.graph.ops import Op, OpKind
from pysrc.preprocessor.utils.nvtx import nvtx_range
from pysrc.preprocessor.utils.validate import plan_checks


def _cols(x) -> list[str]:
    # normalize requires/provides that might be a method, property, set, list, etc.
    v = x() if callable(x) else x
    if v is None:
        return []
    if isinstance(v, (set, list, tuple)):
        return list(v)
    return [v]


class Node(ABC):
    def __init__(self, op: Op):
        self.op: Op = op
        self.inputs: list[Node] = []
        self.outputs: list[Node] = []

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return self is other

    @abstractmethod
    def to_ir(self) -> dict[str, Any]:
        pass


class SimpleNode(Node):
    def to_ir(self) -> dict[str, Any]:
        ir = {
            "op": self.op.name,
            "kind": self.op.KIND.value,
            "params": dict(self.op.params),
            "requires": _cols(self.op.requires),
            "provides": _cols(self.op.provides),
        }
        if self.op.is_fittable():
            ir["fittable"] = True
            state = self.op.state_dict()
            if state:
                ir["state"] = state
        return ir


class FusedNode(Node):
    def __init__(self, sub_ops: list[Op], fused_kind: OpKind):
        if not sub_ops:
            raise ValueError("FusedNode requires at least one sub-op")
        if not all(isinstance(op, Op) for op in sub_ops):
            raise TypeError(f"All sub_ops must be Op instances, got {[type(op) for op in sub_ops]}")
        fused_op = self._synthesize_fused_op(sub_ops, fused_kind)
        super().__init__(fused_op)
        self.sub_ops: list[Op] = sub_ops

    def _synthesize_fused_op(self, sub_ops: list[Op], fused_kind: OpKind) -> Op:
        class FusedOp(Op):
            NAME = f"fused.{fused_kind.value}"
            KIND = fused_kind

            def __init__(self, sub_params: list[dict[str, Any]]):
                self.params = {"sub_ops": sub_params}
                self.validate_params()

            def requires(self) -> set[str]:
                reqs = set()
                for sub in sub_ops:
                    reqs.update(sub.requires())
                return reqs

            def provides(self) -> set[str]:
                provs = set()
                for sub in sub_ops:
                    provs.update(sub.provides())
                return provs

            def to_ir(self) -> dict[str, Any]:
                return {
                    "op": self.NAME,
                    "kind": self.KIND.value,
                    "params": self.params,
                    "requires": list(self.requires()),
                    "provides": list(self.provides()),
                    "sub_irs": [sub.to_ir() for sub in sub_ops],
                }

        return FusedOp([sub.params for sub in sub_ops])

    def to_ir(self) -> dict[str, Any]:
        return self.op.to_ir()


_NODE_FACTORIES: dict[str, Callable[[Op], Node]] = {
    "default": SimpleNode,
    "fused": FusedNode,
}


def register_node_factory(key: str, factory: Callable[[Op], Node]) -> None:
    if not isinstance(key, str):
        raise TypeError(f"Factory key must be str, got {type(key).__name__}")
    if not callable(factory):
        raise TypeError(f"Factory must be callable, got {type(factory).__name__}")
    if key in _NODE_FACTORIES:
        raise ValueError(f"Node factory '{key}' already registered")
    _NODE_FACTORIES[key] = factory


class Graph:
    def __init__(self):
        self.nodes: list[Node] = []
        self.col_providers: dict[str, list[Node]] = defaultdict(list)
        self.input_requires: set[str] = set()

    def add_op(self, op: Op) -> Node:
        factory = _NODE_FACTORIES.get(op.KIND.value, _NODE_FACTORIES["default"])
        node = factory(op)
        self.nodes.append(node)
        for col in _cols(op.provides):
            self.col_providers[col].append(node)
        self.input_requires.update(_cols(op.requires))

        return node

    def merge(self, other: Graph) -> None:
        self.nodes.extend(other.nodes)
        for col, providers in other.col_providers.items():
            self.col_providers[col].extend(providers)
        self.input_requires.update(other.input_requires)

    @nvtx_range("optimize_graph")
    def optimize(self) -> None:
        plan_checks(self._to_dict())  # Validate before/after
        self._apply_pass(self._fuse_compatible_rollings)
        self._apply_pass(self._reorder_commutative_ops)
        seen = {}
        for n in self.nodes:
            for c in _cols(n.op.provides):
                if c in seen:
                    raise UnsupportedPlan(f"duplicate column {c} after fusion")
                seen[c] = 1

    def _apply_pass(self, pass_fn: Callable[[], None]) -> None:
        pass_fn()

    def _fuse_compatible_rollings(self) -> None:
        i = 0
        while i < len(self.nodes) - 1:
            curr = self.nodes[i]
            next_ = self.nodes[i + 1]
            if (
                OpKind.rolling == curr.op.KIND
                and OpKind.rolling == next_.op.KIND
                and curr.op.params.get("window") == next_.op.params.get("window")
                and set(curr.outputs) == {next_}
            ):
                fused = FusedNode([curr.op, next_.op], OpKind.rolling)
                fused.inputs = curr.inputs
                for inp in fused.inputs:
                    inp.outputs = [o for o in inp.outputs if o != curr] + [fused]
                fused.outputs = next_.outputs
                for out in fused.outputs:
                    out.inputs = [i for i in out.inputs if i != next_] + [fused]
                del self.nodes[i + 1]
                self.nodes[i] = fused
                for col in _cols(curr.op.provides):
                    self.col_providers[col] = [p for p in self.col_providers[col] if p != curr]
                for col in _cols(next_.op.provides):
                    self.col_providers[col] = [p for p in self.col_providers[col] if p != next_]
                for col in _cols(fused.op.provides):
                    self.col_providers[col].append(fused)
                i += 1
            else:
                i += 1

    def _reorder_commutative_ops(self) -> None:
        """
        Determinize ordering of *parallel* elementwise ops without breaking chains.
        - We leave linear dependency chains (a -> b -> c) intact.
        - We only reorder siblings: elementwise ops that share the same inputs and
          have no edges between them. That preserves topo order for chains while
          removing nondeterminism among peers.
        """
        order = self.topological_sort()
        # Group candidate elementwise nodes by their (multiset) of input providers
        from collections import defaultdict

        groups: dict[tuple[int, ...], list[Node]] = defaultdict(list)

        for n in order:
            if getattr(n.op, "KIND", None) != OpKind.elementwise:
                continue
            key = tuple(sorted(id(inp) for inp in n.inputs))
            groups[key].append(n)

        for nodes in groups.values():
            if len(nodes) < 2:
                continue

            # If there is any edge between members, it's a chain (or partially so) — skip.
            has_internal_edge = any(b in a.outputs for a in nodes for b in nodes)
            if has_internal_edge:
                continue

            # Stable name-based order for siblings; do not touch edges, only list positions.
            sorted_nodes = sorted(nodes, key=lambda n: n.op.name)

            # Replace their positions in self.nodes to influence tie-breaking deterministically.
            idxs = sorted(self.nodes.index(n) for n in nodes)
            for idx, node in zip(idxs, sorted_nodes, strict=False):
                self.nodes[idx] = node

    def topological_sort(self) -> list[Node]:
        """Kahn's algorithm with parallel-ready structure."""
        indegree = {n: len(n.inputs) for n in self.nodes}
        queue = deque([n for n in self.nodes if indegree[n] == 0])
        order = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for out in n.outputs:
                indegree[out] -= 1
                if indegree[out] == 0:
                    queue.append(out)
        if len(order) != len(self.nodes):
            raise UnsupportedPlan("Graph has cycles")
        return order

    def has_cycle(self) -> bool:
        """Fast cycle detection using DFS with colored marking.

        Uses three-color marking: WHITE=unvisited, GRAY=visiting, BLACK=finished.
        A back edge to a GRAY node indicates a cycle.
        """
        if not self.nodes:
            return False

        # Three-color marking for cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[Node, int] = {}

        def dfs(node: Node) -> bool:
            """Return True if cycle detected from this node."""
            color[node] = GRAY

            # Check all outgoing edges
            for neighbor in getattr(node, "outputs", []):
                neighbor_color = color.get(neighbor, WHITE)

                if neighbor_color == GRAY:
                    # Back edge found - cycle exists
                    return True

                if neighbor_color == WHITE and dfs(neighbor):
                    return True

            color[node] = BLACK
            return False

        # Visit all nodes (handles disconnected components)
        return any(color.get(node, WHITE) == WHITE and dfs(node) for node in self.nodes)

    def shortest_path(self, start: Node, end: Node) -> list[Node]:
        """BFS-based shortest path with parallel frontier expansion capability."""
        if start not in self.nodes or end not in self.nodes:
            raise ValueError("Start or end node not in graph")

        if start == end:
            return [start]

        # BFS with parent tracking
        visited = {start}
        parent = {start: None}
        frontier = deque([start])

        while frontier:
            current = frontier.popleft()

            for neighbor in current.outputs:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current

                    if neighbor == end:
                        # Reconstruct path
                        path = []
                        node = end
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        return list(reversed(path))

                    frontier.append(neighbor)

        raise ValueError(f"No path from {start.op.name} to {end.op.name}")

    def connected_components(self) -> list[set[Node]]:
        """Find weakly connected components using union-find with path compression."""
        if not self.nodes:
            return []

        parent = {n: n for n in self.nodes}
        rank = dict.fromkeys(self.nodes, 0)

        def find(node: Node) -> Node:
            if parent[node] != node:
                parent[node] = find(parent[node])  # Path compression
            return parent[node]

        def union(n1: Node, n2: Node) -> None:
            root1, root2 = find(n1), find(n2)
            if root1 == root2:
                return
            # Union by rank
            if rank[root1] < rank[root2]:
                parent[root1] = root2
            elif rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root2] = root1
                rank[root1] += 1

        # Build undirected connectivity
        for node in self.nodes:
            for out in node.outputs:
                union(node, out)

        # Group by root
        components: dict[Node, set[Node]] = defaultdict(set)
        for node in self.nodes:
            components[find(node)].add(node)

        return list(components.values())

    def _to_dict(self) -> dict[Any, list[Any]]:
        return {n: n.outputs for n in self.nodes}

    def _prune(self, graph: Graph) -> Graph:
        reachable, stack = set(), [n for n in graph.nodes if not n.outputs]
        while stack:
            n = stack.pop()
            if n in reachable:
                continue
            reachable.add(n)
            stack.extend(n.inputs)
        pruned = Graph()
        for n in graph.nodes:
            if n in reachable:
                pruned.nodes.append(n)
                for col in _cols(n.op.provides):
                    pruned.col_providers[col].append(n)
                pruned.input_requires = set().union(*(_cols(n.op.requires) for n in pruned.nodes))
        return pruned


def serialize(graph: Graph) -> dict[str, Any]:
    """Serialize graph to self-describing IR with topology metadata."""
    order = graph.topological_sort()
    node_id_map = {n: idx for idx, n in enumerate(order)}

    serialized_nodes = []
    for node in order:
        node_data = node.to_ir()
        node_data["_id"] = node_id_map[node]
        node_data["_inputs"] = [node_id_map[inp] for inp in node.inputs]
        node_data["_outputs"] = [node_id_map[out] for out in node.outputs]
        serialized_nodes.append(node_data)

    return {
        "version": "1.0",
        "nodes": serialized_nodes,
        "input_requires": sorted(graph.input_requires),
        "topology": {
            "node_count": len(order),
            "edge_count": sum(len(n.outputs) for n in order),
            "components": len(graph.connected_components()),
        },
    }


def deserialize(data: dict[str, Any]) -> Graph:
    """Deserialize graph from IR, reconstructing topology and relationships."""
    if data.get("version") != "1.0":
        raise ValueError(f"Unsupported serialization version: {data.get('version')}")

    graph = Graph()
    node_list = []

    # First pass: create all nodes
    for node_data in data["nodes"]:
        # Reconstruct op from IR
        op_name = node_data["op"]
        kind_str = node_data["kind"]

        # Map kind string back to OpKind if available
        try:
            kind = OpKind[kind_str] if hasattr(OpKind, kind_str) else kind_str
        except (KeyError, AttributeError, NameError):
            kind = kind_str

        # Create fake op for deserialization
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            op = Op()  # type: ignore
        else:

            class DeserializedOp(Op):
                NAME = op_name
                KIND = kind if hasattr(kind, "value") else type("K", (), {"value": str(kind)})()

                def __init__(self, params_dict: dict[str, Any]):
                    self.params = params_dict
                    self.validate_params()

                @property
                def name(self) -> str:
                    return self.NAME

                def requires(self) -> set[str]:
                    return set(node_data.get("requires", []))

                def provides(self) -> set[str]:
                    return set(node_data.get("provides", []))

                def is_fittable(self) -> bool:
                    return node_data.get("fittable", False)

                def state_dict(self) -> dict[str, Any]:
                    return node_data.get("state", {})

                def validate_params(self) -> None:
                    # Deserialized ops accept already-validated IR; noop for robustness.
                    return None

                def to_ir(self) -> dict[str, Any]:
                    # Provide a minimal IR to satisfy the Op ABC and enable roundtrips
                    ir = {
                        "op": self.NAME,
                        "kind": getattr(self.KIND, "value", str(self.KIND)),
                        "params": dict(self.params),
                        "requires": sorted(node_data.get("requires", [])),
                        "provides": sorted(node_data.get("provides", [])),
                    }
                    if self.is_fittable():
                        ir["fittable"] = True
                        state = self.state_dict()
                        if state:
                            ir["state"] = dict(state)
                    return ir

            op = DeserializedOp(node_data.get("params", {}))

        node = graph.add_op(op)
        node_list.append(node)

    # Second pass: reconstruct edges
    for idx, node_data in enumerate(data["nodes"]):
        current_node = node_list[idx]
        for inp_id in node_data.get("_inputs", []):
            if inp_id < len(node_list):
                current_node.inputs.append(node_list[inp_id])
        for out_id in node_data.get("_outputs", []):
            if out_id < len(node_list):
                current_node.outputs.append(node_list[out_id])

    graph.input_requires = set(data.get("input_requires", []))
    return graph


__all__ = [
    "Graph",
    "Node",
    "SimpleNode",
    "FusedNode",
    "register_node_factory",
    "serialize",
    "deserialize",
]
