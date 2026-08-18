# preprocessor/graph/planner.py
from collections import defaultdict, deque
from typing import Any

from pysrc.preprocessor.utils.plan_costs import HeuristicPlanner as BasePlanner
from pysrc.preprocessor.utils.plan_costs import PlanSegment, score_segment
from pysrc.preprocessor.utils.specs import SpecFactory

from .graph import FusedNode, Graph, Node
from .ops import OpKind


class Planner(BasePlanner):
    def __init__(self):
        super().__init__()
        self.history: dict[str, dict[str, float]] = {}
        self.metrics: dict[str, float] = {}  # Track per-node execution metrics
        self.weights: dict[str, float] = {
            "time": 1.0,
            "memory": 1.0,
        }  # Cost weights for optimization

    def plan(self, graph: Graph, group_by: list[str]) -> list[dict[str, Any]]:
        graph.optimize()  # run local graph opts
        pruned_graph = self._prune(graph)
        ordered = pruned_graph.topological_sort()
        segments = self._segment(ordered)
        for seg in segments:
            self._fuse_segment(seg)
        for node in ordered:
            if node.op.KIND in (OpKind.rolling, OpKind.sequence, OpKind.scaling):
                node.op.params["group_by"] = group_by
                node.op.params["window_spec"] = SpecFactory.build("window", partition_by=group_by)
        ir = [n.to_ir() for n in ordered]
        cost = sum(score_segment(PlanSegment(ops=[n.op]), None) for n in ordered)
        if cost > self._threshold():
            self._reoptimize(ordered)
            ir = [n.to_ir() for n in ordered]
        self.update_from_history(
            [{"op": n.op.name, "time": cost / max(1, len(ordered))} for n in ordered]
        )
        return ir

    def _prune(self, graph: Graph) -> Graph:
        """Prune unreachable nodes by traversing backwards from terminal nodes."""
        # Find terminal nodes (nodes with no outputs)
        terminals = [n for n in graph.nodes if not n.outputs]
        if not terminals:
            # No terminals means all nodes feed into something else
            # Fall back to keeping all nodes
            terminals = graph.nodes

        reachable = set()
        queue = deque(terminals)
        while queue:
            n = queue.popleft()
            if n in reachable:
                continue
            reachable.add(n)
            queue.extend(n.inputs)

        pruned = Graph()
        pruned.nodes = [n for n in graph.nodes if n in reachable]
        pruned.col_providers = defaultdict(list)
        for n in pruned.nodes:
            for col in n.op.provides:
                pruned.col_providers[col].append(n)

        # Update input_requires: columns needed but not provided by any pruned node
        provided_by_pruned = set()
        for n in pruned.nodes:
            provided_by_pruned.update(n.op.provides)
        pruned.input_requires = graph.input_requires - provided_by_pruned
        return pruned

    def _segment(self, ordered: list[Node]) -> list[list[Node]]:
        """Group consecutive nodes of the same OpKind for fusion opportunities."""
        segments = []
        current = []
        for n in ordered:
            if current and n.op.KIND != current[0].op.KIND:
                segments.append(current)
                current = []
            current.append(n)
        if current:
            segments.append(current)
        return segments

    def _fuse_segment(self, seg: list[Node]) -> None:
        """Fuse adjacent nodes in a segment if they form a linear chain."""
        i = 0
        while i < len(seg) - 1:
            curr, next_n = seg[i], seg[i + 1]
            if curr.op.KIND == next_n.op.KIND and set(curr.outputs) == {next_n}:
                fused = FusedNode([curr.op, next_n.op], curr.op.KIND)
                fused.inputs = curr.inputs
                fused.outputs = next_n.outputs
                seg[i] = fused
                del seg[i + 1]
            else:
                i += 1

    def update_from_history(self, records: list[dict[str, Any]]) -> None:
        """Update planner history with new performance records."""
        for record in records:
            op_name = record.get("op", "unknown")
            time_cost = record.get("time", 0.0)

            # Update metrics for reoptimization
            self.metrics[op_name] = time_cost

            if op_name not in self.history:
                self.history[op_name] = {"time": time_cost, "count": 1}
            else:
                # Update with exponential moving average
                alpha = 0.7  # Learning rate
                self.history[op_name]["time"] = (
                    alpha * time_cost + (1 - alpha) * self.history[op_name]["time"]
                )
                self.history[op_name]["count"] += 1

    def _threshold(self) -> float:
        """Compute dynamic threshold for triggering reoptimization."""
        if not self.history:
            return 10.0  # Default for cold start
        times = [h["time"] for h in self.history.values()]
        mean = sum(times) / len(times)
        std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
        return mean + 2 * std  # Dynamic: mean + 2σ

    def _reoptimize(self, ordered: list[Node]) -> None:
        """Reorder nodes based on historical performance metrics."""
        ordered.sort(key=lambda n: self.metrics.get(n.op.name, float("inf")), reverse=True)

    def _evolve_weights(self) -> None:
        """Adjust cost weights based on observed performance."""
        for _op_name, time in self.metrics.items():
            if time > 1.0:
                self.weights["time"] = min(self.weights["time"] + 0.1, 5.0)  # Cap at 5.0
