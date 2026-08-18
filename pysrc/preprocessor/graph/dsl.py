# preprocessor/graph/dsl.py
# Enhanced DSL: Integrates with factory/registry, supports backend hints, combinatoric with backend-aware compositions.
import importlib
from collections.abc import Callable
from typing import Any

from .factory import _OP_REGISTRY, resolve_name
from .graph import Graph
from .ops import Op, OpKind


# Abstract Op with backend hint for dynamic lowering.
class BackendAwareOp(Op):
    def __init__(self, **params: Any):
        super().__init__(**params)
        self.backend_hint: str | None = params.get("backend_hint")

    def to_ir(self) -> dict[str, Any]:
        ir = super().to_ir()
        if self.backend_hint:
            ir["backend_hint"] = self.backend_hint
        return ir


# Factory extension: Dynamic loading from ops_*.py, register backend-aware.
class OpFactory:
    @staticmethod
    def create(op_symbol: str, **params: Any) -> Op:
        # Parse token format "name:arg1,arg2" into name + inline params
        from .factory import _parse_token, register_builtin_ops

        base_name, inline_params = _parse_token(op_symbol)
        merged_params = {**inline_params, **params}

        key = resolve_name(base_name)
        if key not in _OP_REGISTRY:
            # Ensure builtins are registered before dynamic fallback
            register_builtin_ops()
            key = resolve_name(base_name)
        if key not in _OP_REGISTRY:
            # Dynamic import from ops_scaling, ops_sequence, etc.
            modules = ["ops_scaling", "ops_sequence", "ops_custom", "ops"]
            for mod_name in modules:
                try:
                    module = importlib.import_module(
                        f".{mod_name}", package="pysrc.preprocessor.graph"
                    )
                    op_cls = getattr(module, key.split(".")[-1].capitalize(), None)
                    if op_cls:
                        from .factory import register

                        register(key, op_cls)
                        break
                except ImportError:
                    continue
            else:
                raise ValueError(f"Unknown op '{op_symbol}'")
        op_cls = _OP_REGISTRY[key]
        return op_cls(**merged_params)


# DSL sugar functions.
def op(op_symbol: str, backend_hint: str | None = None, **params: Any) -> Op:
    params["backend_hint"] = backend_hint
    return OpFactory.create(op_symbol, **params)


def sequence(*ops: Op | str | dict[str, Any]) -> Graph:
    graph = Graph()
    prev_node = None
    for o in ops:
        if isinstance(o, str):
            o = op(o)
        elif isinstance(o, dict):
            o = op(o["symbol"], **o.get("params", {}))
        node = graph.add_op(o)
        if prev_node:
            node.inputs.append(prev_node)
            prev_node.outputs.append(node)
        prev_node = node
    return graph


def parallel(*subgraphs: Graph | Op | str | dict[str, Any]) -> Graph:
    combined = Graph()
    for sg in subgraphs:
        if isinstance(sg, (str, dict, Op)):
            sg = sequence(sg)
        combined.merge(sg)
    return combined


# Combinatoric operators.
def __rshift__(self, other: Op | str | dict[str, Any] | Graph) -> Graph:
    if not isinstance(self, Graph):
        self = sequence(self)
    if not isinstance(other, Graph):
        other = sequence(other)
    self.merge(other)
    # Auto-connect last of self to first of other if compatible.
    if self.nodes and other.nodes:
        self.nodes[-1].outputs.append(other.nodes[0])
        other.nodes[0].inputs.append(self.nodes[-1])
    return self


def __or__(self, other: Op | str | dict[str, Any] | Graph) -> Graph:
    return parallel(self, other)


Op.__rshift__ = __rshift__
Op.__or__ = __or__
Graph.__rshift__ = __rshift__
Graph.__or__ = __or__


# Self-evolving: Composite op with dynamic backend selection.
def combine_ops(
    name: str,
    *sub_ops: Op,
    backend_selector: Callable[[list[Op]], str] = lambda ops: (
        "cudf" if any(OpKind.scaling == o.KIND for o in ops) else "polars"
    ),
) -> type[Op]:
    class CompositeOp(BackendAwareOp):
        def __init__(self, **params: Any):
            super().__init__(**params)
            self.sub_ops = [o.clone() for o in sub_ops]
            self.backend_hint = backend_selector(self.sub_ops)

        def requires(self) -> set[str]:
            return set.union(*(o.requires() for o in self.sub_ops))

        def provides(self) -> set[str]:
            return set.union(*(o.provides() for o in self.sub_ops))

        def is_fittable(self) -> bool:
            return any(o.is_fittable() for o in self.sub_ops)

        def state_dict(self) -> dict[str, Any]:
            return {str(i): o.state_dict() for i, o in enumerate(self.sub_ops)}

        def load_state_dict(self, state: dict[str, Any]) -> "CompositeOp":
            for i, o in enumerate(self.sub_ops):
                o.load_state_dict(state.get(str(i), {}))
            return self

        def to_ir(self) -> dict[str, Any]:
            ir = super().to_ir()
            ir["sub_irs"] = [o.to_ir() for o in self.sub_ops]
            return ir

    from .factory import register

    register(name, CompositeOp)
    return CompositeOp
