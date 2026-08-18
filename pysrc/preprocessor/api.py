from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pysrc.core.errors import PreprocessingError
from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.utils.errors import OOMRetry
from pysrc.preprocessor.utils.nvtx import nvtx_range

from .graph.executor import ExecutorFactory
from .graph.factory import build_graph
from .graph.planner import Planner

logger = get_logger(__name__)

Backend = Literal["auto", "cpu", "gpu", "polars", "cudf"]


@dataclass
class PlanSpec:
    """Specification for a preprocessing plan."""

    ops: list[dict[str, Any]] = field(default_factory=list)
    target: dict[str, Any] | None = None
    sequence: dict[str, Any] | None = None
    scaling: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


def merge_specs(*specs: PlanSpec) -> PlanSpec:
    """Merge multiple PlanSpecs into one, preserving provenance."""
    ops: list[dict[str, Any]] = []
    meta_chain: list[dict[str, Any]] = []
    target = None
    sequence = None
    scaling = None

    for s in specs:
        if not s:
            continue
        ops.extend(s.ops or [])
        if s.meta:
            meta_chain.append(s.meta)
        # last-wins for optional fields
        target = s.target or target
        sequence = s.sequence or sequence
        scaling = s.scaling or scaling

    # Keep a chain of provenance for auditability
    meta = {"provenance_chain": [m.get("provenance") for m in meta_chain if "provenance" in m]}
    # Preserve any top-level export / extra hints from the last spec that had them
    for s in reversed(specs):
        if s and s.meta:
            for k, v in s.meta.items():
                if k != "provenance":
                    meta.setdefault(k, v)

    return PlanSpec(ops=ops, target=target, sequence=sequence, scaling=scaling, meta=meta)


class ModelRegistry:
    """Registry for model references used in preprocessing ops."""

    _by_name: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, obj: Any) -> None:
        cls._by_name[name] = obj

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._by_name.get(name)

    @classmethod
    def clear(cls) -> None:
        cls._by_name.clear()


def resolve_models_in_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve model_ref strings to actual model objects."""
    out = []
    for o in ops:
        if "model_ref" in o and isinstance(o["model_ref"], str):
            o = dict(o)
            model = ModelRegistry.get(o["model_ref"])
            if model is not None:
                o["__model__"] = model
        out.append(o)
    return out


@dataclass(frozen=True)
class Plan:
    """Immutable preprocessing plan."""

    ops: list[str]
    params: dict[str, list[dict[str, Any]]]
    group_by: list[str]
    version: str = "1.0"


def get_executor(backend: Backend = "auto"):
    """Get an executor for the specified backend."""
    if backend == "auto":
        return ExecutorFactory.create("auto")
    if backend in ("cpu", "gpu"):
        from .graph.executor import PolarsExecutor

        return PolarsExecutor(engine_pref=backend)
    return ExecutorFactory.create(backend)


@nvtx_range("compile_plan")
def _compile(plan: Plan, backend: Backend = "auto"):
    """Compile a Plan into executable IR."""
    graph = build_graph(plan.ops, plan.params)
    planner = Planner()
    compiled = planner.plan(graph, plan.group_by)

    # Warn if a backend hint in an op disagrees with the user request
    if backend != "auto":
        for ir in compiled:
            hint = ir.get("backend_hint")
            if hint and hint != backend:
                logger.warning(
                    "Backend hint %s differs from requested %s for op %s",
                    hint,
                    backend,
                    ir.get("op"),
                )
    return compiled, planner


def _to_polars(out: Any, pl_module: Any) -> Any:
    """Convert output to Polars DataFrame if needed."""
    # Already a Polars DataFrame
    if isinstance(out, pl_module.DataFrame):
        return out

    # Polars LazyFrame -> collect
    if isinstance(out, pl_module.LazyFrame):
        return out.collect()

    # pandas DataFrame -> Polars
    try:
        import pandas as pd

        if isinstance(out, pd.DataFrame):
            return pl_module.from_pandas(out)
    except ImportError:
        pass

    # Generic fallback
    try:
        return pl_module.DataFrame(out)
    except (TypeError, ValueError) as e:
        raise PreprocessingError(
            f"Failed to convert {type(out).__name__} to Polars DataFrame"
        ) from e


@nvtx_range("run_plan")
def run(
    df: Any,
    plan: Plan | dict[str, Any],
    *,
    backend: Backend = "auto",
    optimize: bool | None = None,  # Accepted but ignored
    pressure: str | None = None,  # Accepted but ignored
    **_ignored,
) -> Any:
    """Execute a preprocessing plan on a DataFrame."""
    # Convert dict to Plan if needed
    if isinstance(plan, dict):
        ops = plan.get("ops", [])
        plan_obj = Plan(
            ops=[op.get("kind", "") for op in ops],
            params={op.get("kind", f"op_{i}"): [op] for i, op in enumerate(ops)},
            group_by=plan.get("group_by", []),
        )
    else:
        plan_obj = plan

    compiled, planner = _compile(plan_obj, backend)
    executor = get_executor(backend=backend)

    try:
        out = executor.execute(compiled, df, plan_obj.group_by)
    except OOMRetry as e:
        raise PreprocessingError("governed path rejected retry under changed semantics") from e
    except Exception as e:
        raise PreprocessingError(f"Execution failed: {e}") from e

    planner.update_from_history(executor.execution_history)

    # Determine if we need Polars output
    try:
        import polars as pl
    except ImportError:
        pl = None

    want_polars = backend == "polars"
    is_input_polars = False

    if pl is not None:
        is_input_polars = isinstance(df, (pl.DataFrame, pl.LazyFrame))

    if (want_polars or is_input_polars) and pl is not None:
        out = _to_polars(out, pl)

    return out


def stream(plan: Plan, *, backend: Backend = "auto"):
    """Streaming execution (not implemented)."""
    raise NotImplementedError("Streaming execution is not implemented yet")


class PreprocessorBuilder:
    """Fluent builder for preprocessing plans."""

    def __init__(self, backend: Backend = "auto"):
        self._ops: list[str] = []
        self._params: dict[str, list[dict[str, Any]]] = {}
        self._group_by: list[str] = ["symbol"]
        self._backend: Backend = backend

    def add_op(self, op_symbol: str, **params) -> PreprocessorBuilder:
        """Add an operation to the plan."""
        self._ops.append(op_symbol)
        if params:
            self._params.setdefault(op_symbol, []).append(params)
        else:
            self._params.setdefault(op_symbol, [])
        return self

    def set_group_by(self, cols: list[str]) -> PreprocessorBuilder:
        """Set the group-by columns."""
        self._group_by = list(cols)
        return self

    def set_backend(self, backend: Backend) -> PreprocessorBuilder:
        """Set the execution backend."""
        self._backend = backend
        return self

    def from_dict(self, cfg: dict[str, Any]) -> PreprocessorBuilder:
        """Load configuration from a dictionary."""
        if "ops" in cfg:
            self._ops.extend(cfg["ops"])
        if "params" in cfg and isinstance(cfg["params"], dict):
            for k, v in cfg["params"].items():
                if isinstance(v, list):
                    self._params.setdefault(k, []).extend(v)
                elif isinstance(v, dict):
                    self._params.setdefault(k, []).append(v)
        if "group_by" in cfg:
            self._group_by = list(cfg["group_by"])
        if "backend" in cfg:
            self._backend = cfg["backend"]
        return self

    def build_plan(self) -> Plan:
        """Build an immutable Plan from the current configuration."""
        return Plan(
            ops=list(self._ops),
            params={k: [dict(p) for p in v] for k, v in self._params.items()},
            group_by=list(self._group_by),
        )

    def build_runner(self) -> Callable[[Any], Any]:
        """Build a callable that executes the plan on a DataFrame."""
        plan = self.build_plan()
        backend = self._backend

        def _runner(df: Any) -> Any:
            return run(df, plan, backend=backend)

        return _runner

    def add_dsl_op(self, op_symbol: str, backend_hint: str = None, **params) -> PreprocessorBuilder:
        """Add an operation with a backend hint."""
        params["backend_hint"] = backend_hint
        return self.add_op(op_symbol, **params)

    def add_sequence(self, *ops: str | dict[str, Any]) -> PreprocessorBuilder:
        """Add a sequence of operations."""
        for o in ops:
            if isinstance(o, str):
                self.add_op(o)
            else:
                self.add_op(o["symbol"], **o.get("params", {}))
        return self

    def add_parallel(self, *builders: PreprocessorBuilder) -> PreprocessorBuilder:
        """Merge operations from multiple builders in parallel."""
        for b in builders:
            plan = b.build_plan()
            self._ops.extend(plan.ops)
            for k, v in plan.params.items():
                self._params.setdefault(k, []).extend(v)
        return self

    def add_transform(self, transform_name: str, **params) -> PreprocessorBuilder:
        """Add a named transform."""
        self.add_op(f"transform_{transform_name}", **params)
        return self


__all__ = [
    "Backend",
    "ModelRegistry",
    "Plan",
    "PlanSpec",
    "PreprocessorBuilder",
    "get_executor",
    "merge_specs",
    "resolve_models_in_ops",
    "run",
    "stream",
]
