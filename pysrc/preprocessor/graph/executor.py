# preprocessor/graph/executor.py
import abc
import hashlib
import time
from typing import Any, Literal

from pysrc.core.errors import PreprocessingError, UnsupportedPlan
from pysrc.ops.mm_logkit import get_logger
from pysrc.ops.telemetry import SPAN_ATTR_UNKNOWN, SPAN_OP_EXECUTE, tracer
from pysrc.preprocessor.graph.backends.registry import get as get_lowering
from pysrc.preprocessor.graph.expr import get_polars_lowering
from pysrc.preprocessor.utils.cuda_runtime import capabilities
from pysrc.preprocessor.utils.errors import OOMRetry
from pysrc.preprocessor.utils.nvtx import nvtx_range
from pysrc.preprocessor.utils.plan_costs import HeuristicPlanner

logger = get_logger(__name__)

Engine = Literal["auto", "cpu", "gpu"]


class Executor(abc.ABC):
    """Base class for plan executors."""

    def __init__(self, backend: str, cache_size: int = 128):
        self.backend = backend
        self._cache: dict[str, Any] = {}  # Plan hash -> cached result
        self._cache_size = cache_size
        self.execution_history: list[dict[str, Any]] = []
        self.planner = HeuristicPlanner()

    @abc.abstractmethod
    def execute(self, plan: list[dict[str, Any]], data: Any, group_by: list[str]) -> Any:
        pass

    @nvtx_range("_execute_node")
    def _execute_node(self, ir: dict[str, Any], data: Any, group_by: list[str]) -> Any:
        """Execute a single IR node."""
        with tracer.start_as_current_span(SPAN_OP_EXECUTE) as span:
            span.set_attribute("op_name", str(ir.get("op", SPAN_ATTR_UNKNOWN)))
            span.set_attribute("op_version", str(ir.get("op_version", SPAN_ATTR_UNKNOWN)))
            span.set_attribute("determinism_tier", str(ir.get("determinism_tier", "D2")))
            return self._execute_node_inner(ir, data, group_by)

    def _execute_node_inner(self, ir: dict[str, Any], data: Any, group_by: list[str]) -> Any:
        lowering = get_lowering(self.backend, ir["op"])
        if not lowering and self.backend == "polars":
            lowering = get_polars_lowering(ir["op"])
        if not lowering:
            raise UnsupportedPlan(f"No lowering for {ir['op']} in {self.backend}")

        start = time.time()
        try:
            try:
                result = lowering(ir, data, group_by=group_by)
            except TypeError:
                # Some lowerings may not accept group_by; retry without it
                result = lowering(ir, data)
        except MemoryError:
            raise OOMRetry(f"OOM in {self.backend} for {ir['op']}")

        exec_time = time.time() - start
        self.execution_history.append({"op": ir["op"], "time": exec_time, "backend": self.backend})
        return result

    def _execute_node_cached(self, ir: dict[str, Any], data: Any, group_by: list[str]) -> Any:
        """Execute with simple LRU-style caching."""
        cache_key = self._make_cache_key(ir, data)

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._execute_node(ir, data, group_by)

        # Simple LRU: evict oldest if at capacity
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = result
        return result

    def _make_cache_key(self, ir: dict[str, Any], data: Any) -> str:
        """Create a cache key from IR and data identity."""
        ir_hash = hashlib.sha256(str(ir).encode()).hexdigest()[:16]
        data_hash = hashlib.sha256(repr(data).encode()).hexdigest()[:16]
        return f"{ir_hash}:{data_hash}"

    def evolve(self, threshold: float = 1.0) -> str | None:
        """Suggest an alternative backend if current one is slow."""
        slow_ops = [h for h in self.execution_history if h["time"] > threshold]
        if len(slow_ops) > len(self.execution_history) / 2:
            if self.backend == "cudf":
                alt = "polars"
            elif capabilities().has_cudf:
                alt = "cudf"
            else:
                alt = None

            if alt:
                logger.info("Evolving: Switching to %s due to slow ops", alt)
                return alt
        return None

    def _hash_plan(self, plan: list[dict[str, Any]]) -> str:
        return hashlib.sha256(str(plan).encode()).hexdigest()


class PolarsExecutor(Executor):
    """Polars-based executor with optional GPU acceleration."""

    def __init__(self, engine_pref: Engine = "auto"):
        super().__init__("polars")
        self.engine_pref = engine_pref

    def execute(self, plan: list[dict[str, Any]], data: Any, group_by: list[str]) -> Any:
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            lf = data
        elif isinstance(data, pl.DataFrame):
            lf = data.lazy()
        else:
            lf = pl.DataFrame(data).lazy()

        for ir in plan:
            lf = self._execute_node(ir, lf, group_by)

        out = self._collect_with_policy(lf)

        return out

    def _collect_with_policy(self, lf):
        """Collect LazyFrame with appropriate engine policy."""
        if self.engine_pref in ("gpu", "auto") and capabilities().has_polars_gpu:
            try:
                return lf.collect(engine="gpu")
            except Exception as exc:
                if self.engine_pref == "gpu":
                    raise PreprocessingError("governed path rejected gpu collect fallback") from exc
        return lf.collect(engine="streaming")


class CuDFExecutor(Executor):
    """cuDF-based executor for GPU acceleration."""

    def __init__(self, pool_size: str = "4GB"):
        super().__init__("cudf")
        self.pool_size = pool_size

    def _to_cudf(self, data: Any):
        """Convert common in-memory frames to cuDF."""
        try:
            import cudf
        except ImportError as e:
            raise PreprocessingError("cuDF not available") from e

        try:
            if isinstance(data, cudf.DataFrame):
                return data
        except TypeError:
            pass  # cudf.DataFrame may not be a proper type (e.g., mocked)

        # pandas -> cuDF
        try:
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                if hasattr(cudf, "from_pandas"):
                    return cudf.from_pandas(data)
                return cudf.DataFrame.from_pandas(data)
        except TypeError:
            pass  # isinstance failed on mocked type
        except ImportError:
            pass

        # polars -> cuDF (via pandas)
        try:
            import polars as pl

            if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
                df_pd = (
                    data.collect().to_pandas()
                    if isinstance(data, pl.LazyFrame)
                    else data.to_pandas()
                )
                if hasattr(cudf, "from_pandas"):
                    return cudf.from_pandas(df_pd)
                return cudf.DataFrame.from_pandas(df_pd)
        except TypeError:
            pass  # isinstance failed on mocked type
        except ImportError:
            pass

        # Fallback: try constructing directly
        try:
            return cudf.DataFrame(data)
        except Exception as e:
            raise PreprocessingError(
                f"Failed to convert {type(data).__name__} to cuDF DataFrame"
            ) from e

    def execute(self, plan: list[dict[str, Any]], data: Any, group_by: list[str]) -> Any:
        gdf = self._to_cudf(data)

        for ir in plan:
            gdf = self._execute_node(ir, gdf, group_by)

        return gdf


class ExecutorFactory:
    """Factory for creating executors."""

    _registry: dict[str, type[Executor]] = {"polars": PolarsExecutor, "cudf": CuDFExecutor}

    @classmethod
    def register(cls, backend: str, executor_cls: type[Executor]) -> None:
        cls._registry[backend] = executor_cls

    @classmethod
    def create(cls, backend: str = "auto", **kwargs) -> Executor:
        if backend == "auto":
            backend = "cudf" if capabilities().has_cudf else "polars"

        if backend == "polars":
            import pysrc.preprocessor.graph.backends.polars  # noqa: F401
        elif backend == "cudf":
            import pysrc.preprocessor.graph.backends.cudf  # noqa: F401

        exec_cls = cls._registry.get(backend)
        if not exec_cls:
            allowed = sorted(cls._registry.keys())
            logger.warning(
                "executor.unknown_backend",
                extra={"backend": backend, "allowed": allowed},
            )
            raise ValueError(f"Unknown backend: {backend} (allowed: {', '.join(allowed)})")

        return exec_cls(**kwargs)
