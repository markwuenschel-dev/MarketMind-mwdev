import os

from pysrc.core.errors import PreprocessingError
from pysrc.core.runtime.optional_imports import optional_import
from pysrc.core.validation import validate_dataframe
from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.graph.backends.registry import get as get_lowering
from pysrc.preprocessor.graph.backends.registry import register
from pysrc.preprocessor.graph.executor import Executor
from pysrc.preprocessor.ops.common.columns import op_chain  # For combinatoric column prep
from pysrc.preprocessor.utils.cuda_runtime import capabilities, init_rmm_pool
from pysrc.preprocessor.utils.errors import OOMRetry
from pysrc.preprocessor.utils.io_gpu import read_parquet_gpu
from pysrc.preprocessor.utils.plan_costs import (  # Integrate for self-evolving plan opt
    HeuristicPlanner,
    PlanSegment,
)
from pysrc.preprocessor.utils.specs import SpecFactory  # For dynamic group/window composition
from pysrc.preprocessor.utils.torch_bridge import to_torch_batch  # Optional tensor bridge
from pysrc.preprocessor.utils.validate import schema_checks

logger = get_logger(__name__)


def robust_scaler_cudf(ir, gdf, group_by=None, **_):
    """
    Robust scaler for cuDF. Handles fresh fit and replay-from-state.
    Accepts optional group_by for forward compatibility.
    """
    preprocessing = optional_import("sklearn.preprocessing")
    RobustScaler = (
        getattr(preprocessing, "RobustScaler", None) if preprocessing is not None else None
    )
    optional_import("cupy")
    if RobustScaler is None:
        raise PreprocessingError("scikit-learn RobustScaler is required for cuDF robust scaling")

    params = ir.get("params", {})
    cols = list(params.get("cols", []))
    ql, qh = params.get("quantile_range", (25, 75))
    with_center = bool(params.get("with_centering", True))
    with_scale = bool(params.get("with_scaling", True))

    state = ir.get("state")
    needs_fit = not isinstance(state, dict) or ("center" not in state and "median" not in state)

    def _to_pylist(x):
        # tolerate cuDF Series, cupy arrays, numpy, etc.
        try:
            if hasattr(x, "to_arrow"):
                return x.to_arrow().to_pylist()
            if hasattr(x, "get"):  # cupy
                import numpy as _np

                return _np.asarray(x.get()).tolist()
            import numpy as _np

            return _np.asarray(x).tolist()
        except Exception:
            return None  # let caller decide defaults

    if needs_fit:
        scaler = RobustScaler(
            quantile_range=(ql, qh),
            with_centering=with_center,
            with_scaling=with_scale,
        )
        gdf[cols] = scaler.fit_transform(gdf[cols])

        center = getattr(scaler, "center_", None)
        scale = getattr(scaler, "scale_", None)
        center_list = _to_pylist(center) if center is not None else None
        scale_list = _to_pylist(scale) if scale is not None else None

        if center_list is None:
            center_list = [0.0] * len(cols)
        if scale_list is None:
            scale_list = [1.0] * len(cols)

        ir["state"] = {"center": center_list, "scale": scale_list}
        return gdf

    # replay from state
    center = state.get("center") or state.get("median")  # tolerate alternative key
    scale = state.get("scale") or state.get("iqr")
    import cupy as _cp

    center = _cp.asarray(center) if center is not None else 0.0
    scale = _cp.asarray(scale) if scale is not None else 1.0

    if with_center:
        gdf[cols] = gdf[cols] - center
    if with_scale:
        gdf[cols] = gdf[cols] / scale
    return gdf


register("cudf", "scaling.robust", robust_scaler_cudf)


class CuDFExecutor(Executor):
    _rmm_inited = False
    _planner = HeuristicPlanner()  # Self-evolving planner instance

    def __init__(self, *, pool_size="4GB", to_torch=False):  # Add flag for tensor output
        super().__init__("cudf")
        self.to_torch = to_torch  # Dev: Enable for ML bridging
        caps = capabilities()
        if not caps.has_cudf:
            raise PreprocessingError("cuDF not available")
        if not CuDFExecutor._rmm_inited:
            try:

                def _parse_bytes(x):  # Dev: Utility for flexible pool sizing
                    if isinstance(x, int):
                        return x
                    s = str(x).strip().upper()
                    mult = 1
                    if s.endswith("KB"):
                        mult, s = 1 << 10, s[:-2]
                    elif s.endswith("MB"):
                        mult, s = 1 << 20, s[:-2]
                    elif s.endswith("GB"):
                        mult, s = 1 << 30, s[:-2]
                    elif s.endswith("TB"):
                        mult, s = 1 << 40, s[:-2]
                    return int(float(s) * mult)

                init_rmm_pool(pool_size=_parse_bytes(pool_size))
            except Exception as e:
                logger.warning("RMM init failed: %s; default allocator", e)
            CuDFExecutor._rmm_inited = True
        if caps.has_kvikio:
            os.environ.setdefault("LIBCUDF_CUFILE_POLICY", "ALWAYS")

    def read_parquet(self, path, columns=None, byte_range=None):
        return read_parquet_gpu(path, columns=columns, byte_range=byte_range)

    def execute(self, compiled_plan, df):
        validate_dataframe(df)
        frame = df
        # Dev: Optimize plan segments dynamically with sample for efficiency
        sample = frame.head(1024) if hasattr(frame, "head") else frame.iloc[:1024]
        segments = [
            PlanSegment(
                ops=[get_lowering("cudf", ir["op"])],
                spec=SpecFactory.build("group", by=compiled_plan.group_by)
                if compiled_plan.group_by
                else None,
            )
            for ir in compiled_plan.nodes
        ]
        optimized_segments = self._planner.optimize(segments, sample)
        node_irs = list(compiled_plan.nodes)
        for seg, node_ir in zip(optimized_segments, node_irs, strict=False):
            for op in seg.ops:
                try:
                    frame = op(node_ir, frame)
                except OOMRetry as e:
                    logger.warning("OOM in cuDF; retry hint=%s", e.details.get("retry_hint"))
                    raise
        # Dev: Apply combinatoric column prep if needed (e.g., cast before scaling)
        if "cols" in compiled_plan.params:
            frame = op_chain("cast_numeric")(frame, compiled_plan.params["cols"], dtype="float32")
        # Dev: Optional tensor bridge for downstream
        if self.to_torch:
            frame = to_torch_batch(frame, cols=compiled_plan.params.get("cols", []))
        report = getattr(compiled_plan, "report", None)
        frame.__report__ = report() if callable(report) else {}
        schema_checks(frame, expected=compiled_plan.expected_schema, strict=True)  # Post-validation
        return frame


# --------- Phase 0 feature op GPU stubs ---------
# These raise NotImplementedError until a cuDF implementation is provided.
# They are registered so the registry is complete; the polars fallback handles execution.


def feature_returns_cudf(ir, gdf, **_):
    """GPU stub for feature.returns. Raises NotImplementedError; use polars backend."""
    raise NotImplementedError(
        "feature.returns GPU lowering not implemented. "
        "Register a cuDF implementation in backends/cudf.pysrc."
    )


register("cudf", "feature.returns", feature_returns_cudf)


def feature_sma_cudf(ir, gdf, **_):
    """GPU stub for feature.sma. Raises NotImplementedError; use polars backend."""
    raise NotImplementedError(
        "feature.sma GPU lowering not implemented. "
        "Register a cuDF implementation in backends/cudf.pysrc."
    )


register("cudf", "feature.sma", feature_sma_cudf)


def feature_rsi_cudf(ir, gdf, **_):
    """GPU stub for feature.rsi. Raises NotImplementedError; use polars backend."""
    raise NotImplementedError(
        "feature.rsi GPU lowering not implemented. "
        "Register a cuDF implementation in backends/cudf.pysrc."
    )


register("cudf", "feature.rsi", feature_rsi_cudf)


def data_load_csv_cudf(ir, gdf, **_):
    """GPU stub for data.load_csv. Raises NotImplementedError; use polars backend."""
    raise NotImplementedError(
        "data.load_csv GPU lowering not implemented. "
        "Register a cuDF implementation in backends/cudf.pysrc."
    )


register("cudf", "data.load_csv", data_load_csv_cudf)
