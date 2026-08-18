from pysrc.core.errors import PreprocessingError
from pysrc.ops.mm_logkit import get_logger

logger = get_logger(__name__)

_REGISTRY = {}


def register(backend: str, op: str, fn):
    key = (backend, op)
    if key in _REGISTRY:
        raise PreprocessingError(f"Lowering already registered for {key}")
    _REGISTRY[key] = fn
    logger.debug(f"Registered {key}")


def get(backend: str, op: str):
    return _REGISTRY.get((backend, op))


def list_ops(backend: str = None) -> list[str]:
    if backend:
        # Filter ops for the specified backend and ensure uniqueness
        ops = {op for be, op in _REGISTRY if be == backend}
    else:
        # List all ops with their backend, ensuring uniqueness
        ops = {f"{be}.{op}" for be, op in _REGISTRY}

    return sorted(ops)


def auto_register_from_utils():
    try:
        from pysrc.preprocessor.graph.utils.expr_builders import ExprFactory

        for name, builder in list(ExprFactory.registry.items()):

            def polars_lowering(ir, lf, _builder=builder, **kw):
                expr = _builder(backend="polars", **ir.get("params", {}))
                return expr(lf)

            register("polars", name, polars_lowering)

            def cudf_lowering(ir, gdf, _builder=builder, **kw):
                expr = _builder(backend="cudf", **ir.get("params", {}))
                return expr(gdf)

            register("cudf", name, cudf_lowering)
    except Exception as e:
        logger.debug("expr auto-register skipped: %s", e)
    try:
        from pysrc.preprocessor.graph.utils.transforms import TransformFactory

        for name, builder in list(TransformFactory._reg.items()):

            def transform_lowering(ir, df, _builder=builder, **kw):
                transform = _builder(**ir.get("params", {}))
                return transform(df)

            register("polars", f"transform_{name}", transform_lowering)
            register("cudf", f"transform_{name}", transform_lowering)
    except Exception as e:
        logger.debug("transform auto-register skipped: %s", e)


auto_register_from_utils()
