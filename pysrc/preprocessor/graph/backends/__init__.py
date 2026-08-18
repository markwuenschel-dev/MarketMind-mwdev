# py/preprocessor/graph/backends/__init__.py

from typing import Literal

from pysrc.ops.mm_logkit import get_logger

from .registry import get, list_ops, register

logger = get_logger(__name__)


def _load_cudf_executor():
    from .cudf import CuDFExecutor

    return CuDFExecutor


def get_executor(backend: Literal["auto", "polars", "cudf", "cpu", "gpu"] = "auto"):
    if backend in ("polars", "cpu"):
        from .polars import PolarsExecutor

        return PolarsExecutor(engine_pref="cpu")

    if backend in ("cudf", "gpu"):
        try:
            return _load_cudf_executor()()
        except Exception as exc:
            logger.warning(
                "Could not initialize CuDFExecutor: %s. GPU backend might not be available.",
                exc,
            )
            if backend != "auto":
                raise

    if backend == "auto":
        try:
            return _load_cudf_executor()()
        except Exception:  # noqa: BLE001
            logger.debug(
                "Auto backend: CuDF not available, falling back to Polars.",
                exc_info=True,
            )
            from .polars import PolarsExecutor

            return PolarsExecutor()

    raise ValueError(f"Unsupported backend requested: '{backend}'")


def __getattr__(name: str):
    if name == "CuDFExecutor":
        return _load_cudf_executor()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CuDFExecutor", "get", "get_executor", "list_ops", "register"]
