from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from time import perf_counter

from pysrc.core.errors import PreprocessingError
from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.utils.cuda_runtime import capabilities
from pysrc.preprocessor.utils.errors import SchemaMismatch, UnsupportedAST

logger = get_logger(__name__)

try:
    import cudf
except ImportError:
    cudf = None
try:
    import polars as pl
except ImportError:
    pl = None
try:
    import pandas as pd
except ImportError:
    pd = None

_prof_lock = threading.Lock()
_prof_metrics: dict[str, float] = {}


def _is_seq_of_str(x) -> bool:
    return isinstance(x, (list, tuple, set)) and all(isinstance(c, str) for c in x)


def _as_list(
    x: str | list[str] | tuple[str, ...] | None,
) -> Sequence[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def _derive_out_names(
    input_cols: str | list[str] | tuple[str, ...] | None,
    suffix: str | None = None,
    out_col: str | list[str] | None = None,
) -> list[str]:
    if out_col:
        return list(_as_list(out_col))
    cols = _as_list(input_cols)
    if not suffix:
        return list(cols)
    return [f"{c}_{suffix}" for c in cols]


class ColumnOp(ABC):
    @abstractmethod
    def apply(self, df, cols: Iterable[str], **kwargs): ...

    def validate(self, df, cols: Iterable[str]) -> None:
        if not _is_seq_of_str(cols):
            raise ValueError("cols must be a sequence of column names (list/tuple/set[str])")
        missing = [c for c in cols if c not in getattr(df, "columns", [])]
        if missing:
            raise SchemaMismatch(f"Missing columns: {missing}", details={"missing": missing})


def profile_op(func: Callable) -> Callable:
    def wrapper(self, *args, **kwargs):
        caps = capabilities()
        key = f"{type(self).__name__}_{'cudf' if caps.has_cudf else 'polars' if caps.has_polars_gpu else 'cpu'}"
        t0 = perf_counter()
        out = func(self, *args, **kwargs)
        dt = perf_counter() - t0
        with _prof_lock:
            if key not in _prof_metrics or dt < _prof_metrics[key]:
                _prof_metrics[key] = dt
                logger.debug("profile:%s best=%.6fs", key, dt)
        return out

    return wrapper


def _fallback_or_raise(*, exc: Exception, governed: bool, op_name: str, df):
    if governed:
        raise PreprocessingError(f"governed path rejected {op_name} fallback: {exc}") from exc
    return df


class CastNumeric(ColumnOp):
    @profile_op
    def apply(self, df, cols: Iterable[str], dtype="float32", governed: bool = False):
        self.validate(df, cols)
        caps = capabilities()
        try:
            if cudf and isinstance(df, cudf.DataFrame) and caps.has_cudf:
                return df.astype(dict.fromkeys(cols, dtype), copy=False)
            if pl and isinstance(df, pl.DataFrame):
                pl_dtype_map = {
                    "float32": pl.Float32,
                    "float64": pl.Float64,
                    "int16": pl.Int16,
                    "int32": pl.Int32,
                    "int64": pl.Int64,
                    "uint16": pl.UInt16,
                    "uint32": pl.UInt32,
                    "uint64": pl.UInt64,
                }
                pl_dtype = pl_dtype_map.get(dtype, pl.Float32)
                lf = df.lazy().with_columns([pl.col(c).cast(pl_dtype) for c in cols])
                try:
                    if caps.has_polars_gpu:
                        engine = pl.GpuEngine(uvm=True)
                        return lf.collect(engine=engine)
                except AttributeError:
                    pass
                return lf.collect()
            if pd and isinstance(df, pd.DataFrame):
                return df.astype(dict.fromkeys(cols, dtype), copy=False)
            raise ValueError("No supported backend")
        except Exception as exc:
            logger.warning("Cast failed: %s; returning original df", exc)
            return _fallback_or_raise(exc=exc, governed=governed, op_name="cast_numeric", df=df)


class PromoteCategorical(ColumnOp):
    @profile_op
    def apply(
        self, df, cols: Iterable[str], ordered: bool = False, governed: bool = False, **kwargs
    ):
        self.validate(df, cols)
        caps = capabilities()
        try:
            if cudf and isinstance(df, cudf.DataFrame) and caps.has_cudf:
                return df.astype(dict.fromkeys(cols, "category"))
            if pl and isinstance(df, pl.DataFrame):
                lf = df.lazy().with_columns(pl.col(c).cast(pl.Categorical) for c in cols)
                try:
                    if caps.has_polars_gpu:
                        engine = pl.GpuEngine()
                        return lf.collect(engine=engine)
                except AttributeError:
                    pass
                return lf.collect()
            if pd and isinstance(df, pd.DataFrame):
                dtype = pd.CategoricalDtype(ordered=ordered)
                return df.astype(dict.fromkeys(cols, dtype), copy=False)
            raise ValueError("No supported backend for categorical promotion")
        except Exception as exc:
            logger.warning("Categorical promotion failed: %s; fallback", exc)
            return _fallback_or_raise(
                exc=exc, governed=governed, op_name="promote_categorical", df=df
            )


class ColumnOpFactory:
    registry: dict[str, Callable[..., ColumnOp]] = {
        "cast_numeric": CastNumeric,
        "promote_categorical": PromoteCategorical,
    }

    @classmethod
    def register(cls, name: str, op_class: Callable[..., ColumnOp]):
        cls.registry[name] = op_class

    @classmethod
    def build(cls, name: str, **kwargs) -> ColumnOp:
        if name not in cls.registry:
            raise UnsupportedAST(f"ColumnOp {name} not registered")
        return cls.registry[name](**kwargs)


def op_chain(*op_names: str, **kwargs) -> Callable:
    ops = [ColumnOpFactory.build(n, **kwargs.get(n, {})) for n in op_names]

    def chained(df, cols, **chain_kwargs):
        for op in ops:
            df = op.apply(df, cols, **chain_kwargs)
        return df

    return chained


def live_after(cols: Iterable[str]) -> list[str]:
    if isinstance(cols, str):
        raise ValueError("cols must be an iterable of column names, not a str")
    seen = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def ensure_unique(cols: Iterable[str], sep: str = "__") -> list[str]:
    if not cols:
        return []
    counts = Counter(cols)
    result: list[str] = []
    current_counts = dict.fromkeys(counts, 0)
    for c in cols:
        k = current_counts[c]
        result.append(c if k == 0 else f"{c}{sep}{k}")
        current_counts[c] += 1
    return result


def save_metrics(file: str = "metrics.json"):
    with open(file, "w", encoding="utf-8") as handle:
        json.dump(_prof_metrics, handle)


__all__ = [
    "CastNumeric",
    "ColumnOp",
    "ColumnOpFactory",
    "PromoteCategorical",
    "_as_list",
    "_derive_out_names",
    "_is_seq_of_str",
    "_prof_metrics",
    "ensure_unique",
    "live_after",
    "op_chain",
    "profile_op",
    "save_metrics",
]
