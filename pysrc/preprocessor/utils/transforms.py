# utils/transforms.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import reduce  # For iterative compose
from time import perf_counter
from typing import Any

from pysrc.ops.mm_logkit import get_logger

logger = get_logger(__name__)

try:
    import polars as pl
except Exception:
    pl = None
try:
    import cudf
except Exception:
    cudf = None

from pysrc.preprocessor.ops.common.columns import op_chain

from .cuda_runtime import capabilities
from .errors import OOMRetry, SchemaMismatch, UnsupportedAST


class Transform(ABC):
    """Combinable, backend-aware transform. Holds a private callable `_fn`."""

    def __init__(self, fn: Callable[[Any], Any], backend: str | None = None):
        self._fn = fn
        self.backend = backend or self._select_backend()

    def _select_backend(self) -> str:
        caps = capabilities()
        if caps.has_cudf:
            return "cudf"
        if caps.has_polars_gpu:
            return "polars"
        return "cpu"

    def __call__(self, df):
        self.validate(df)
        return self._fn(df)

    def apply(self, df):
        """Compatibility shim if callers use .apply(df)."""
        return self(df)

    def __add__(self, other: Transform) -> Transform:
        def combined(df):
            try:  # Add try for chain robustness
                df = self(df)
                return other(df)
            except MemoryError:
                raise OOMRetry("Transform chain OOM", {"reduce_cols": True})

        return CompositeTransform(combined, backend=self.backend)

    @abstractmethod
    def validate(self, df) -> None: ...


class CompositeTransform(Transform):
    def __init__(self, fn: Callable[[Any], Any], backend: str | None = None):
        super().__init__(fn, backend)

    def validate(self, df) -> None:
        pass


def profile_transform(func: Callable) -> Callable:
    import threading

    metrics: dict[str, float] = {}
    lock = threading.Lock()

    def wrapper(self: Transform, df):
        key = f"{type(self).__name__}_{self.backend}"
        t0 = perf_counter()
        out = func(self, df)
        dt = perf_counter() - t0
        with lock:
            if key not in metrics or dt < metrics[key]:
                metrics[key] = dt
                logger.debug("evolved: %s best=%.6fs", key, dt)
        return out

    return wrapper


class NormalizeTransform(Transform):
    def __init__(self, col: str, mean: float | None = None, std: float | None = None):
        self.col, self.mean, self.std = col, mean, std

        def _fn(df):
            c = self.col
            if pl and isinstance(df, pl.DataFrame):
                mu = self.mean if self.mean is not None else pl.col(c).mean()
                sd = self.std if self.std is not None else pl.col(c).std()
                return df.with_columns(((pl.col(c) - mu) / sd).alias(f"{c}_norm"))
            # cuDF / pandas fallback
            m = self.mean if self.mean is not None else df[c].mean()
            s = self.std if self.std is not None else df[c].std()
            return df.assign(**{f"{c}_norm": (df[c] - m) / s})

        super().__init__(_fn)

    @profile_transform
    def apply(self, df):
        return self(df)

    def validate(self, df) -> None:
        if self.col not in getattr(df, "columns", []):
            raise SchemaMismatch(f"Column {self.col} not found")


class BollingerTransform(Transform):
    def __init__(
        self, col: str, window: int = 20, num_std: float = 2.0, output_cols: list[str] | None = None
    ):
        self.col, self.window, self.num_std = col, window, num_std
        self.output_cols = output_cols or [f"{col}_ma", f"{col}_upper", f"{col}_lower"]

        def _fn(df):
            c, w, k = self.col, self.window, self.num_std
            if pl and isinstance(df, pl.DataFrame):
                ma = pl.col(c).rolling_mean(w)
                sd = pl.col(c).rolling_std(w)
                return df.with_columns(
                    [
                        ma.alias(self.output_cols[0]),
                        (ma + sd * k).alias(self.output_cols[1]),
                        (ma - sd * k).alias(self.output_cols[2]),
                    ]
                )
            # cuDF / pandas
            roll = df[c].rolling(window=w, min_periods=1)
            ma = roll.mean()
            sd = roll.std()
            return df.assign(
                **{
                    self.output_cols[0]: ma,
                    self.output_cols[1]: ma + sd * k,
                    self.output_cols[2]: ma - sd * k,
                }
            )

        super().__init__(_fn)

    @profile_transform
    def apply(self, df):
        return self(df)

    def validate(self, df) -> None:
        if self.col not in getattr(df, "columns", []):
            raise SchemaMismatch(f"Column {self.col} not found")


class LogTransform(Transform):
    def __init__(self, col: str, base: float = 10.0, eps: float = 1e-12):
        self.col, self.base, self.eps = col, base, eps

        def _fn(df):
            c = self.col
            if pl and isinstance(df, pl.DataFrame):
                return df.with_columns(
                    pl.col(c).clip(lower_bound=self.eps).log(self.base).alias(f"{c}_log")
                )
            # cuDF / pandas
            if hasattr(df[c], "log"):
                return df.assign(**{f"{c}_log": (df[c].clip(lower=self.eps)).log(self.base)})
            import math

            return df.assign(
                **{f"{c}_log": (df[c].clip(lower=self.eps)).apply(lambda x: math.log(x, self.base))}
            )

        super().__init__(_fn)

    @profile_transform
    def apply(self, df):
        return self(df)

    def validate(self, df) -> None:
        if self.col not in getattr(df, "columns", []):
            raise SchemaMismatch(f"Column {self.col} not found")


class MinMaxScaleTransform(Transform):
    def __init__(
        self,
        col: str,
        min_val: float | None = None,
        max_val: float | None = None,
        eps: float = 1e-12,
    ):
        self.col, self.min_val, self.max_val, self.eps = col, min_val, max_val, eps

        def _fn(df):
            c = self.col
            mn = self.min_val if self.min_val is not None else df[c].min()
            mx = self.max_val if self.max_val is not None else df[c].max()
            denom = (mx - mn) if mx is not None and mn is not None else self.eps
            if denom == 0:
                logger.debug(f"MinMax denom zero for {c}; using eps={self.eps}")
            return df.assign(**{f"{c}_scaled": (df[c] - mn) / (denom if denom != 0 else self.eps)})

        super().__init__(_fn)

    @profile_transform
    def apply(self, df):
        return self(df)

    def validate(self, df) -> None:
        if self.col not in getattr(df, "columns", []):
            raise SchemaMismatch(f"Column {self.col} not found")


class ToTorchTransform(Transform):
    def __init__(
        self, cols: list[str], dtypes: dict[str, Any] | None = None, include_lengths: bool = False
    ):
        self.cols, self.dtypes, self.include_lengths = cols, dtypes, include_lengths
        from .torch_bridge import bridge_factory

        def _fn(df):
            bridge = bridge_factory(df)
            return bridge.to_torch(df, self.cols, self.dtypes, self.include_lengths)

        super().__init__(_fn)

    def validate(self, df) -> None:
        missing = [c for c in self.cols if c not in getattr(df, "columns", [])]
        if missing:
            raise SchemaMismatch(f"Missing columns: {missing}")


class TransformFactory:
    _reg: dict[str, Callable[..., Transform]] = {}

    @classmethod
    def register(cls, name: str, builder: Callable[..., Transform]):
        cls._reg[name] = builder

    @classmethod
    def build(cls, name: str, **kwargs) -> Transform:
        if name not in cls._reg:
            raise UnsupportedAST(f"Transform {name} not registered")
        return cls._reg[name](**kwargs)

    @classmethod
    def compose(cls, *names: str, **kwargs) -> Transform:
        tx = [cls.build(n, **kwargs.get(n, {})) for n in names]
        return reduce(lambda a, b: a + b, tx) if tx else CompositeTransform(lambda df: df)

    @classmethod
    def auto_register_from_exprs(cls):
        """Dynamic: If expr_builders present, wrap/register as transforms."""
        try:
            from .expr_builders import ExprFactory

            for name, builder in ExprFactory.registry.items():

                def wrapped_builder(**kw):
                    def _fn(df):
                        expr = builder(**kw)
                        return expr(df) if callable(expr) else df.with_columns(expr)

                    return CompositeTransform(_fn)

                cls.register(f"expr_{name}", wrapped_builder)
        except ImportError:
            pass  # Silent if no expr_builders


TransformFactory.auto_register_from_exprs()  # Call once

TransformFactory.register("normalize", lambda **kw: NormalizeTransform(**kw))
TransformFactory.register("bollinger", lambda **kw: BollingerTransform(**kw))
TransformFactory.register("log", lambda **kw: LogTransform(**kw))
TransformFactory.register("minmax_scale", lambda **kw: MinMaxScaleTransform(**kw))
TransformFactory.register("to_torch", lambda **kw: ToTorchTransform(**kw))


def feature_engineer_chain(cols: list[str]) -> Transform:
    """Example: cast -> normalize -> bollinger on the first column."""

    def _fn(df):
        df = op_chain("cast_numeric")(df, cols)
        df = NormalizeTransform(col=cols[0])(df)
        df = BollingerTransform(col=cols[0])(df)
        return df

    return CompositeTransform(_fn)
