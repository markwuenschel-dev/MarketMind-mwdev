from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from pysrc.preprocessor.graph.backends.registry import register as reg_register

from .cuda_runtime import capabilities
from .errors import UnsupportedAST


def register_expr(name: str, builder: Callable[..., Any]) -> None:
    ExprFactory.register(name, builder)
    reg_register(
        "polars", name, lambda ir, lf, **kw: ExprFactory.build(name, backend="polars").func(lf)
    )
    reg_register(
        "cudf", name, lambda ir, gdf, **kw: ExprFactory.build(name, backend="cudf").func(gdf)
    )


class Expr:
    def __init__(self, func: Callable[[Any], Any]) -> None:
        self.func = func

    def __add__(self, other: Expr) -> Expr:
        return Expr(lambda df: other.func(self.func(df)))


class ExprFactory:
    registry: dict[str, Callable[..., Any]] = {}  # Registry of expr builders

    @classmethod
    def register(cls, name: str, builder: Callable[..., Any]) -> None:
        cls.registry[name] = builder

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> Any:
        if name not in cls.registry:
            raise UnsupportedAST(f"Expr {name} not registered")
        caps = capabilities()
        if caps.has_polars_gpu:
            return cls.registry[name](**kwargs, backend="polars")
        elif caps.has_cudf:
            return cls.registry[name](**kwargs, backend="cudf")
        return Expr(cls.registry[name](**kwargs, backend="fallback"))


def safe_div_builder(eps: float = 1e-12, backend: str = "polars") -> Callable[[Any, Any], Any]:
    if backend == "polars":

        def safe_div(a: pl.Expr, b: pl.Expr) -> pl.Expr:
            return a / pl.max_horizontal(b, pl.lit(eps))

        return safe_div
    elif backend == "cudf":

        def safe_div(a: Any, b: Any) -> Any:
            return a / b.clip(lower=eps)

        return safe_div

    def safe_div(a: Any, b: Any) -> Any:
        return a / (b + eps)

    return safe_div


def zscore_builder(
    col: str, mean: Any | None = None, std: Any | None = None, backend: str = "polars"
) -> Callable[[Any], Any]:
    if backend == "polars":
        return lambda df: df.with_columns(
            ((pl.col(col) - (mean or pl.col(col).mean())) / (std or pl.col(col).std())).alias(
                f"{col}_zscore"
            )
        )
    elif backend == "cudf":
        return lambda df: df.assign(
            **{f"{col}_zscore": (df[col] - (mean or df[col].mean())) / (std or df[col].std())}
        )
    return lambda df: df.assign(
        **{f"{col}_zscore": (df[col] - (mean or df[col].mean())) / (std or df[col].std())}
    )


def bollinger_builder(
    col: str, window: int = 20, num_std: float = 2.0, backend: str = "polars"
) -> Callable[[Any], Any]:
    if backend == "polars":
        return lambda df: df.with_columns(
            ma=pl.col(col).rolling_mean(window),
            upper=pl.col(col).rolling_mean(window) + pl.col(col).rolling_std(window) * num_std,
            lower=pl.col(col).rolling_mean(window) - pl.col(col).rolling_std(window) * num_std,
        )
    elif backend == "cudf":
        return lambda df: df.assign(
            ma=df[col].rolling(window, min_periods=1).mean(),
            upper=df[col].rolling(window, min_periods=1).mean()
            + df[col].rolling(window, min_periods=1).std() * num_std,
            lower=df[col].rolling(window, min_periods=1).mean()
            - df[col].rolling(window, min_periods=1).std() * num_std,
        )
    return lambda df: df.assign(
        ma=df[col].rolling(window, min_periods=1).mean(),
        upper=df[col].rolling(window, min_periods=1).mean()
        + df[col].rolling(window, min_periods=1).std() * num_std,
        lower=df[col].rolling(window, min_periods=1).mean()
        - df[col].rolling(window, min_periods=1).std() * num_std,
    )


# Register example builders
ExprFactory.register("safe_div", safe_div_builder)
ExprFactory.register("zscore", zscore_builder)
ExprFactory.register("bollinger", bollinger_builder)
