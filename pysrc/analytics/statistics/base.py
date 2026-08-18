# py/analytics/statistics/base.py
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Any,
    Protocol,
    cast,
)

import numpy as np
import pandas as pd

# --- Optional imports ---
from pysrc.core.runtime.optional_imports import optional_import


class _LoggerLike(Protocol):
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> Any: ...


class _MlflowLike(Protocol):
    def active_run(self) -> bool: ...
    def log_metric(self, *args: Any, **kwargs: Any) -> None: ...
    def log_params(self, *args: Any, **kwargs: Any) -> None: ...


# Optional Polars: supported for inputs, but pandas is canonical inside tests
pl: Any = optional_import("polars")

# Optional MLflow: no-op if unavailable
mlflow: _MlflowLike | Any = optional_import("mlflow")
if mlflow is None:

    class _MLFStub:
        # A silent, no-op stub for MLflow to avoid conditional logic in tests
        @staticmethod
        def active_run() -> bool:
            return False

        @staticmethod
        def log_metric(*args: Any, **kwargs: Any) -> None: ...

        @staticmethod
        def log_params(*args: Any, **kwargs: Any) -> None: ...

    mlflow = _MLFStub()

# Statsmodels: core dependency for all statistical tests
from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore[import-untyped]
from statsmodels.tsa.stattools import (  # type: ignore[import-untyped]
    adfuller,
    grangercausalitytests,
    kpss,
)
from statsmodels.tsa.vector_ar.var_model import VAR  # type: ignore[import-untyped]
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # type: ignore[import-untyped]

# --- Project-specific imports ---
# Robust error types for clear, specific exceptions.
from pysrc.core.errors import BaseError, InvalidInputError, StatisticalTestError


# Specific statistical test exception hierarchy for precise error handling
class StatisticalComputationError(StatisticalTestError):
    """Raised when underlying numerical computation fails (LinAlg, convergence, etc.)"""

    pass


class StatisticalDataError(StatisticalTestError):
    """Raised when data structure/quality prevents test execution"""

    pass


class StatisticalBackendError(StatisticalTestError):
    """Raised when statsmodels/numpy backend encounters an error"""

    pass


# Centralized, structured logging from the ops module.
from pysrc.ops.mm_logkit import get_logger

# Observability decorator for timing and counting
from pysrc.ops.observability import instrument

# --- Module-level setup ---
__all__ = [
    "StatTest",
    "register_test",
    "StatTestFactory",
    "run_tests",
    "granger_causality_test",
    "johansen_cointegration_test",
    "adf_test",
    "kpss_test",
    "ljung_box_test",
    "InvalidInputError",
    "StatisticalBackendError",
    "StatisticalComputationError",
    "StatisticalDataError",
    "acorr_ljungbox",
    "adfuller",
    "coint_johansen",
    "grangercausalitytests",
    "kpss",
    "GrangercausalityTest",
    "JohansencointegrationTest",
    "AdfTest",
    "KpssTest",
    "LjungboxTest",
]
_logger: _LoggerLike | None = None


def _get_logger() -> _LoggerLike:
    # Ensure the module always has a logger with a .debug(...) method.
    # Some internal loggers expose .d/.info but not .debug; we adapt here.
    class _LoggerAdapter:
        def __init__(self, base: Any) -> None:
            self._base = base

        def debug(self, msg: str, *args: Any, **kwargs: Any) -> Any:
            # Prefer native .debug; fall back to .d, then .info, then no-op
            if hasattr(self._base, "debug"):
                return self._base.debug(msg, *args, **kwargs)
            if hasattr(self._base, "d"):
                return self._base.d(msg, *args, **kwargs)
            if hasattr(self._base, "info"):
                return self._base.info(msg, *args, **kwargs)
            return None  # no-op

        def __getattr__(self, name: str) -> Any:
            # Delegate all other attributes to the base logger
            return getattr(self._base, name)

    global _logger
    if _logger is None:
        base = get_logger(__name__)
        # Wrap only if .debug is missing
        _logger = cast(_LoggerLike, base) if hasattr(base, "debug") else _LoggerAdapter(base)
    return _logger


_test_registry: dict[str, type[StatTest]] = {}  # Holds all registered StatTest classes


# ---------------------------
# Internal Utilities
# ---------------------------

# Self-monitoring error boundary with circuit breaker pattern
_mlflow_failure_count = 0
_mlflow_circuit_open = False
_MLFLOW_CIRCUIT_THRESHOLD = 5


def _log_mlflow_metrics(prefix: str, metrics: Mapping[str, float]) -> None:
    """Safely log metrics with circuit breaker to prevent cascading failures"""
    global _mlflow_failure_count, _mlflow_circuit_open

    if _mlflow_circuit_open:
        return  # fail fast when circuit is open

    if not mlflow.active_run():
        return

    for k, v in metrics.items():
        try:
            mlflow.log_metric(f"{prefix.replace(' ', '_')}_{k}", float(v))
            _mlflow_failure_count = 0  # reset on success
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            _mlflow_failure_count += 1
            _get_logger().debug(
                "MLflow metric logging failed",
                extra={
                    "key": k,
                    "value": v,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "failure_count": _mlflow_failure_count,
                },
            )
            if _mlflow_failure_count >= _MLFLOW_CIRCUIT_THRESHOLD:
                _mlflow_circuit_open = True
                _get_logger().debug(
                    "MLflow circuit breaker opened", extra={"threshold": _MLFLOW_CIRCUIT_THRESHOLD}
                )


def _to_pd_series(x: Any, *, name: str) -> pd.Series:
    # Coerces various array-likes to a numeric pandas Series. Enforces numeric type.
    if isinstance(x, pd.Series):
        s = x
    elif pl and isinstance(x, pl.Series):
        s = x.to_pandas()
    elif isinstance(x, np.ndarray):
        s = pd.Series(x.flatten())  # Ensure 1D
    elif isinstance(x, (list, tuple)):
        s = pd.Series(x)
    else:
        raise InvalidInputError(
            f"Unsupported input type for '{name}'", details={"type": type(x).__name__}
        )

    if s.empty:
        raise InvalidInputError(f"Input '{name}' is empty or contains no data")

    # Attempt numeric conversion, raising on failure.
    try:
        s_numeric = pd.to_numeric(s, errors="coerce")
    except (TypeError, ValueError):
        raise InvalidInputError(
            f"Input '{name}' cannot be converted to numeric data",
            details={"type": type(s).__name__},
        )

    if s_numeric.isna().all():
        raise InvalidInputError(f"Input '{name}' contains no valid numeric data after coercion")
    return s_numeric


def _to_pd_dataframe(df: Any, *, name: str) -> pd.DataFrame:
    # Coerces various table-likes to a pandas DataFrame.
    if isinstance(df, pd.DataFrame):
        out = df
    elif pl and isinstance(df, pl.DataFrame):
        out = df.to_pandas()
    elif isinstance(df, (np.ndarray, dict)):
        out = pd.DataFrame(df)
    else:
        raise InvalidInputError(
            f"Unsupported input type for '{name}'", details={"type": type(df).__name__}
        )

    if out.empty or out.shape[1] == 0:
        raise InvalidInputError(f"DataFrame '{name}' is empty or has no columns")
    return out


def _select_var_order_aic(df: pd.DataFrame, maxlags_cap: int = 15) -> int:
    # Selects the optimal VAR lag order based on AIC. Used for Granger and Johansen.
    n_obs = df.shape[0]
    # Heuristic: maxlags should be reasonable relative to sample size.
    maxlags = min(maxlags_cap, max(5, n_obs // 5))
    if n_obs < maxlags * df.shape[1] * 2:  # Guard against too many features for too few obs
        lag = 1
        _get_logger().debug(
            "Insufficient data for robust auto-lag selection; defaulting to 1.",
            extra={"n_obs": n_obs, "maxlags": maxlags},
        )
        return lag

    try:
        var_model = VAR(df.dropna())
        selection = var_model.select_order(maxlags=maxlags)
        lag = int(selection.aic) if selection.aic is not None else 1
        return max(1, lag)  # Ensure lag is at least 1.
    except np.linalg.LinAlgError as e:
        _get_logger().debug(
            "VAR order selection computation failed, defaulting to lag=1",
            extra={
                "error": str(e),
                "error_type": "LinAlgError",
                "n_obs": n_obs,
                "maxlags": maxlags,
                "n_vars": df.shape[1],
            },
        )
        return 1
    except (ValueError, TypeError) as e:
        _get_logger().debug(
            "VAR order selection data error, defaulting to lag=1",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "n_obs": n_obs,
                "maxlags": maxlags,
            },
        )
        return 1
    except (RuntimeError, AttributeError) as e:
        _get_logger().debug(
            "VAR order selection backend error, defaulting to lag=1",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "statsmodels_function": "VAR.select_order",
            },
        )
        return 1


# ---------------------------
# ABC / Factory / Registration
# ---------------------------


def register_test(name: str | None = None) -> Callable[[type[StatTest]], type[StatTest]]:
    def decorator(cls: type[StatTest]) -> type[StatTest]:
        reg_name = (name or cls.__name__).removesuffix("Test").lower()
        if reg_name in _test_registry:
            # idempotent registration: just return
            if _test_registry[reg_name] is cls:
                return cls
            raise ValueError(f"Test name '{reg_name}' is already registered with different class.")
        _test_registry[reg_name] = cls
        try:
            _get_logger().debug(
                "Registered statistical test.",
                extra={"test_name": reg_name, "class_name": cls.__name__},
            )
        except AttributeError:
            # Logger not yet initialized during import; will log on first use
            pass
        return cls

    return decorator


class StatTest(ABC):
    # Abstract base class for all statistical test implementations.
    @abstractmethod
    def run(self, data: Any, **kwargs: Any) -> dict[str, Any]:
        # Each test must implement a `run` method to execute the test logic.
        raise NotImplementedError


class StatTestFactory:
    @classmethod
    def get_test(cls, test_name: str) -> StatTest:
        key = test_name.lower()
        if key not in _test_registry:
            available = ", ".join(sorted(_test_registry.keys()))
            raise ValueError(f"Unknown test: '{key}'. Available tests: [{available}]")
        return _test_registry[key]()

    @classmethod
    def run_test(cls, test_name: str, data: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return cls.get_test(test_name).run(data, **kwargs)
        except ValueError:
            # architect note: clean warning for expected unknown tests
            _get_logger().debug(
                "Unknown statistical test requested.", extra={"test_name": test_name}
            )
            raise
        except Exception as e:
            _get_logger().debug(
                "Unexpected statistical test error.",
                extra={"test_name": test_name, "error": str(e)},
            )
            raise


# ---------------------------
# Concrete Test Implementations
# ---------------------------


@register_test()
class GrangercausalityTest(StatTest):
    @instrument(name="grangercausality_test", measure_latency=True)
    def run(
        self,
        data: Any,
        *,
        maxlag: int | str = 5,
        significance_level: float = 0.05,
        verbose: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        # --- Input validation and preparation ---
        if isinstance(data, Mapping):
            if "y" not in data or "x" not in data:
                raise InvalidInputError(
                    "Input dict must contain keys 'y' and 'x'.", details={"keys": list(data.keys())}
                )
            y = _to_pd_series(data["y"], name="y")
            x = _to_pd_series(data["x"], name="x")
            df = pd.concat([y, x], axis=1).dropna()
            df.columns = ["y", "x"]
        else:
            df = _to_pd_dataframe(data, name="data").dropna()
            if "y" not in df.columns or "x" not in df.columns:
                raise InvalidInputError(
                    "Input DataFrame must contain columns 'y' and 'x'.",
                    details={"columns": list(df.columns)},
                )
            df = df[["y", "x"]]

        n_obs = df.shape[0]
        if n_obs < 10:
            raise InvalidInputError(
                "Insufficient observations for Granger causality test.",
                details={"n_obs": n_obs, "min_required": 10},
            )

        # --- Lag selection ---
        if isinstance(maxlag, str) and maxlag.lower() == "auto":
            final_maxlag = _select_var_order_aic(df)
            _get_logger().debug(
                "Auto-selected Granger maxlag via AIC.", extra={"maxlag": final_maxlag}
            )
        else:
            final_maxlag = int(maxlag)

        if n_obs <= final_maxlag + 1:
            raise InvalidInputError(
                "Not enough observations for the chosen maxlag.",
                details={"n_obs": n_obs, "maxlag": final_maxlag},
            )

        # --- Test execution ---
        try:
            res = grangercausalitytests(df, maxlag=final_maxlag, verbose=verbose)
        except np.linalg.LinAlgError as e:
            raise StatisticalComputationError(
                "Granger causality test numerical computation failed",
                details={
                    "error": str(e),
                    "error_type": "LinAlgError",
                    "maxlag": final_maxlag,
                    "n_observations": n_obs,
                    "hint": "Singular matrix likely due to highly correlated series or insufficient variation",
                },
            ) from e
        except (ValueError, TypeError) as e:
            raise StatisticalDataError(
                "Granger causality test received invalid data or parameters",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "maxlag": final_maxlag,
                    "n_observations": n_obs,
                    "data_shape": df.shape,
                },
            ) from e
        except (RuntimeError, AttributeError, ImportError) as e:
            raise StatisticalBackendError(
                "Granger causality test backend error",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "statsmodels_function": "grangercausalitytests",
                },
            ) from e

        # --- Result parsing and packaging ---
        p_values = [res[lag][0]["ssr_ftest"][1] for lag in range(1, final_maxlag + 1)]
        min_p_val = min(p_values)
        best_lag = p_values.index(min_p_val) + 1
        any_causality = bool(min_p_val < significance_level)

        summary = (
            f"Granger causality: best lag={best_lag} (p={min_p_val:.4g}). "
            f"Causality detected at α={significance_level}? {'Yes' if any_causality else 'No'}."
        )

        _log_mlflow_metrics("granger", {"min_p_value": min_p_val, "best_lag": float(best_lag)})
        return {
            "p_values": p_values,
            "min_p_value": min_p_val,
            "best_lag": best_lag,
            "any_causality": any_causality,
            "summary": summary,
        }


@register_test()
class JohansencointegrationTest(StatTest):
    @instrument(name="johansencointegration_test", measure_latency=True)
    def run(
        self,
        data: Any,
        *,
        det_order: int = 0,
        k_ar_diff: int | str = 1,
        significance_level: float = 0.05,
        **_: Any,
    ) -> dict[str, Any]:
        # --- Input validation and preparation ---
        df = _to_pd_dataframe(data, name="data").dropna()
        n_obs, n_vars = df.shape
        if n_vars < 2:
            raise InvalidInputError(
                "Johansen test requires at least two variables.", details={"n_vars": n_vars}
            )
        if n_obs < 25:
            raise InvalidInputError(
                "Insufficient observations for Johansen test.",
                details={"n_obs": n_obs, "min_required": 25},
            )

        # --- Lag selection ---
        if isinstance(k_ar_diff, str) and k_ar_diff.lower() == "auto":
            final_k_ar_diff = _select_var_order_aic(df)
            _get_logger().debug(
                "Auto-selected Johansen k_ar_diff via AIC.", extra={"k_ar_diff": final_k_ar_diff}
            )
        else:
            final_k_ar_diff = int(k_ar_diff)

        # --- Test execution ---
        try:
            joh = coint_johansen(df.values, det_order, final_k_ar_diff)
        except np.linalg.LinAlgError as e:
            raise StatisticalComputationError(
                "Johansen cointegration test numerical computation failed",
                details={
                    "error": str(e),
                    "error_type": "LinAlgError",
                    "det_order": det_order,
                    "k_ar_diff": final_k_ar_diff,
                    "n_observations": n_obs,
                    "n_variables": n_vars,
                    "hint": "Eigenvalue decomposition failed, likely due to near-singular covariance matrix",
                },
            ) from e
        except (ValueError, TypeError) as e:
            raise StatisticalDataError(
                "Johansen cointegration test received invalid data or parameters",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "det_order": det_order,
                    "k_ar_diff": final_k_ar_diff,
                    "n_observations": n_obs,
                    "n_variables": n_vars,
                },
            ) from e
        except (RuntimeError, AttributeError, ImportError) as e:
            raise StatisticalBackendError(
                "Johansen cointegration test backend error",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "statsmodels_function": "coint_johansen",
                },
            ) from e

        # --- Result parsing and packaging ---
        # Critical values are typically checked at 90%, 95%, 99% (indices 0, 1, 2)
        crit_idx = {0.1: 0, 0.05: 1, 0.025: 1, 0.01: 2}.get(significance_level, 1)
        rank = sum(1 for i in range(n_vars) if joh.lr1[i] > joh.cvt[i, crit_idx])
        summary = (
            f"Johansen test suggests cointegration rank of {rank} "
            f"(at α={significance_level}, k_ar_diff={final_k_ar_diff})."
        )

        _log_mlflow_metrics("johansen", {"rank_estimate": float(rank)})
        return {
            "trace_stat": joh.lr1.tolist(),
            "max_eig_stat": joh.lr2.tolist(),
            "crit_values": joh.cvt.tolist(),
            "rank": rank,
            "summary": summary,
        }


@register_test()
class AdfTest(StatTest):
    @instrument(name="adf_test", measure_latency=True)
    def run(
        self,
        data: Any,
        *,
        significance_level: float = 0.05,
        regression: str = "c",
        autolag: str | None = "AIC",
        maxlag: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        s = _to_pd_series(data, name="series")
        s_clean = s.dropna()
        if s_clean.size < 10:
            raise InvalidInputError(
                "Not enough observations for ADF test.",
                details={"n_obs": s_clean.size, "min_required": 10},
            )

        # Check for constant series (with numerical tolerance for floating-point precision)
        series_std = s_clean.std()
        if series_std < 1e-10 or (series_std == 0):
            raise InvalidInputError(
                "Series is constant - ADF test cannot be performed on constant data."
            )

        try:
            stat, p, usedlag, nobs, crit, icbest = adfuller(
                s_clean.values, regression=regression, autolag=autolag, maxlag=maxlag
            )
        except np.linalg.LinAlgError as e:
            raise StatisticalComputationError(
                "ADF test numerical computation failed",
                details={
                    "error": str(e),
                    "error_type": "LinAlgError",
                    "regression": regression,
                    "autolag": autolag,
                    "maxlag": maxlag,
                    "series_size": s_clean.size,
                },
            ) from e
        except (ValueError, TypeError) as e:
            raise StatisticalDataError(
                "ADF test received invalid data or parameters",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "regression": regression,
                    "autolag": autolag,
                    "series_size": s_clean.size,
                },
            ) from e
        except (RuntimeError, AttributeError, ImportError) as e:
            raise StatisticalBackendError(
                "ADF test backend error",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "statsmodels_function": "adfuller",
                },
            ) from e

        is_stationary = bool(p < significance_level)  # H0: unit root is present (non-stationary)
        summary = (
            f"ADF test: p-value={p:.4g}. "
            f"Series is likely {'stationary' if is_stationary else 'non-stationary'} at α={significance_level}."
        )

        _log_mlflow_metrics("adf", {"p_value": p, "test_stat": stat})
        return {
            "test_statistic": stat,
            "p_value": p,
            "used_lag": usedlag,
            "crit_values": crit,
            "is_stationary": is_stationary,
            "summary": summary,
        }


@register_test()
class KpssTest(StatTest):
    @instrument(name="kpss_test", measure_latency=True)
    def run(
        self,
        data: Any,
        *,
        significance_level: float = 0.05,
        nlags: int | str = "auto",
        regression: str = "c",
        **_: Any,
    ) -> dict[str, Any]:
        s = _to_pd_series(data, name="series")
        s_clean = s.dropna()
        if s_clean.size < 10:
            raise InvalidInputError(
                "Not enough observations for KPSS test.",
                details={"n_obs": s_clean.size, "min_required": 10},
            )

        # Check for constant series (with numerical tolerance for floating-point precision)
        series_std = s_clean.std()
        if series_std < 1e-10 or (series_std == 0):
            raise InvalidInputError(
                "Series is constant - KPSS test cannot be performed on constant data."
            )

        try:
            stat, p, lags, crit = kpss(s_clean.values, nlags=nlags, regression=regression)
        except np.linalg.LinAlgError as e:
            raise StatisticalComputationError(
                "KPSS test numerical computation failed",
                details={
                    "error": str(e),
                    "error_type": "LinAlgError",
                    "nlags": nlags,
                    "regression": regression,
                },
            ) from e
        except (ValueError, TypeError) as e:
            raise StatisticalDataError(
                "KPSS test received invalid data or parameters",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "series_size": s_clean.size,
                },
            ) from e
        except (RuntimeError, AttributeError, ImportError) as e:
            raise StatisticalBackendError(
                "KPSS test backend error", details={"error": str(e), "error_type": type(e).__name__}
            ) from e

        is_stationary = bool(p > significance_level)  # H0: series is stationary
        summary = (
            f"KPSS test: p-value={p:.4g}. "
            f"Series is likely {'stationary' if is_stationary else 'non-stationary'} at α={significance_level}."
        )

        _log_mlflow_metrics("kpss", {"p_value": p, "test_stat": stat})
        return {
            "test_statistic": stat,
            "p_value": p,
            "lags": lags,
            "crit_values": crit,
            "is_stationary": is_stationary,
            "summary": summary,
        }


@register_test()
class LjungboxTest(StatTest):
    @instrument(name="ljungbox_test", measure_latency=True)
    def run(
        self,
        data: Any,
        *,
        lags: int | Sequence[int] | str | None = None,
        significance_level: float = 0.05,
        model_df: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        s = _to_pd_series(data, name="series")
        s_clean = s.dropna()
        n = s_clean.size
        if n < 10:
            raise InvalidInputError(
                "Not enough observations for Ljung-Box test.",
                details={"n_obs": n, "min_required": 10},
            )

        # --- Lag selection ---
        if lags is None or (isinstance(lags, str) and lags.lower() == "auto"):
            # A common heuristic for max lag choice.
            lags_list = list(range(1, min(11, max(2, n // 5))))
        elif isinstance(lags, int):
            lags_list = list(range(1, lags + 1))
        else:
            lags_list = sorted({int(l) for l in lags if int(l) > 0})

        if not lags_list or n <= max(lags_list) + model_df:
            raise InvalidInputError(
                "Not enough observations for the chosen lags.",
                details={"n_obs": n, "max_lag": max(lags_list, default=0), "model_df": model_df},
            )

        # --- Test execution ---
        try:
            res = acorr_ljungbox(
                s_clean.values, lags=lags_list, boxpierce=False, return_df=True, model_df=model_df
            )
        except np.linalg.LinAlgError as e:
            raise StatisticalComputationError(
                "Ljung-Box test numerical computation failed",
                details={
                    "error": str(e),
                    "error_type": "LinAlgError",
                    "lags": lags_list,
                    "max_lag": max(lags_list),
                    "model_df": model_df,
                    "series_size": n,
                    "hint": "Autocorrelation matrix computation failed",
                },
            ) from e
        except (ValueError, TypeError) as e:
            raise StatisticalDataError(
                "Ljung-Box test received invalid data or parameters",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "lags": lags_list,
                    "model_df": model_df,
                    "series_size": n,
                },
            ) from e
        except (RuntimeError, AttributeError, ImportError) as e:
            raise StatisticalBackendError(
                "Ljung-Box test backend error",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "statsmodels_function": "acorr_ljungbox",
                },
            ) from e

        # --- Result parsing and packaging ---
        p_values = res["lb_pvalue"].tolist()
        is_significant = [pv < significance_level for pv in p_values]
        autocorr_detected = any(is_significant)
        summary = (
            f"Ljung-Box test: autocorrelation {'detected' if autocorr_detected else 'not detected'} "
            f"within {max(lags_list)} lags at α={significance_level}."
        )

        return {
            "p_values": p_values,
            "lags": lags_list,
            "is_significant": is_significant,
            "autocorr_detected": autocorr_detected,
            "summary": summary,
        }


# ---------------------------
# Public API Functions
# ---------------------------

from concurrent.futures import ThreadPoolExecutor, as_completed


def run_tests(
    tests: list[str | Mapping[str, Any]],
    data: Any,
    parallel: bool = False,
    max_workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Composable runner for executing multiple statistical tests on a given dataset.

    Args:
        tests: List of test names or test specifications
        data: Input data for all tests
        parallel: If True, execute tests in parallel (thread-safe)
        max_workers: Max parallel workers (defaults to CPU count)

    Returns:
        Dict mapping test names to results or errors
    """

    def _run_single_test(test_spec: str | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        """Execute a single test with comprehensive error handling"""
        if isinstance(test_spec, str):
            name, kwargs = test_spec, {}
        else:
            name = str(test_spec.get("name"))
            kwargs = dict(test_spec.get("kwargs", {}))

        try:
            result = StatTestFactory.run_test(name, data, **kwargs)
            return (name, result)
        except StatisticalDataError as e:
            _get_logger().debug(
                "Data validation error in test", extra={"test_name": name, "error": e.to_dict()}
            )
            return (name, {"error": e.to_dict()})
        except StatisticalComputationError as e:
            _get_logger().debug(
                "Numerical computation error in test",
                extra={"test_name": name, "error": e.to_dict()},
            )
            return (name, {"error": e.to_dict()})
        except StatisticalBackendError as e:
            _get_logger().debug(
                "Backend error in test", extra={"test_name": name, "error": e.to_dict()}
            )
            return (name, {"error": e.to_dict()})
        except BaseError as e:
            _get_logger().debug(
                "Statistical test failed", extra={"test_name": name, "error": e.to_dict()}
            )
            return (name, {"error": e.to_dict()})
        except (
            ValueError,
            TypeError,
            OverflowError,
            ZeroDivisionError,
            np.linalg.LinAlgError,
            RuntimeError,
            KeyError,
            IndexError,
            AttributeError,
        ) as e:
            _get_logger().debug(
                "Unexpected error in test",
                extra={"test_name": name, "error_type": type(e).__name__, "error": str(e)},
            )
            return (
                name,
                {
                    "error": {
                        "code": "UNHANDLED_EXCEPTION",
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                },
            )

    if not parallel or len(tests) == 1:
        # Sequential execution
        return dict(_run_single_test(t) for t in tests)

    # Parallel execution with proper error aggregation
    results: dict[str, dict[str, Any]] = {}
    workers = max_workers or min(len(tests), (os.cpu_count() or 4))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_test = {executor.submit(_run_single_test, t): t for t in tests}

        for future in as_completed(future_to_test):
            try:
                name, result = future.result(timeout=30.0)  # 30s per test timeout
                results[name] = result
            except TimeoutError:
                test_spec = future_to_test[future]
                test_name = (
                    test_spec if isinstance(test_spec, str) else test_spec.get("name", "unknown")
                )
                _get_logger().debug("Test execution timeout", extra={"test_name": test_name})
                results[test_name] = {
                    "error": {"code": "TIMEOUT", "message": "Test exceeded 30s timeout"}
                }
            except Exception as e:
                test_spec = future_to_test[future]
                test_name = (
                    test_spec if isinstance(test_spec, str) else test_spec.get("name", "unknown")
                )
                _get_logger().debug(
                    "Future execution error", extra={"test_name": test_name, "error": str(e)}
                )
                results[test_name] = {"error": {"code": "EXECUTION_ERROR", "message": str(e)}}

    return results


# Convenience wrappers
def adf_test(series: Any, **kwargs: Any) -> dict[str, Any]:
    return StatTestFactory.run_test("adf", series, **kwargs)


def kpss_test(series: Any, **kwargs: Any) -> dict[str, Any]:
    return StatTestFactory.run_test("kpss", series, **kwargs)


def ljung_box_test(series: Any, **kwargs: Any) -> dict[str, Any]:
    return StatTestFactory.run_test("ljungbox", series, **kwargs)


def granger_causality_test(x: Any, y: Any, **kwargs: Any) -> dict[str, Any]:
    return StatTestFactory.run_test("grangercausality", {"y": y, "x": x}, **kwargs)


def johansen_cointegration_test(data: Any, **kwargs: Any) -> dict[str, Any]:
    return StatTestFactory.run_test("johansencointegration", data, **kwargs)
