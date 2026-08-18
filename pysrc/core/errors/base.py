from __future__ import annotations

import os
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Protocol, cast


# lazy imports to avoid cycles; resolve at first use
class _LoggerLike(Protocol):
    def warning(self, *args: Any, **kwargs: Any) -> Any: ...
    def debug(self, *args: Any, **kwargs: Any) -> Any: ...


class _MetricsLike(Protocol):
    def counter(self, *args: Any, **kwargs: Any) -> Any: ...
    def record_counter(self, *args: Any, **kwargs: Any) -> Any: ...


_logger: _LoggerLike | None = None
_metrics: _MetricsLike | None = None


def _get_logger() -> _LoggerLike:
    global _logger
    if _logger is None:
        try:
            from pysrc.ops.mm_logkit import get_logger as _get

            _logger = cast(_LoggerLike, _get(__name__))
        except Exception:

            class _Noop:
                def warning(self, *args: Any, **kwargs: Any) -> None:
                    pass

                def debug(self, *args: Any, **kwargs: Any) -> None:
                    pass

            _logger = _Noop()
    return _logger


def _get_metrics() -> _MetricsLike:
    global _metrics
    if _metrics is None:
        try:
            from pysrc.ops.observability import get_metrics as _gm

            _metrics = cast(_MetricsLike, _gm())
        except Exception:

            class _Noop:
                def counter(self, *a: Any, **k: Any) -> tuple[str, None]:
                    return ("noop", None)

                def record_counter(self, *a: Any, **k: Any) -> None:
                    pass

            _metrics = _Noop()
    return _metrics


_ENABLE_LOGGING: bool = os.getenv("MM_EXCEPTION_LOGGING", "0") == "1"
_ENABLE_METRICS: bool = os.getenv("MM_EXCEPTION_METRICS", "1") == "1"
_RATE_LIMIT_WINDOW_SEC: float = float(os.getenv("MM_EXCEPTION_RL_WINDOW_SEC", "60"))
_RATE_LIMIT_THRESHOLD: int = int(os.getenv("MM_EXCEPTION_RL_THRESHOLD", "500"))


class ErrorCodeEnum(StrEnum):
    GENERIC = "GENERIC_ERROR"
    DATA_FETCH = "DATA_FETCH_ERROR"
    API_CONNECTION = "API_CONNECTION_ERROR"
    STREAM_CONNECTION = "STREAM_CONNECTION_ERROR"
    DATA_TIMEOUT = "DATA_TIMEOUT_ERROR"
    DATA_VALIDATION = "DATA_VALIDATION_ERROR"
    SCHEMA_VALIDATION = "SCHEMA_VALIDATION_ERROR"
    FILE_FORMAT = "FILE_FORMAT_ERROR"
    STATISTICAL_TEST = "STATISTICAL_TEST_ERROR"
    DATA_DRIFT = "DATA_DRIFT_ERROR"
    NO_DATA = "NO_DATA_ERROR"
    CONFIG_VALIDATION = "CONFIG_VALIDATION_ERROR"
    PREPROCESSING = "PREPROCESSING_ERROR"
    MODEL_TRAINING = "MODEL_TRAINING_ERROR"
    TRADING_EXECUTION = "TRADING_EXECUTION_ERROR"
    INVALID_INPUT = "INVALID_INPUT_ERROR"
    DATA_PRECONDITION = "DATA_PRECONDITION_ERROR"
    MODEL_CHECKPOINT = "MODEL_CHECKPOINT_ERROR"
    MODEL_INFERENCE = "MODEL_INFERENCE_ERROR"
    UNSUPPORTED_PLAN = "UNSUPPORTED_PLAN"
    PIT_VIOLATION = "PIT_VIOLATION_ERROR"
    STALENESS = "STALENESS_ERROR"


@dataclass
class _ErrorStorm:
    timestamps: list[float] = field(default_factory=list)
    suppressed: int = 0


class _RateLimiter:
    # per-code leaky bucket limiting
    def __init__(self, window_sec: float, threshold: int):
        self._window = window_sec
        self._threshold = threshold
        self._store: dict[str, _ErrorStorm] = {}
        self._lock = threading.RLock()

    def should_log(self, code: str) -> bool:
        now = time.time()
        with self._lock:
            s = self._store.setdefault(code, _ErrorStorm())
            cutoff = now - self._window
            s.timestamps[:] = [t for t in s.timestamps if t >= cutoff]
            if len(s.timestamps) >= self._threshold:
                s.suppressed += 1
                return False
            s.timestamps.append(now)
            return True

    def suppressed_count(self, code: str) -> int:
        with self._lock:
            s = self._store.get(code)
            return s.suppressed if s else 0


_rate_limiter = _RateLimiter(_RATE_LIMIT_WINDOW_SEC, _RATE_LIMIT_THRESHOLD)


def _get_trace_context() -> dict[str, str]:
    try:
        from pysrc.ops.observability import get_tracing  # lazy import

        tracing = get_tracing()
        if not tracing:
            return {}
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span:
            return {}
        ctx = span.get_span_context()
        # avoid invalid zero IDs
        if getattr(ctx, "trace_id", 0) and getattr(ctx, "span_id", 0):
            return {"trace_id": f"{ctx.trace_id:032x}", "span_id": f"{ctx.span_id:016x}"}
    except Exception:
        pass
    return {}


def _get_tenant_context() -> dict[str, str]:
    try:
        from pysrc.ops.observability import get_strategy, get_tenant

        return {"tenant_id": get_tenant(), "strategy_id": get_strategy()}
    except Exception:
        return {}


def _enrich(details: dict[str, Any] | None) -> dict[str, Any]:
    # build a fresh, serializable details map
    d: dict[str, Any] = dict(details) if details else {}
    d.update(_get_tenant_context())
    d.update(_get_trace_context())
    # drop callables and non-serializables to protect logs/metrics backends
    for k, v in list(d.items()):
        try:
            repr(v)  # best-effort viability check
        except Exception:
            d[k] = "<non-serializable>"
    return d


class BaseError(RuntimeError):
    # lean footprint and deterministic fields for logging/metrics
    __slots__ = ("msg", "code", "details", "__cause__")
    _metrics_counter: ClassVar[Any] = None

    def __init__(
        self,
        msg: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ):
        resolved = self._resolve_code(code)
        super().__init__(msg)
        self.msg = msg
        self.code = resolved
        self.details = _enrich(details)
        self.__cause__ = cause

        if _ENABLE_METRICS:
            try:
                m = _get_metrics()
            except Exception:
                m = None

            if m is not None:
                try:
                    if BaseError._metrics_counter is None:
                        # lazily allocate the shared counter once
                        BaseError._metrics_counter = m.counter(
                            "exceptions_total",
                            description="Exceptions observed",
                        )
                    m.record_counter(
                        BaseError._metrics_counter,
                        1,
                        {"code": self.code, "class": self.__class__.__name__},
                    )
                except Exception:
                    # metrics are strictly best-effort and must never break exceptions
                    BaseError._metrics_counter = None

        if _ENABLE_LOGGING and _rate_limiter.should_log(self.code):
            _get_logger().warning(
                "exception_created",
                code=self.code,
                message=self.msg,
                details=self.details,
                cause=str(cause) if cause else None,
            )

    # allow subclasses to feed semantic codes early in __init__
    def _resolve_code(self, proposed: str | None) -> str:
        return proposed or self.__class__.__name__

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.msg, "details": self.details}

    def __str__(self) -> str:
        return f"{self.code}: {self.msg}" + (f" | {self.details}" if self.details else "")


class UnsupportedPlan(BaseError):
    def __init__(self, msg: str = "Unsupported execution plan", **kwargs: Any):
        super().__init__(msg, code=ErrorCodeEnum.UNSUPPORTED_PLAN.value, **kwargs)


class DataError(BaseError, ABC):
    # ensure semantic codes flow into BaseError init
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code=self.get_error_code(), details=details)

    @abstractmethod
    def get_error_code(self) -> str: ...  # subclasses provide stable codes


class GenericDataError(DataError):
    def __init__(self, message: str = "An error occurred", details: dict[str, Any] | None = None):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.GENERIC.value


class ExceptionRegistry:
    _exceptions: ClassVar[dict[str, type[DataError]]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _instantiation_counter: ClassVar[Any] = None

    @classmethod
    def register(cls, error_type: str, exception_class: type[DataError]) -> None:
        with cls._lock:
            # last-write-wins to keep API simple
            cls._exceptions[error_type] = exception_class
            _get_logger().debug(
                "exception_registry.registered",
                error_type=error_type,
                class_name=exception_class.__name__,
            )

    @classmethod
    def get_exception(
        cls, error_type: str, message: str, details: dict[str, Any] | None = None
    ) -> DataError:
        with cls._lock:
            exc_cls = cls._exceptions.get(error_type, GenericDataError)
            if _ENABLE_METRICS:
                m = _get_metrics()
                if cls._instantiation_counter is None:
                    cls._instantiation_counter = m.counter(
                        "exception_registry_instantiations_total",
                        description="Instantiations via registry",
                    )
                m.record_counter(cls._instantiation_counter, 1, {"error_type": error_type})
            return exc_cls(message, details)


class AggregateError(DataError):
    def __init__(
        self, message: str = "Multiple errors occurred", errors: list[DataError] | None = None
    ):
        super().__init__(message, details={"errors": [str(e) for e in (errors or [])]})
        self.errors = errors or []

    def get_error_code(self) -> str:
        return "AGGREGATE_ERROR"

    def add_error(self, error: DataError) -> None:
        self.errors.append(error)
        self.details["errors"].append(str(error))


class DataFetchError(DataError):
    def __init__(self, message: str = "Data fetch failed", details: dict[str, Any] | None = None):
        if details and "source" in details:
            message = f"{message}: Source '{details['source']}'"
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.DATA_FETCH.value


class APIConnectionError(DataFetchError):
    def __init__(
        self, message: str = "Failed to connect to API", details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.API_CONNECTION.value


class StreamConnectionError(DataFetchError):
    def __init__(
        self,
        message: str = "Failed to connect to streaming source",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.STREAM_CONNECTION.value


class DataTimeoutError(DataFetchError):
    def __init__(
        self, message: str = "Data fetch timed out", details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.DATA_TIMEOUT.value


class DataValidationError(DataError):
    class Code(StrEnum):
        MISSING_COLUMNS = "MISSING_COLUMNS"
        INVALID_TYPE = "INVALID_TYPE"
        EMPTY_DATA = "EMPTY_DATA"
        SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
        FILE_FORMAT = "FILE_FORMAT"

    def __init__(
        self,
        message: str = "Data validation failed",
        codes: list[DataValidationError.Code] | None = None,
        details: dict[str, Any] | None = None,
    ):
        # Set codes BEFORE calling super().__init__() since get_error_code() needs it
        self.codes = codes or []
        if details and "field" in details:
            message = f"{message}: Invalid field '{details['field']}'"
        super().__init__(message, details)

    def get_error_code(self) -> str:
        if not self.codes:
            return ErrorCodeEnum.DATA_VALIDATION.value
        return ",".join([c.value for c in self.codes])


class SchemaValidationError(DataValidationError):
    def __init__(
        self, message: str = "Schema validation failed", details: dict[str, Any] | None = None
    ):
        super().__init__(message, codes=[DataValidationError.Code.SCHEMA_MISMATCH], details=details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.SCHEMA_VALIDATION.value


class FileFormatError(DataValidationError):
    def __init__(
        self,
        message: str = "Invalid/unsupported file format",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, codes=[DataValidationError.Code.FILE_FORMAT], details=details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.FILE_FORMAT.value


class PITViolationError(DataError):
    """Raised when point-in-time constraints are violated (missing temporal columns, empty frame, corrupted store)."""

    def __init__(
        self, message: str = "Point-in-time violation", details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.PIT_VIOLATION.value


class StalenessError(PITViolationError):
    """Raised when a resolved field is stale or missing and that field's MissingPolicy is FAIL."""

    def __init__(
        self, message: str = "Stale or missing field", details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.STALENESS.value


class StatisticalTestError(DataError):
    def __init__(
        self, message: str = "Statistical test failed", details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.STATISTICAL_TEST.value


class DataDriftError(StatisticalTestError):
    def __init__(self, message: str = "Data drift detected", details: dict[str, Any] | None = None):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.DATA_DRIFT.value


class IBKRConnectionError(StreamConnectionError):
    def __init__(
        self,
        message: str = "Failed to connect to Interactive Brokers",
        details: dict[str, Any] | None = None,
    ):
        warnings.warn(
            "IBKRConnectionError is deprecated; use StreamConnectionError",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(message, details)

    def get_error_code(self) -> str:
        # preserved for compatibility; prefer StreamConnectionError
        return "IBKR_CONNECTION_ERROR"


class NoDataError(DataFetchError):
    def __init__(self, symbol: str, details: dict[str, Any] | None = None):
        message = f"No historical data returned for {symbol}"
        details = details or {"symbol": symbol}
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.NO_DATA.value


class ConfigValidationError(DataError):
    def __init__(
        self,
        message: str = "Configuration validation failed",
        validation_errors: list[str] | None = None,
    ):
        super().__init__(message, details={"validation_errors": validation_errors or []})

    def get_error_code(self) -> str:
        return ErrorCodeEnum.CONFIG_VALIDATION.value


class PreprocessingError(DataError):
    def __init__(
        self, message: str = "Data preprocessing failed", details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.PREPROCESSING.value


class ModelTrainingError(DataError):
    def __init__(
        self, message: str = "Model training failed", details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.MODEL_TRAINING.value


class TradingExecutionError(DataError):
    def __init__(
        self, message: str = "Trading execution failed", details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.TRADING_EXECUTION.value


class InvalidInputError(DataError):
    def __init__(self, message: str = "Invalid input data", details: dict[str, Any] | None = None):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.INVALID_INPUT.value


class DataPreconditionError(DataError):
    """Input buffers or indices violate a documented contract (fail-closed)."""

    def __init__(
        self, message: str = "Data precondition violated", details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.DATA_PRECONDITION.value


class ModelCheckpointError(DataError):
    def __init__(
        self, message: str = "Model checkpointing failed", details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.MODEL_CHECKPOINT.value


class ModelInferenceError(DataError):
    def __init__(
        self, message: str = "Model inference failed", details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)

    def get_error_code(self) -> str:
        return ErrorCodeEnum.MODEL_INFERENCE.value


# Known types
ExceptionRegistry.register(ErrorCodeEnum.DATA_DRIFT.value, DataDriftError)
ExceptionRegistry.register(ErrorCodeEnum.DATA_FETCH.value, DataFetchError)
ExceptionRegistry.register(ErrorCodeEnum.API_CONNECTION.value, APIConnectionError)
ExceptionRegistry.register(ErrorCodeEnum.STREAM_CONNECTION.value, StreamConnectionError)
ExceptionRegistry.register(ErrorCodeEnum.DATA_VALIDATION.value, DataValidationError)
ExceptionRegistry.register(ErrorCodeEnum.SCHEMA_VALIDATION.value, SchemaValidationError)
ExceptionRegistry.register(ErrorCodeEnum.FILE_FORMAT.value, FileFormatError)
ExceptionRegistry.register(ErrorCodeEnum.STATISTICAL_TEST.value, StatisticalTestError)
ExceptionRegistry.register(ErrorCodeEnum.NO_DATA.value, NoDataError)
ExceptionRegistry.register(ErrorCodeEnum.CONFIG_VALIDATION.value, ConfigValidationError)
ExceptionRegistry.register(ErrorCodeEnum.PREPROCESSING.value, PreprocessingError)
ExceptionRegistry.register(ErrorCodeEnum.MODEL_TRAINING.value, ModelTrainingError)
ExceptionRegistry.register(ErrorCodeEnum.TRADING_EXECUTION.value, TradingExecutionError)
ExceptionRegistry.register(ErrorCodeEnum.INVALID_INPUT.value, InvalidInputError)
ExceptionRegistry.register(ErrorCodeEnum.DATA_PRECONDITION.value, DataPreconditionError)
ExceptionRegistry.register(ErrorCodeEnum.MODEL_CHECKPOINT.value, ModelCheckpointError)
ExceptionRegistry.register(ErrorCodeEnum.MODEL_INFERENCE.value, ModelInferenceError)
ExceptionRegistry.register(ErrorCodeEnum.PIT_VIOLATION.value, PITViolationError)
ExceptionRegistry.register(ErrorCodeEnum.STALENESS.value, StalenessError)
