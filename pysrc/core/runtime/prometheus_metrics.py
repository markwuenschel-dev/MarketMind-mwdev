from __future__ import annotations

from pysrc.core.runtime.optional_imports import optional_import


class _NoOpTimer:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _NoOpMetric:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def labels(self, *args: object, **kwargs: object) -> _NoOpMetric:
        return self

    def inc(self, amount: float = 1.0) -> None:
        _ = amount

    def observe(self, value: float) -> None:
        _ = value

    def time(self) -> _NoOpTimer:
        return _NoOpTimer()


_prometheus_client = optional_import("prometheus_client")
Counter = getattr(_prometheus_client, "Counter", _NoOpMetric)
Histogram = getattr(_prometheus_client, "Histogram", _NoOpMetric)

__all__ = ["Counter", "Histogram"]
