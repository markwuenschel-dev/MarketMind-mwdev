"""Retry utilities: exponential backoff with jitter for transient failures."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class MaxRetriesExceededError(RuntimeError):
    """Raised when all retry attempts are exhausted."""


def _jitter(base: float, seed: int) -> float:
    """Return base * (0.75 + 0.5 * frac) using a lightweight deterministic jitter."""
    frac = ((seed * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF
    return base * (0.75 + 0.5 * frac)


class RetryPolicy:
    """Exponential-backoff retry configuration."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        eta: float = 2.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.eta = eta


def with_retries[T](
    fn: Callable[[], T],
    policy: RetryPolicy,
    retryable: type[Exception] = Exception,
) -> T:
    """Call fn, retrying on *retryable* exceptions per policy; re-raises on exhaustion."""
    backoff = policy.initial_backoff
    for attempt in range(policy.max_attempts):
        try:
            return fn()
        except retryable as exc:
            if attempt == policy.max_attempts - 1:
                raise MaxRetriesExceededError(
                    f"All {policy.max_attempts} attempts exhausted. Last error: {exc}"
                ) from exc
            wait = min(_jitter(backoff, attempt), policy.max_backoff)
            time.sleep(wait)
            backoff *= policy.eta
    raise MaxRetriesExceededError("Retry loop exited without result")  # unreachable


__all__ = ["RetryPolicy", "MaxRetriesExceededError", "with_retries"]
