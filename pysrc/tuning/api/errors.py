"""Typed API-layer errors and mapping from domain errors to API errors.

Provides a thin translation layer so domain exceptions never leak through the HTTP boundary.
"""

from __future__ import annotations

__all__ = [
    "APIError",
    "JobNotFoundError",
    "InvalidRequestError",
    "GateFailedError",
    "map_domain_error",
]


class APIError(Exception):
    """Base API error; carries an HTTP status code and human-readable detail."""

    status_code: int = 500
    detail: str

    def __init__(self, detail: str, status_code: int = 500) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class JobNotFoundError(APIError):
    """Raised when the requested tuning job does not exist."""

    status_code = 404

    def __init__(self, detail: str) -> None:
        super().__init__(detail, status_code=404)


class InvalidRequestError(APIError):
    """Raised when the incoming request fails schema or semantic validation."""

    status_code = 422

    def __init__(self, detail: str) -> None:
        super().__init__(detail, status_code=422)


class GateFailedError(APIError):
    """Raised when a promotion candidate does not pass the statistical gate."""

    status_code = 409

    def __init__(self, detail: str) -> None:
        super().__init__(detail, status_code=409)


def map_domain_error(exc: Exception) -> APIError:
    """Map a domain TuningError to the appropriate APIError subclass.

    Falls back to a generic 500 APIError for unrecognised domain exceptions.
    """
    # Avoid a hard import of the domain layer; use string matching on the type
    # name so the API layer remains decoupled from pysrc.tuning.core.
    type_name = type(exc).__name__
    detail = str(exc)

    if "NotFound" in type_name:
        return JobNotFoundError(detail)
    if "InvalidRequest" in type_name or "ValueError" in type_name:
        return InvalidRequestError(detail)
    if "GateFailed" in type_name:
        return GateFailedError(detail)
    return APIError(detail)
