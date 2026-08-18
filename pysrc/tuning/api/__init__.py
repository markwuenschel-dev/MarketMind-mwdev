"""Public entry-point for the tuning HTTP API layer.

Re-exports boundary schemas, handler callables, typed errors, and the optional FastAPI app.
"""

from __future__ import annotations

from .errors import (
    APIError,
    GateFailedError,
    InvalidRequestError,
    JobNotFoundError,
    map_domain_error,
)
from .handlers import get_job_status, submit_promotion, submit_tuning_job
from .schemas import (
    GateResultResponse,
    PromotionRequest,
    PromotionResponse,
    SearchStatusResponse,
    TuningJobRequest,
    TuningJobResponse,
)
from .server import app

__all__ = [
    # Schemas
    "TuningJobRequest",
    "TuningJobResponse",
    "PromotionRequest",
    "PromotionResponse",
    "SearchStatusResponse",
    "GateResultResponse",
    # Handlers
    "submit_tuning_job",
    "get_job_status",
    "submit_promotion",
    # Errors
    "APIError",
    "JobNotFoundError",
    "InvalidRequestError",
    "GateFailedError",
    "map_domain_error",
    # Server (may be None if FastAPI is absent)
    "app",
]
