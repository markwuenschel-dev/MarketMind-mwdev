"""Optional FastAPI shell; gracefully absent if FastAPI is not installed.

Layer rule: Shell only — registers routes, delegates to handlers, contains no business logic.
"""

from __future__ import annotations

from typing import Any

from .errors import map_domain_error
from .handlers import get_job_status, submit_promotion, submit_tuning_job
from .schemas import (
    PromotionRequest,
    PromotionResponse,
    SearchStatusResponse,
    TuningJobRequest,
    TuningJobResponse,
)

__all__ = ["app"]

try:
    from fastapi import FastAPI

    app: Any = FastAPI(title="MarketMind Tuning API", version="0.1.0")

    @app.post("/jobs", response_model=TuningJobResponse)  # type: ignore[untyped-decorator]
    def _post_jobs(req: TuningJobRequest) -> TuningJobResponse:
        """Submit a new tuning job."""
        try:
            return submit_tuning_job(req)
        except Exception as exc:
            api_err = map_domain_error(exc)
            # FastAPI requires HTTPException; import lazily to avoid hard coupling.
            from fastapi import HTTPException  # noqa: PLC0415

            raise HTTPException(status_code=api_err.status_code, detail=api_err.detail) from exc

    @app.get("/jobs/{job_id}", response_model=SearchStatusResponse)  # type: ignore[untyped-decorator]
    def _get_job(job_id: str) -> SearchStatusResponse:
        """Retrieve current status for a tuning job."""
        try:
            return get_job_status(job_id)
        except Exception as exc:
            api_err = map_domain_error(exc)
            from fastapi import HTTPException  # noqa: PLC0415

            raise HTTPException(status_code=api_err.status_code, detail=api_err.detail) from exc

    @app.post("/promotions", response_model=PromotionResponse)  # type: ignore[untyped-decorator]
    def _post_promotions(req: PromotionRequest) -> PromotionResponse:
        """Submit a promotion request for a tuning candidate."""
        try:
            return submit_promotion(req)
        except Exception as exc:
            api_err = map_domain_error(exc)
            from fastapi import HTTPException  # noqa: PLC0415

            raise HTTPException(status_code=api_err.status_code, detail=api_err.detail) from exc

except ImportError:
    # FastAPI is an optional dependency; the tuning system functions without it.
    app = None
