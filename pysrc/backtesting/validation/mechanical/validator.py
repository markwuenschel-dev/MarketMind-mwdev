from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pysrc.backtesting.contracts.registry import register_validator
from pysrc.backtesting.contracts.types import ValidationReport, ValidationStatus
from pysrc.backtesting.validation.mechanical.properties.invariants import (
    InvariantViolation,
    assert_fill_timestamps_within_window,
)


class MechanicalValidator:
    def validate(self, result, ctx: dict[str, Any], store) -> ValidationReport:
        fills = [
            {
                "symbol": fill.symbol,
                "quantity": fill.quantity,
                "price": fill.price,
                "side": fill.side,
                "timestamp": fill.timestamp,
            }
            for fill in result.fills
        ]
        window_start = ctx.get("window_start", datetime(1970, 1, 1, tzinfo=UTC))
        window_end = ctx.get("window_end", datetime.now(UTC))
        status = ValidationStatus.PASS
        message = "Mechanical invariants passed"
        if fills:
            try:
                assert_fill_timestamps_within_window(fills, window_start, window_end)
            except InvariantViolation as exc:
                status = ValidationStatus.FAIL
                message = str(exc)
        payload = {
            "schema_version": "1.0.0",
            "status": status.value,
            "message": message,
            "fill_count": len(fills),
        }
        ref = store.put_json("invariants_report.json", payload)
        return ValidationReport(
            status=status,
            reason_code=f"MECH_{status.value}",
            message=message,
            artifacts={"invariants_report.json": ref},
        )


register_validator("mechanical.v1", lambda: MechanicalValidator())
