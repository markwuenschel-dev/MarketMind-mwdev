from __future__ import annotations

import math
from typing import Any


class SanitizationError(ValueError):
    """Raised when an artifact payload cannot be safely serialized."""


_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


def sanitize_json_payload(payload: Any) -> Any:
    """Recursively sanitize a payload for JSON artifact emission."""
    if isinstance(payload, dict):
        return {str(key): sanitize_json_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [sanitize_json_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return [sanitize_json_payload(item) for item in payload]
    if isinstance(payload, float):
        if math.isnan(payload) or math.isinf(payload):
            raise SanitizationError(
                "NaN and Inf are not allowed in artifact-registry JSON payloads."
            )
        return payload
    if isinstance(payload, _JSON_SCALAR_TYPES):
        return payload
    raise SanitizationError(f"Unsupported artifact payload type: {type(payload).__name__}.")
