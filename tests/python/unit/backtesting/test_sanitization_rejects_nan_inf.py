from __future__ import annotations

import math

import pytest

from pysrc.artifact_registry.sanitization import SanitizationError, sanitize_json_payload


@pytest.mark.determinism("d1")
def test_sanitization_rejects_nan_and_inf() -> None:
    with pytest.raises(SanitizationError):
        sanitize_json_payload({"bad": math.nan})
    with pytest.raises(SanitizationError):
        sanitize_json_payload({"bad": math.inf})
