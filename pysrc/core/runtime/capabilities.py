from __future__ import annotations

CAPABILITIES: dict[str, list[object]] = {
    "dataframe": ["pandas"],
    "classifier": [],
    "ml_engine": [],
    "array": [],
    "tensor": [],
}

__all__ = ["CAPABILITIES"]
