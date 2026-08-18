"""Typed schema validation errors for config and IR objects."""

from __future__ import annotations


class SchemaError(ValueError):
    """Base class for all schema validation errors in the tuning core."""


class ConfigSchemaError(SchemaError):
    """Raised when a config dict fails schema validation."""


class IRSchemaError(SchemaError):
    """Raised when an IR object contains malformed fields."""


__all__ = ["SchemaError", "ConfigSchemaError", "IRSchemaError"]
