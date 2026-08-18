# py/infra/brokers/pipeline_config.py
"""Parses broker-specific configs from pipeline_config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# v1/v2 compatibility for validators
try:
    from pydantic import field_validator  # pydantic v2

    def validator(*fields: str, pre: bool = False, **kwargs: Any):
        if pre:
            kwargs["mode"] = "before"
        return field_validator(*fields, **kwargs)
except ImportError:  # v1
    from pydantic import validator  # type: ignore

from pysrc.core.errors import ConfigValidationError, DataValidationError
from pysrc.core.validation import validate_date  # example extra validation
from pysrc.ops.mm_logkit import get_logger

logger = get_logger(__name__)


class BrokerConfig(BaseModel):
    """Pydantic model for broker configuration validation."""

    host: str = Field(default="localhost", description="Broker API host.")
    port: int = Field(default=7497, description="Broker API port.")
    client_id: int = Field(default=1, description="Client ID for connection.")
    account: str | None = Field(default=None, description="Account identifier.")
    timeout: float = Field(default=10.0, description="Request timeout in seconds.")
    retries: int = Field(default=3, description="Retry attempts for API calls.")
    # Extensible: Add broker-specific fields here (tokens, endpoints, etc.)

    @validator("port")
    def validate_port(self, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535.")
        return v

    @validator("account", pre=True)
    def validate_account(self, v: str | None) -> str | None:
        if v:
            from pysrc.core.validation import validate_symbol  # reuse your validator

            try:
                validate_symbol(v)
            except DataValidationError as e:
                raise ValueError(f"Invalid account: {e}") from e
        return v


def load_broker_config(
    config_path: str = "pipeline_config.yaml", section: str = "ibkr"
) -> BrokerConfig:
    """
    Load and validate a broker pipeline_config from YAML.
    - Abstract/Dynamic: sectioned YAML with Pydantic validation.
    - Extensible: drop-in new sections/fields.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r") as f:
        config_data: dict[str, Any] = yaml.safe_load(f) or {}

    if section not in config_data:
        raise ConfigValidationError(f"Section '{section}' not found in pipeline_config.")

    data = config_data[section]
    try:
        cfg = BrokerConfig(**data)
        # Optional cross-field/date validation example
        if "start_date" in data:
            validate_date(data["start_date"])
        return cfg
    except Exception as e:
        logger.error("Config validation failed", exc_info=True)
        raise ConfigValidationError(str(e)) from e
