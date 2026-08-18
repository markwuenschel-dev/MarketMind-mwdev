# Adapter so MultiTierClient satisfies the dataprep orchestrator cache protocol
# (exists, save_npz, load_npz, save_json, load_json, save_df, load_df).
from __future__ import annotations

import contextlib
from typing import Any

from pysrc.ops.multi_tier_cache import MultiTierClient


class MultiTierCacheAdapter:
    """Wraps MultiTierClient to expose the orchestrator cache protocol."""

    def __init__(self, client: MultiTierClient):
        self._client = client

    def exists(self, key: str) -> bool:
        return self._client.get(key) is not None or (
            getattr(self._client, "l4", None) is not None and self._client.l4.exists(key)
        )

    def save_npz(self, key: str, data: Any) -> None:
        with contextlib.suppress(ValueError, TypeError, AttributeError):
            self._client.set(key, data, write_through=True)

    def load_npz(self, key: str) -> Any | None:
        try:
            return self._client.get(key)
        except (ValueError, TypeError, AttributeError):
            return None

    def save_json(self, key: str, data: Any) -> None:
        with contextlib.suppress(ValueError, TypeError, AttributeError):
            self._client.set(key, data, write_through=True)

    def load_json(self, key: str) -> Any | None:
        try:
            return self._client.get(key)
        except (ValueError, TypeError, AttributeError):
            return None

    def save_df(self, key: str, df: Any, **kwargs) -> None:
        try:
            version = kwargs.get("version", "v1")
            if getattr(self._client, "l4", None) is not None:
                self._client.l4.save_df(key, df, version=version)
        except (FileNotFoundError, PermissionError, OSError, ValueError, TypeError):
            pass

    def load_df(self, key: str, **kwargs) -> Any | None:
        try:
            expected_version = kwargs.get("expected_version")
            if getattr(self._client, "l4", None) is not None:
                return self._client.l4.load_df(key, expected_version=expected_version)
            return self._client.get(key)
        except (FileNotFoundError, PermissionError, OSError, ValueError, TypeError):
            return None
