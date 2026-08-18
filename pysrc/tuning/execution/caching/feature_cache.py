"""FeatureCache: keyed cache for pre-computed feature matrices."""

from __future__ import annotations

from typing import Any


class FeatureCache:
    """Cache for feature matrices keyed by (feature_hash, symbol, as_of)."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def _key(self, feature_hash: str, symbol: str, as_of: str) -> str:
        return f"{feature_hash}|{symbol}|{as_of}"

    def get(self, feature_hash: str, symbol: str, as_of: str) -> Any | None:
        return self._store.get(self._key(feature_hash, symbol, as_of))

    def put(self, feature_hash: str, symbol: str, as_of: str, data: Any) -> None:
        self._store[self._key(feature_hash, symbol, as_of)] = data


__all__ = ["FeatureCache"]
