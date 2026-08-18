# --- MARKET-SOURCES-REGISTRY-SHIM ---
from typing import Any

_REGISTRY: dict[str, Any] = {}


def register_source(name: str):
    """Decorator to register a data source."""

    def decorator(src: Any):
        _REGISTRY[name] = src
        return src

    return decorator


def get_registry() -> dict[str, Any]:
    # shallow copy to reduce accidental mutation in tests
    return dict(_REGISTRY)


class DataSource:
    """Minimal protocol/class for smoke imports."""

    async def get_historical(self, *args, **kwargs):
        raise NotImplementedError


__all__ = ["register_source", "get_registry", "DataSource"]
# NOTE: do NOT auto-import provider modules here.
# --- /MARKET-SOURCES-REGISTRY-SHIM ---
