"""Signal catalog with stable slot_index at registration (Phase I-E).

Signal ABC: __call__(data, params) -> pl.Series; signal_embedding placeholder;
slot_index assigned at registration, immutable. SignalCatalog maintains monotonic
slot assignment; register() is idempotent by spec_hash (re-registration returns
existing slot_index). No slot reuse in Phase I.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, runtime_checkable

import numpy as np

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.dynamic_k_contract import MAX_SIGNALS

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


def _compute_spec_hash(signal_name: str, module_path: str = "") -> str:
    """Deterministic spec_hash for dedup; Phase I uses name + module."""
    payload = f"{signal_name}:{module_path}"
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


@runtime_checkable
class SignalProtocol(Protocol):
    """Protocol for signal callable and metadata. slot_index set at registration."""

    @property
    def signal_embedding(self) -> np.ndarray[Any, np.dtype[np.floating]] | None:
        """Placeholder until Phase IV; must be None in Phase I."""
        ...

    @property
    def slot_index(self) -> int:
        """Stable slot assigned at registration; immutable."""
        ...

    def __call__(self, data: Any, params: dict[str, Any]) -> Any:
        """Compute signal series from data and params. Returns pl.Series when polars available."""
        ...


class RegisteredSignal:
    """Wrapper holding a signal implementation and its registration metadata."""

    __slots__ = ("_impl", "_slot_index", "_spec_hash", "_signal_name")

    def __init__(
        self,
        impl: SignalProtocol,
        slot_index: int,
        spec_hash: str,
        signal_name: str,
    ) -> None:
        self._impl = impl
        self._slot_index = slot_index
        self._spec_hash = spec_hash
        self._signal_name = signal_name

    @property
    def signal_embedding(self) -> np.ndarray[Any, np.dtype[np.floating]] | None:
        return getattr(self._impl, "signal_embedding", None)

    @property
    def slot_index(self) -> int:
        return self._slot_index

    @property
    def spec_hash(self) -> str:
        return self._spec_hash

    @property
    def signal_name(self) -> str:
        return self._signal_name

    def __call__(self, data: Any, params: dict[str, Any]) -> Any:
        return self._impl(data, params)


class SignalCatalog:
    """Catalog of signals with monotonic slot assignment. Idempotent register by spec_hash."""

    def __init__(self) -> None:
        self._next_slot = 0
        self._by_spec_hash: dict[str, RegisteredSignal] = {}
        self._by_slot: dict[int, RegisteredSignal] = {}

    def register(
        self,
        signal: SignalProtocol,
        spec_hash: str | None = None,
        signal_name: str = "",
    ) -> int:
        """Register a signal; assign slot_index. Idempotent: same spec_hash returns existing slot."""
        name = (
            signal_name
            or getattr(signal, "signal_name", "")
            or getattr(signal, "__name__", "unknown")
        )
        sh = spec_hash if spec_hash is not None else _compute_spec_hash(name)
        if sh in self._by_spec_hash:
            return self._by_spec_hash[sh].slot_index
        if self._next_slot >= MAX_SIGNALS:
            raise DataPreconditionError(
                "SignalCatalog slot_index would exceed MAX_SIGNALS (MLN-04 Dynamic-K); "
                "reclamation is an explicit governed event, not silent reuse",
                details={"next_slot": self._next_slot, "max_signals": MAX_SIGNALS},
            )
        slot = self._next_slot
        self._next_slot += 1
        wrapped = RegisteredSignal(impl=signal, slot_index=slot, spec_hash=sh, signal_name=name)
        self._by_spec_hash[sh] = wrapped
        self._by_slot[slot] = wrapped
        return slot

    def get_by_spec_hash(self, spec_hash: str) -> RegisteredSignal | None:
        return self._by_spec_hash.get(spec_hash)

    def get_by_slot(self, slot_index: int) -> RegisteredSignal | None:
        return self._by_slot.get(slot_index)

    def __len__(self) -> int:
        return len(self._by_spec_hash)


# ---------------------------------------------------------------------------
# Five starter signals (slots 0-4): stubs until full implementations exist
# ---------------------------------------------------------------------------


class _StubSignal:
    """Minimal signal stub; slot_index set by catalog at registration."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._slot_index = -1  # Replaced by RegisteredSignal after register

    @property
    def signal_embedding(self) -> np.ndarray[Any, np.dtype[np.floating]] | None:
        return None

    @property
    def slot_index(self) -> int:
        return self._slot_index

    def __call__(self, data: Any, params: dict[str, Any]) -> Any:
        if pl is not None and hasattr(data, "height"):
            return pl.Series(self._name, [0.0] * data.height)
        try:
            import pandas as pd  # type: ignore[import-untyped]

            if hasattr(data, "shape"):
                return pd.Series(0.0, index=range(data.shape[0]))
        except ImportError:
            pass
        return []


_STARTER_NAMES = [
    "stat_arb.spread_zscore",
    "stat_arb.hedge_ratio",
    "momentum.TSMOM",
    "momentum.XSMOM",
    "RSI.baseline",
]


def _create_starter(name: str) -> _StubSignal:
    """Factory for entry_points; returns a stub signal with the given name."""
    return _StubSignal(name)


# Entry-point factories for pyproject.toml [tool.poetry.plugins."marketmind.signals"]
def create_starter_spread_zscore() -> _StubSignal:
    return _create_starter("stat_arb.spread_zscore")


def create_starter_hedge_ratio() -> _StubSignal:
    return _create_starter("stat_arb.hedge_ratio")


def create_starter_TSMOM() -> _StubSignal:
    return _create_starter("momentum.TSMOM")


def create_starter_XSMOM() -> _StubSignal:
    return _create_starter("momentum.XSMOM")


def create_starter_RSI_baseline() -> _StubSignal:
    return _create_starter("RSI.baseline")


def _ensure_starters_registered(catalog: SignalCatalog) -> None:
    """Register the 5 starter signals in order (slots 0-4) if not already present."""
    for name in _STARTER_NAMES:
        spec_hash = _compute_spec_hash(name)
        if spec_hash in catalog._by_spec_hash:
            continue
        stub = _StubSignal(name)
        catalog.register(stub, spec_hash=spec_hash, signal_name=name)


# Singleton used by entry_points and callers
_default_catalog: SignalCatalog | None = None


def get_catalog() -> SignalCatalog:
    """Return the default SignalCatalog singleton; registers 5 starter signals on first use."""
    global _default_catalog
    if _default_catalog is None:
        _default_catalog = SignalCatalog()
        _ensure_starters_registered(_default_catalog)
    return _default_catalog
