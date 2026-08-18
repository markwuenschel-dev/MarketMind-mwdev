"""MLC-0 · Signal ABC with ``signal_embedding`` field.

Canonical path resolution (MLC-0 Step 1)
----------------------------------------

``MetaLearningCore.md`` §5.5 specifies ``py/registry/signal_abc.py``.
The in-repo Python root is ``pysrc/``; canonical path is therefore
``pysrc/registry/signal_abc.py``.

What this module adds
---------------------

Introduces :class:`SignalABC`, an abstract base class that formalizes
the in-repo signal interface and declares the
``signal_embedding: Optional[np.ndarray] = None`` field required by the
MLC-0 brief.

Relation to the existing :class:`pysrc.registry.signal_catalog.SignalProtocol`
is intentionally loose: the Protocol remains the structural type used by
:class:`pysrc.registry.signal_catalog.SignalCatalog` for registration,
and :class:`SignalABC` is the nominal base for implementers that want
explicit ABC inheritance.  :class:`SignalABC` is a ``SignalProtocol`` by
construction (it defines the same attributes), so both paths remain
compatible.

Phase II MLC-1 will populate ``signal_embedding`` once
``context_encoder.py`` is online.  Until then the field must remain
``None`` on every concrete signal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SignalABC",
]


class SignalABC(ABC):
    """Abstract base class for a MarketMind signal.

    Attributes
    ----------
    signal_embedding
        Optional dense embedding vector emitted by the Phase II context
        encoder.  **Must be ``None`` until MLC-1** populates it once
        ``context_encoder.py`` is online (see MLC-0 brief §2, §5 Step 6).
        Concrete subclasses that do not (yet) produce embeddings should
        leave the default in place.

    Subclasses must implement :meth:`__call__` returning a signal series
    (typically a ``polars.Series`` when polars is available, otherwise a
    framework-native series object) and must expose a stable
    :attr:`slot_index` assigned by
    :class:`pysrc.registry.signal_catalog.SignalCatalog` at registration.
    """

    # Phase II MLC-1 will populate this field once context_encoder.py is online.
    signal_embedding: NDArray[np.floating[Any]] | None = None

    @property
    @abstractmethod
    def slot_index(self) -> int:
        """Stable slot assigned at registration by :class:`SignalCatalog`."""

    @abstractmethod
    def __call__(self, data: Any, params: dict[str, Any]) -> Any:
        """Compute the signal series from ``data`` and ``params``."""
