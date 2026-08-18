"""Fixed-shape masks, slot masks, and active-k masks for candidate selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["SlotMask"]


@dataclass(frozen=True)
class SlotMask:
    """Boolean mask indicating which slots are active in a fixed-shape ensemble."""

    mask: NDArray[np.bool_]
    n_active: int

    @classmethod
    def from_k(cls, total_slots: int, k: int) -> SlotMask:
        """Create a mask with the first k slots active."""
        if k > total_slots:
            raise ValueError(f"k={k} exceeds total_slots={total_slots}")
        arr = np.zeros(total_slots, dtype=np.bool_)
        arr[:k] = True
        return cls(mask=arr, n_active=k)

    def active_indices(self) -> NDArray[np.intp]:
        """Return indices of active slots."""
        return np.nonzero(self.mask)[0]
