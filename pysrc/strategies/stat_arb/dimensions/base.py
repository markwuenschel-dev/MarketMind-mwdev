from __future__ import annotations

from abc import ABC, abstractmethod


class DimensionProfile(ABC):
    """Future dimension-profile contract. Live Phase I-D runtime must not depend on this."""

    @abstractmethod
    def dimension_name(self) -> str:
        """Return the future profile name."""
