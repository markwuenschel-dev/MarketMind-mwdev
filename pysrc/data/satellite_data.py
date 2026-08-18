"""Satellite and alternative data integration.

Provides typed contracts for non-traditional data sources
including satellite imagery, social sentiment, and IoT feeds.

TODO: Registry hook for alternative data providers.
TODO: Schema validation for satellite data formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


@dataclass(frozen=True, slots=True)
class SatelliteDataSpec:
    """Satellite data request specification.

    Attributes:
        provider: Data provider identifier
        region: Geographic region code
        date_range: Requested date interval
        resolution: Spatial resolution in meters
    """

    provider: str
    region: str
    date_range: tuple[str, str]
    resolution: int


class SatelliteDataProvider(ABC):
    """Abstract satellite data provider.

    Implementations fetch and normalize alternative data
    with consistent schema and point-in-time guarantees.

    TODO: Factory integration for provider instantiation.
    """

    @abstractmethod
    def fetch(self, spec: SatelliteDataSpec) -> Mapping[str, Any]:
        """Fetch satellite data for specification.

        Args:
            spec: Data request specification

        Returns:
            Normalized data payload
        """
        ...
