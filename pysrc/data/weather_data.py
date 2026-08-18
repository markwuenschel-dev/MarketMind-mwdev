"""Weather and environmental data integration.

Provides typed contracts for meteorological data sources
including forecasts, historical observations, and climate indices.

TODO: Registry hook for weather data providers.
TODO: Integration with strategy features for weather signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


@dataclass(frozen=True, slots=True)
class WeatherDataSpec:
    """Weather data request specification.

    Attributes:
        station_id: Weather station or grid identifier
        variables: List of weather variables to fetch
        frequency: Data frequency (hourly, daily, etc.)
        date_range: Requested date interval
    """

    station_id: str
    variables: list[str]
    frequency: str
    date_range: tuple[str, str]


class WeatherDataProvider(ABC):
    """Abstract weather data provider.

    Implementations fetch and normalize meteorological data
    with consistent schema for feature engineering.

    TODO: Factory integration for provider instantiation.
    """

    @abstractmethod
    def fetch(self, spec: WeatherDataSpec) -> Mapping[str, Any]:
        """Fetch weather data for specification.

        Args:
            spec: Data request specification

        Returns:
            Normalized weather data payload
        """
        ...
