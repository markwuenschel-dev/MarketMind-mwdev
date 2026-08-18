"""Order book data structures and operations.

Provides typed representations of L2/L3 order book state
with efficient update handling for market data ingestion.

TODO: Registry hook for order book implementations.
TODO: Integration with DataView for point-in-time snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class PriceLevel:
    """Immutable price level representation.

    Attributes:
        price: Price level value
        size: Aggregated size at level
        count: Number of orders at level
    """

    price: Decimal
    size: Decimal
    count: int


class OrderBook:
    """Abstract order book interface.

    Implementations provide efficient L2/L3 order book
    maintenance with deterministic snapshot capabilities.

    TODO: ABC with concrete implementations per exchange.
    """

    pass
