"""PIT-safe dataset view definitions — pure boundary objects, no data loading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = ["DatasetView"]


@dataclass(frozen=True)
class DatasetView:
    """Declares the PIT-safe view of a dataset: symbol + as_of timestamp."""

    symbol: str
    as_of: datetime
    feature_hash: str

    def validate(self) -> None:
        """Raise if the view is degenerate (missing symbol or empty feature_hash)."""
        if not self.symbol:
            raise ValueError("DatasetView.symbol must be non-empty")
        if not self.feature_hash:
            raise ValueError("DatasetView.feature_hash must be non-empty")
