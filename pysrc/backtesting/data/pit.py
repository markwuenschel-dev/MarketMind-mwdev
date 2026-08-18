from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pysrc.backtesting.contracts.protocols import AsOfView
from pysrc.backtesting.contracts.types import MarketSlice, PitMeta


@dataclass(frozen=True)
class PITSafeDataView:
    view: AsOfView
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_of(self, ts: datetime) -> MarketSlice:
        return self.view.as_of(ts)

    def pit_meta(self) -> PitMeta | None:
        return self.view.pit_meta()


@dataclass(frozen=True)
class PitUnsafeFrame:
    payload_ref: str
    metadata: dict[str, Any] = field(default_factory=dict)
