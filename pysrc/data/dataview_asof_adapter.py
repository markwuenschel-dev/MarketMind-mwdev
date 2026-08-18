from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd  # type: ignore[import-untyped]

from pysrc.backtesting.contracts.protocols import AsOfView
from pysrc.backtesting.contracts.types import MarketSlice, PitMeta
from pysrc.data.dataview import DataView

_CANONICAL_VALID = "valid_time"
_CANONICAL_KNOW = "knowledge_time"


@dataclass
class DataViewAsOfAdapter(AsOfView):
    """
    Adapter that exposes a DataView instance through the AsOfView protocol.

    Snapshot semantics are delegated to DataView.as_of(symbols, fields, knowledge_date),
    which returns one row per symbol as of the provided knowledge_date.
    """

    dataview: DataView
    symbols: list[str]
    fields: list[str]
    _last_as_of: datetime | None = field(default=None, init=False)

    def as_of(self, ts: datetime) -> MarketSlice:
        """
        Delegate to DataView.as_of using snapshot semantics and emit a MarketSlice
        whose rows live in the prices payload.
        """
        knowledge_date = ts.date()
        snapshot: pd.DataFrame = self.dataview.as_of(
            symbols=self.symbols,
            fields=self.fields,
            knowledge_date=knowledge_date,
        )
        records = snapshot.to_dict(orient="records")

        metadata = {
            "pit_enforced": True,
            "pit_front_door": "pysrc.data.dataview.DataView",
            "knowledge_date": knowledge_date.isoformat(),
        }

        pit_meta = PitMeta(
            as_of=ts.isoformat(),
            source="pysrc.data.dataview.DataView",
            knowledge_cutoff=knowledge_date.isoformat(),
        )

        self._last_as_of = ts

        return MarketSlice(
            as_of=ts.isoformat(),
            prices=records,
            features=[],
            metadata=metadata,
            pit_meta=pit_meta,
        )

    def as_wide_frame(self, knowledge_dates: list[datetime]) -> pd.DataFrame:
        """
        Return one row per knowledge date with columns for each symbol's fields
        (e.g. {leg_a}.close, {leg_b}.close), preserving PIT semantics.
        Merges one row per symbol from each PIT snapshot into one wide row per date.
        """
        rows: list[dict] = []
        for ts in knowledge_dates:
            slice_ = self.as_of(ts)
            recs = slice_.prices or []
            wide: dict = {}
            for r in recs:
                for f in self.fields:
                    if f in r and pd.notna(r.get(f)):
                        wide[f] = r[f]
                if _CANONICAL_VALID in r:
                    wide[_CANONICAL_VALID] = r[_CANONICAL_VALID]
                if _CANONICAL_KNOW in r:
                    wide[_CANONICAL_KNOW] = r[_CANONICAL_KNOW]
            if wide:
                rows.append(wide)
        if not rows:
            cols = list(self.fields) + [_CANONICAL_VALID, _CANONICAL_KNOW]
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame(rows)
        out = out.sort_values(_CANONICAL_KNOW, kind="stable")
        out.index = pd.DatetimeIndex(pd.to_datetime(out[_CANONICAL_KNOW]))
        return out

    def pit_meta(self) -> PitMeta | None:
        """
        Return PIT provenance metadata for the last as_of call.
        """
        if self._last_as_of is None:
            return None

        knowledge_date = self._last_as_of.date()
        return PitMeta(
            as_of=self._last_as_of.isoformat(),
            source="pysrc.data.dataview.DataView",
            knowledge_cutoff=knowledge_date.isoformat(),
        )
