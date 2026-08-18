from __future__ import annotations

from pysrc.backtesting.contracts.errors import NotImplementedLaneError
from pysrc.backtesting.contracts.registry import register_engine


class EventDrivenEngineAdapter:
    def run(self, plan, data, store):
        raise NotImplementedLaneError(
            "event_driven engine is scaffolded only. Use engine_id='vectorized.sma' for executable runtime behavior."
        )


register_engine("event_driven.scaffold", lambda: EventDrivenEngineAdapter())
