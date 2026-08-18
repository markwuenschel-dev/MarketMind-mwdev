"""
Governed universe membership: relist-safe append-only event model.

Phase I-A operates at daily granularity (event_date is a date). Sub-day TTLs and
intraday restatement ordering are out of scope for this phase.

Membership is never inferred from price or store presence. A symbol is active at T iff
the latest membership event at or before T is a listing/activation event. Delist excludes
on and after delist_date. Same-date tie-break: when multiple events share the same
event_date, delist wins over list so that "exclude on and after delist_date" is deterministic.

Phase I convenience: FIXTURE_SEED is temporary; not the long-term governed security-master.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

# Event: (symbol, event_date, event_type, reason_code)
_EventType = str
_EVENT_LIST = "list"
_EVENT_DELIST = "delist"
_Event = tuple[str, date, _EventType, str]


def _tie_break_key(ev: _Event) -> tuple[date, int]:
    """Sort key for same-date tie-break: (event_date, 1 if delist else 0). Delist wins (larger)."""
    _, ev_date, et, _ = ev
    return (ev_date, 1 if et == _EVENT_DELIST else 0)


class MembershipReasonCode(StrEnum):
    """Reason for a membership event. Mandatory on every record."""

    FIXTURE_SEED = "fixture_seed"  # Phase I convenience; not long-term security-master
    GOVERNED = "governed"
    MANUAL = "manual"
    RELIST = "relist"  # Relisting after a prior delist


class Universe:
    """
    Append-only event log. One event type per row: list or delist.
    members_as_of(T) selects latest event per symbol with event_date <= T;
    same-date tie-break: delist wins over list. Symbol is active iff that latest event is a list event.
    """

    def __init__(self) -> None:
        self._events: list[_Event] = []

    def register(
        self,
        symbol: str,
        list_date: date,
        *,
        reason_code: MembershipReasonCode = MembershipReasonCode.GOVERNED,
    ) -> None:
        """Append a list/activation event. Relist-safe: call again after a delist to relist."""
        self._events.append((symbol, list_date, _EVENT_LIST, reason_code.value))

    def delist(
        self,
        symbol: str,
        delist_date: date,
        *,
        reason_code: MembershipReasonCode = MembershipReasonCode.GOVERNED,
    ) -> None:
        """Append a delist event. Symbol excluded for T >= delist_date."""
        self._events.append((symbol, delist_date, _EVENT_DELIST, reason_code.value))

    def has_listing(self, symbol: str) -> bool:
        """True if this symbol has ever had a list event (legacy/fixture dedupe)."""
        return any(s == symbol and et == _EVENT_LIST for s, _, et, _ in self._events)

    def _latest_event_at_or_before(self, symbol: str, as_of_date: date) -> _Event | None:
        """
        Latest event for symbol with event_date <= as_of_date.
        Same-date tie-break: delist wins over list (deterministic).
        """
        candidates = [ev for ev in self._events if ev[0] == symbol and ev[1] <= as_of_date]
        if not candidates:
            return None
        return max(candidates, key=_tie_break_key)

    def effective_state_at(self, symbol: str, as_of_date: date) -> tuple[bool, _Event | None]:
        """
        Effective membership state for symbol at as_of_date.

        Returns (is_active, latest_event). Symbol is active iff the latest event
        at or before as_of_date is a list event. Used by fixture seeding to decide
        whether to append a new list event (relist) or skip (already active).
        """
        latest = self._latest_event_at_or_before(symbol, as_of_date)
        if latest is None:
            return (False, None)
        return (latest[2] == _EVENT_LIST, latest)

    def members_as_of(self, as_of_date: date) -> set[str]:
        """
        Return symbols active at as_of_date.

        For each symbol, take the latest event with event_date <= as_of_date
        (same-date tie-break: delist wins over list). Symbol is active iff that event is a list event.
        """
        result: set[str] = set()
        symbols_seen = {s for s, _, _, _ in self._events}
        for symbol in symbols_seen:
            is_active, _ = self.effective_state_at(symbol, as_of_date)
            if is_active:
                result.add(symbol)
        return result
