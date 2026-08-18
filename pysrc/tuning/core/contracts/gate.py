"""GateProtocol: interface for statistical promotion gates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


@runtime_checkable
class GateProtocol(Protocol):
    """Evaluates whether a candidate passes a statistical gate."""

    def evaluate(self, candidate_id: str, scores: dict[str, float]) -> bool: ...

    def reason(self, candidate_id: str, scores: dict[str, float]) -> str: ...


__all__ = ["GateProtocol"]
