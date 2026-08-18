"""SearchProtocol: interface for search algorithm implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pysrc.tuning.core.ir.search_ir import SearchIR


@runtime_checkable
class SearchProtocol(Protocol):
    """Stateless search algorithm that proposes and ranks candidates."""

    def propose(self, ir: SearchIR, n: int) -> list[dict[str, object]]: ...

    def update(self, ir: SearchIR, result: dict[str, float]) -> SearchIR: ...

    def best(self, ir: SearchIR) -> dict[str, object]: ...


__all__ = ["SearchProtocol"]
