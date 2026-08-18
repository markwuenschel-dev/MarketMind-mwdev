from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmptyCorporateActionsFeed:
    def as_of(self, timestamp: str) -> list[dict[str, str]]:
        return []
