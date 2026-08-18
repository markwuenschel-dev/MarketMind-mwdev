from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairsColumns:
    """Canonical live column names for the Phase I-D pairs slice."""

    leg_a: str
    leg_b: str

    @property
    def a_close(self) -> str:
        return f"{self.leg_a}.close"

    @property
    def b_close(self) -> str:
        return f"{self.leg_b}.close"

    @property
    def beta(self) -> str:
        return f"beta_{self.leg_a}_{self.leg_b}"

    @property
    def spread(self) -> str:
        return f"spread_{self.leg_a}_{self.leg_b}"

    @property
    def half_life(self) -> str:
        return f"hl_{self.spread}"
