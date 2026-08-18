from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiquidityModelConfig:
    max_participation_rate: float = 1.0


class LiquidityModel:
    def __init__(self, config: LiquidityModelConfig | None = None) -> None:
        self.config = config or LiquidityModelConfig()

    def to_execution_assumptions(self) -> dict[str, float]:
        return {"max_participation_rate": self.config.max_participation_rate}
