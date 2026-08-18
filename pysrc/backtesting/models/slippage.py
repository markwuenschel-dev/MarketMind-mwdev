from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageModelConfig:
    slippage_bps: float = 0.0


class SlippageModel:
    def __init__(self, config: SlippageModelConfig | None = None) -> None:
        self.config = config or SlippageModelConfig()

    def to_execution_assumptions(self) -> dict[str, float]:
        return {"slippage_bps": self.config.slippage_bps}
