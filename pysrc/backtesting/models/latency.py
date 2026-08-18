from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyModelConfig:
    latency_ms: int = 0


class LatencyModel:
    def __init__(self, config: LatencyModelConfig | None = None) -> None:
        self.config = config or LatencyModelConfig()

    def to_execution_assumptions(self) -> dict[str, int]:
        return {"latency_ms": self.config.latency_ms}
