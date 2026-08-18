from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pysrc.backtesting.contracts.plan import BacktestPlan
from pysrc.backtesting.contracts.types import (
    BacktestResult,
    CostEstimate,
    LedgerSnapshot,
    MarketSlice,
    PitMeta,
    RunBundleRef,
    ValidationReport,
)


@runtime_checkable
class AsOfView(Protocol):
    def as_of(self, ts: datetime) -> MarketSlice:
        """Return a PIT-safe market slice."""

    def pit_meta(self) -> PitMeta | None:
        """Return PIT provenance metadata when available."""


@runtime_checkable
class BacktestArtifactStore(Protocol):
    def put_json(self, role: str, payload: dict[str, Any]) -> Any:
        """Persist a JSON artifact under a filename role."""

    def put_bytes(self, role: str, payload: bytes, media_type: str) -> Any:
        """Persist a binary artifact under a filename role."""

    def get_json(self, ref: Any) -> dict[str, Any]:
        """Load a JSON artifact from a stored reference."""


@runtime_checkable
class BacktestEngine(Protocol):
    def run(self, plan: BacktestPlan, data: Any, store: BacktestArtifactStore) -> BacktestResult:
        """Execute a backtest for the provided plan."""


@runtime_checkable
class ExecutionModel(Protocol):
    def simulate(self, orders: list[dict[str, Any]], ctx: dict[str, Any]) -> list[dict[str, Any]]:
        """Simulate order execution and return fill payloads."""


@runtime_checkable
class CostModel(Protocol):
    def estimate(self, fills: list[dict[str, Any]], ctx: dict[str, Any]) -> CostEstimate:
        """Estimate transaction costs for simulated fills."""


@runtime_checkable
class Ledger(Protocol):
    def apply(
        self,
        fills: list[dict[str, Any]],
        corporate_actions: list[dict[str, Any]] | None,
    ) -> LedgerSnapshot:
        """Apply fills and corporate actions to produce a ledger snapshot."""


@runtime_checkable
class BacktestValidator(Protocol):
    def validate(
        self,
        result: BacktestResult,
        ctx: dict[str, Any],
        store: BacktestArtifactStore,
    ) -> ValidationReport:
        """Validate a backtest result and optionally emit artifacts."""


@runtime_checkable
class BacktestSuiteOrchestrator(Protocol):
    def execute(self, suite_plan: Any) -> RunBundleRef:
        """Execute a backtest suite and return a bundle reference."""
