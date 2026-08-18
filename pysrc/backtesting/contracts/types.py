from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any


class ValidationStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class ArtifactRef:
    role: str
    path: str
    cas: str | None = None
    attest: str | None = None


@dataclass(frozen=True)
class RunBundleRef:
    bundle_path: str
    run_id: str | None = None
    manifest_ref: ArtifactRef | None = None


@dataclass(frozen=True)
class PitMeta:
    as_of: str | None = None
    source: str | None = None
    knowledge_cutoff: str | None = None


@dataclass(frozen=True)
class MarketSlice:
    as_of: str
    prices: list[dict[str, Any]] = field(default_factory=list)
    features: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    pit_meta: PitMeta | None = None


@dataclass(frozen=True)
class Fill:
    symbol: str
    quantity: float
    price: float
    side: str
    timestamp: str


@dataclass(frozen=True)
class CostEstimate:
    total_cost: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LedgerPosition:
    symbol: str
    quantity: float
    average_price: float


@dataclass(frozen=True)
class LedgerSnapshot:
    timestamp: str
    cash: float
    positions: list[LedgerPosition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestResult:
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    ledger: LedgerSnapshot | None = None


@dataclass(frozen=True)
class ValidationReport:
    status: ValidationStatus
    reason_code: str
    message: str
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketDataPoint:
    symbol: str
    timestamp: str
    value: float
    field_name: str = "close"


def to_primitive(value: Any) -> Any:
    """Convert nested dataclasses and datetimes into JSON-friendly values."""
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_primitive(item) for item in value]
    if isinstance(value, tuple):
        return [to_primitive(item) for item in value]
    return value
