# py/domain/interfaces.py
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import (
    Any,
    Protocol,
    TypeVar,
)

import polars as pl  # Prioritize Polars for efficient, lazy data handling
from polars._typing import SchemaDict

from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.stages.cleaning.validators.contracts import (
    MarketDataFrameSchema,  # From contracts.py
)

_LOG = get_logger(__name__)


# Accurate models based on IBKR API docs
@dataclass
class Order:
    order_id: int = 0
    action: str = ""  # "BUY" or "SELL"
    total_quantity: float = 0.0
    order_type: str = ""  # "MKT", "LMT", "STP", etc.
    lmt_price: float | None = None  # Limit price
    aux_price: float | None = None  # Stop or trailing amount
    tif: str = "DAY"  # Time in force: "DAY", "GTC", etc.
    account: str | None = None
    symbol: str | None = None  # From contract
    solicited: bool = False  # Broker/adviser initiated
    extra: dict[str, Any] = field(default_factory=dict)  # Extensible (e.g., ocaGroup, transmit)

    @classmethod
    def market(cls, symbol: str, qty: float) -> "Order":
        """Create a market order."""
        return cls(
            symbol=symbol,
            total_quantity=abs(qty),
            action="BUY" if qty > 0 else "SELL",
            order_type="MKT",
        )


@dataclass
class Position:
    account: str = ""
    symbol: str = ""  # From contract
    position: float = 0.0  # Quantity (pos)
    avg_cost: float = 0.0
    model_code: str | None = None  # For models
    extra: dict[str, Any] = field(default_factory=dict)  # Extensible (e.g., unrealizedPNL)


# Schemas (extended)
PositionSchema: SchemaDict = {
    "account": pl.Utf8,
    "symbol": pl.Utf8,
    "position": pl.Float64,
    "avg_cost": pl.Float64,
    "model_code": pl.Utf8,
}

HistoricalSchema: SchemaDict = MarketDataFrameSchema().required_columns
HistoricalSchema["timestamp"] = pl.Datetime


# Protocols for abstract interfaces (extensible via subclassing)
class OrderExecutor(Protocol):
    """Protocol for order execution, extensible for different brokers."""

    def submit(self, order: Order) -> str:
        """Submit an order and return its ID. Supports single orders; extend for batches."""
        ...

    def submit_batch(self, orders: list[Order]) -> list[str]:
        """Combinatoric: Submit multiple orders efficiently in one call."""
        ...  # Default: [self.submit(o) for o in orders] in implementations

    def cancel(self, order_id: str) -> None: ...

    def status(self, order_id: str) -> str: ...


class PositionService(Protocol):
    """Protocol for position management, returning Polars DataFrame for efficiency."""

    def get_positions(self) -> list[Position]:
        """Legacy list return; kept for compatibility."""
        ...

    def get_positions_as_polars(self) -> pl.DataFrame:
        """Polars-prioritized: Return positions as DataFrame for vectorized ops/combinatorics."""
        # Ensures schema: return df.cast(PositionSchema) in implementations
        ...


class MarketDataProvider(Protocol):
    """Protocol for market data, prioritizing Polars for lazy, efficient queries."""

    def get_price(self, symbol: str) -> float:
        """Get current price for a single symbol."""
        ...

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Combinatoric: Batch fetch prices for multiple symbols efficiently."""
        ...  # Default: {s: self.get_price(s) for s in symbols}

    def get_historical(
        self, symbol: str, start: str, end: str, interval: str = "1min", lazy: bool = True
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Get historical OHLCV as Polars LazyFrame (default for efficiency) or eager DataFrame.
        Supports lazy for deferred computation in pipelines.
        Raises DataFetchError on failure.
        # Ensures schema: return lf.cast(HistoricalSchema) if lazy else df.cast(HistoricalSchema)
        """
        ...

    def get_historical_batch(
        self, symbols: list[str], start: str, end: str, interval: str = "1min", lazy: bool = True
    ) -> dict[str, pl.LazyFrame | pl.DataFrame]:
        """Combinatoric: Batch historical data for multiple symbols, composable with Polars joins."""
        ...  # Default: {s: self.get_historical(s, start, end, interval, lazy) for s in symbols}

    def map_over(
        self,
        symbols: list[str],
        fn: Callable[[str], pl.LazyFrame],
        combine: Callable[[list[pl.LazyFrame]], pl.LazyFrame] = pl.concat,
    ) -> pl.LazyFrame:
        """Combinatoric: Apply `fn` lazily over symbols and combine (e.g., for cross-symbol features)."""
        return combine([fn(s) for s in symbols])

    async def stream_realtime(
        self, symbol: str, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        """Stream real-time chunks as Polars frames."""
        ...


# Optional async mix-in for efficiency in IO-bound providers
class AsyncMarketDataProvider(MarketDataProvider, Protocol):
    async def get_price_async(self, symbol: str) -> float: ...
    async def get_prices_async(self, symbols: list[str]) -> dict[str, float]: ...

    # Sync wrappers in impls: def get_price(self, s): return asyncio.run(self.get_price_async(s))


# Extensible addition: Protocol for economic/alternative data (e.g., FRED, ESG)
class EconomicDataProvider(Protocol):
    def get_indicator(self, indicator: str, start: str, end: str) -> pl.LazyFrame:
        """Fetch economic indicator (e.g., GDP) as lazy Polars frame for efficiency."""
        ...


# Generic over the provider type for narrowed, type-safe returns
T_co = TypeVar("T_co", covariant=True)


# Abstract Factory with registry (aligned with infra_factory.py and pipeline_core_base.py cleaning)
class ProviderFactory[T_co](ABC):
    """Abstract factory with registry for dynamic instantiation. Supports combinatorics via composition."""

    _registry: dict[str, type["ProviderFactory"]] = {}  # Extensible registry (key: provider_type)

    @classmethod
    def register(
        cls, provider_type: str
    ) -> Callable[[type["ProviderFactory"]], type["ProviderFactory"]]:
        """Decorator for registering factory subclasses dynamically (like infra_factory.py)."""

        def decorator(subclass: type["ProviderFactory"]) -> type["ProviderFactory"]:
            cls._registry[provider_type.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def load_entry_points(cls, group: str = "pysrc.providers") -> int:
        """Load providers from entry points (extensible plugins, like infra_factory.py)."""
        loaded = 0
        for ep in entry_points(group=group):
            try:
                obj = ep.load()
                cls.register(ep.name)(obj)
                loaded += 1
            except Exception as e:
                _LOG.warning("Failed to load provider entry point %s: %s", ep.name, e)
        return loaded

    @classmethod
    def create(
        cls: type["ProviderFactory[T_co]"], provider_type: str, config: dict[str, Any], **kwargs
    ) -> T_co:
        """Dynamic factory: Create provider based on type/pipeline_config. Raises ValueError if unregistered."""
        provider_type_lc = provider_type.lower()
        if provider_type_lc not in cls._registry:
            raise ValueError(
                f"Unknown provider type: {provider_type}. Available: {', '.join(cls._registry.keys())}"
            )
        factory = cls._registry[provider_type_lc]()
        instance = factory.build_provider(config, **kwargs)
        # Post-creation validation (Pydantic-like from pipeline_config.py)
        if hasattr(instance, "validate_section"):
            instance.validate_section()
        return instance

    @abstractmethod
    def build_provider(self, config: dict[str, Any], **kwargs) -> T_co:
        """Abstract method: Subclasses implement to build specific providers (e.g., IBKR, FRED)."""
        ...


# Example registration (extensible; implementations would decorate their factories)
# @ProviderFactory.register("ibkr")
# class IBKRFactory(ProviderFactory[MarketDataProvider]):
#     def build_provider(self, pipeline_config: Dict[str, Any], **kwargs) -> MarketDataProvider:
#         # Return IBKR implementation, e.g., using pipeline_config['host']
#         ...


# Additional trading service classes
class RiskManager:
    """Risk management service."""

    def validate(self, order):
        """Validate order against risk constraints."""
        pass


class PositionSizer:
    """Position sizing service."""

    def size(self, symbol: str, signal: float, price: float) -> float:
        """Calculate position size based on signal and price."""
        return 100.0  # Default size


# ---------------------------------------------------------------------------
# Abstract API data manager (merged from pysrc.data.base; deprecated there)
# ---------------------------------------------------------------------------
def __get_abstract_api_deps():
    from pysrc.core.errors import DataFetchError, NoDataError
    from pysrc.core.validation import lazy_validate_ohlcv
    from pysrc.ops.mm_logkit import get_logger
    from pysrc.pipeline.stages.market_data.sources.runtime import APIDataSource

    return APIDataSource, DataFetchError, NoDataError, get_logger, lazy_validate_ohlcv


class AbstractAPIDataManager:
    """Abstract base for API data managers (registry + load_data)."""

    registry: dict[str, type[Any]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry = {}

    @classmethod
    def register(cls, source_type: str):
        def decorator(klass: type[Any]):
            cls.registry[source_type] = klass
            _APIDataSource, _DataFetchError, _NoDataError, _get_logger, _ = (
                __get_abstract_api_deps()
            )
            _get_logger(__name__).info(
                "Registered data source", extra={"type": source_type, "manager": cls.__name__}
            )
            return klass

        return decorator

    def __init__(self, config: dict[str, Any]):
        self.config = config
        APIDataSource, DataFetchError, NoDataError, get_logger, lazy_validate_ohlcv = (
            __get_abstract_api_deps()
        )
        self._APIDataSource = APIDataSource
        self._DataFetchError = DataFetchError
        self._NoDataError = NoDataError
        self._lazy_validate_ohlcv = lazy_validate_ohlcv
        self.sources: dict[str, Any] = {}
        self._register_sources(config.get(self._config_key, []))

    @property
    def _config_key(self) -> str:
        raise NotImplementedError("Subclasses must define _config_key")

    def _register_sources(self, sources_cfg: list[dict]):
        for source_cfg in sources_cfg:
            source_type = source_cfg["type"]
            if source_type not in self.registry:
                raise ValueError(f"Unknown data source type: {source_type}")
            src = self.registry[source_type](config=source_cfg)
            self.sources[source_type] = src

    def add_source(self, source_type: str, source: Any):
        self.sources[source_type] = source

    async def load_data(
        self, query: str | list[str], source_name: str | None = None
    ) -> pl.LazyFrame | dict[str, pl.LazyFrame | Exception]:
        import asyncio

        if isinstance(query, str):
            src = self.sources.get(source_name or next(iter(self.sources)))
            if not src:
                raise ValueError(f"No source found: {source_name}")
            try:
                params = self._build_params(query)
                json_data = await src._request(src.base_url, params, src.timeout)
                if not json_data:
                    raise self._NoDataError(query)
                lf = pl.LazyFrame(json_data)
                self._lazy_validate_ohlcv(lf)
                return lf
            except Exception as e:
                raise self._DataFetchError(
                    f"Load failed for {query}", details={"error": str(e)}
                ) from e

        # list
        async def fetch_one(q: str):
            try:
                return await self.load_data(q, source_name)
            except Exception as e:
                return e

        results = await asyncio.gather(*(fetch_one(q) for q in query), return_exceptions=False)
        return dict(zip(query, results, strict=False))

    def _build_params(self, query: str) -> dict:
        return {"query": query}
