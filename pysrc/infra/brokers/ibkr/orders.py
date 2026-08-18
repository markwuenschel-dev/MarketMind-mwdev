# infra/brokers/ibkr/orders.py
"""IBKR order executor interface (paper account schema; no live submission in research lane)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pysrc.domain.interfaces import Order, OrderExecutor


@dataclass(frozen=True, slots=True)
class IBKRPaperAccountConfig:
    """Paper trading connection schema per PDR-003."""

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    account_id: str = "DU000000"
    read_only: bool = False


class IBKROrderExecutor(OrderExecutor):
    """Translate domain orders to IBKR API calls (stub — research lane)."""

    def __init__(self, conn_cfg: IBKRPaperAccountConfig) -> None:
        self.cfg = conn_cfg

    def submit(self, order: Order) -> str:
        raise NotImplementedError(
            "IBKR paper order submission is PDR-003 execution lane — not enabled in research scaffold"
        )

    def cancel(self, order_id: str) -> None:
        raise NotImplementedError("IBKR cancel not enabled in research scaffold")

    def status(self, order_id: str) -> str:
        raise NotImplementedError("IBKR status not enabled in research scaffold")


class BrokerConnection(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...


__all__ = ["IBKRPaperAccountConfig", "IBKROrderExecutor", "BrokerConnection"]
