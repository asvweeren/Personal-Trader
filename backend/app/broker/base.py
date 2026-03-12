from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import pandas as pd


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None


@dataclass
class OrderResult:
    order_id: str
    status: str
    filled_price: float | None = None
    filled_quantity: int | None = None
    message: str = ""


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class AccountSummary:
    total_value: float
    cash: float
    buying_power: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class Portfolio:
    account_summary: AccountSummary
    positions: list[Position] = field(default_factory=list)


@dataclass
class BarData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


def round_to_tick(price: float, min_tick: float) -> float:
    """Round a price to the nearest valid tick increment."""
    if min_tick <= 0:
        return round(price, 2)
    return round(round(price / min_tick) * min_tick, 8)


class BrokerAdapter(ABC):
    """Abstract interface for broker interactions. Supports both paper and live trading."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the broker."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connected to the broker."""

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResult:
        """Place an order with the broker."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""

    async def cancel_open_orders_for_symbol(self, symbol: str) -> int:
        """Cancel all open orders at the broker for a given symbol.

        Returns the number of orders cancelled. Default implementation
        returns 0 (no-op) for adapters that don't support it.
        """
        return 0

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderResult:
        """Get the status of an order."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all current positions."""

    @abstractmethod
    async def get_portfolio(self) -> Portfolio:
        """Get full portfolio including account summary and positions."""

    @abstractmethod
    async def get_account_summary(self) -> AccountSummary:
        """Get account summary (cash, value, P&L)."""

    @abstractmethod
    async def subscribe_market_data(
        self, symbols: list[str], callback: callable
    ) -> None:
        """Subscribe to real-time market data for symbols."""

    @abstractmethod
    async def unsubscribe_market_data(self, symbols: list[str]) -> None:
        """Unsubscribe from market data."""

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        duration: str,
        bar_size: str,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
