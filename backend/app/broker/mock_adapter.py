import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import structlog

from app.broker.base import (
    AccountSummary,
    BrokerAdapter,
    OrderRequest,
    OrderResult,
    OrderSide,
    Portfolio,
    Position,
)

logger = structlog.get_logger()


@dataclass
class MockPosition:
    symbol: str
    quantity: int
    avg_cost: float


class MockBrokerAdapter(BrokerAdapter):
    """Mock broker adapter for testing. Simulates order fills and portfolio state."""

    def __init__(self, initial_cash: float = 5000.0):
        self._connected = False
        self._cash = initial_cash
        self._positions: dict[str, MockPosition] = {}
        self._orders: dict[str, OrderResult] = {}
        self._prices: dict[str, float] = {}
        self._callbacks: dict[str, callable] = {}

    async def connect(self) -> None:
        self._connected = True
        logger.info("mock_broker.connected")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("mock_broker.disconnected")

    async def is_connected(self) -> bool:
        return self._connected

    def set_price(self, symbol: str, price: float) -> None:
        """Set a mock price for a symbol (for testing)."""
        self._prices[symbol] = price

    async def place_order(self, order: OrderRequest) -> OrderResult:
        order_id = str(uuid.uuid4())[:8]
        price = self._prices.get(order.symbol, order.limit_price or 100.0)

        # Simulate immediate fill for market orders
        cost = price * order.quantity
        if order.side == OrderSide.BUY:
            if cost > self._cash:
                return OrderResult(
                    order_id=order_id, status="REJECTED", message="Insufficient funds"
                )
            self._cash -= cost
            pos = self._positions.get(order.symbol)
            if pos:
                total_qty = pos.quantity + order.quantity
                pos.avg_cost = (pos.avg_cost * pos.quantity + cost) / total_qty
                pos.quantity = total_qty
            else:
                self._positions[order.symbol] = MockPosition(
                    symbol=order.symbol, quantity=order.quantity, avg_cost=price
                )
        else:
            pos = self._positions.get(order.symbol)
            if not pos or pos.quantity < order.quantity:
                return OrderResult(
                    order_id=order_id, status="REJECTED", message="Insufficient position"
                )
            self._cash += cost
            pos.quantity -= order.quantity
            if pos.quantity == 0:
                del self._positions[order.symbol]

        result = OrderResult(
            order_id=order_id,
            status="FILLED",
            filled_price=price,
            filled_quantity=order.quantity,
        )
        self._orders[order_id] = result

        logger.info(
            "mock_broker.order_filled",
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=price,
        )
        return result

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = "CANCELLED"
            return True
        return False

    async def get_order_status(self, order_id: str) -> OrderResult:
        return self._orders.get(
            order_id, OrderResult(order_id=order_id, status="UNKNOWN")
        )

    async def get_positions(self) -> list[Position]:
        result = []
        for pos in self._positions.values():
            market_price = self._prices.get(pos.symbol, pos.avg_cost)
            result.append(
                Position(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    avg_cost=pos.avg_cost,
                    market_price=market_price,
                    market_value=market_price * pos.quantity,
                    unrealized_pnl=(market_price - pos.avg_cost) * pos.quantity,
                )
            )
        return result

    async def get_portfolio(self) -> Portfolio:
        summary = await self.get_account_summary()
        positions = await self.get_positions()
        return Portfolio(account_summary=summary, positions=positions)

    async def get_account_summary(self) -> AccountSummary:
        positions = await self.get_positions()
        positions_value = sum(p.market_value for p in positions)
        unrealized = sum(p.unrealized_pnl for p in positions)
        total = self._cash + positions_value

        return AccountSummary(
            total_value=total,
            cash=self._cash,
            buying_power=self._cash,
            unrealized_pnl=unrealized,
            realized_pnl=0.0,
        )

    async def subscribe_market_data(self, symbols: list[str], callback: callable) -> None:
        for symbol in symbols:
            self._callbacks[symbol] = callback
            logger.info("mock_broker.subscribed", symbol=symbol)

    async def unsubscribe_market_data(self, symbols: list[str]) -> None:
        for symbol in symbols:
            self._callbacks.pop(symbol, None)

    async def get_historical_data(
        self,
        symbol: str,
        duration: str = "30 D",
        bar_size: str = "1 hour",
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Generate mock historical data for testing."""
        rng = np.random.default_rng(42)
        n_bars = 100
        base_price = self._prices.get(symbol, 100.0)

        dates = pd.date_range(end=datetime.now(timezone.utc), periods=n_bars, freq="h")
        returns = rng.normal(0, 0.02, n_bars)
        prices = base_price * np.cumprod(1 + returns)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices * (1 + rng.normal(0, 0.005, n_bars)),
                "high": prices * (1 + abs(rng.normal(0, 0.01, n_bars))),
                "low": prices * (1 - abs(rng.normal(0, 0.01, n_bars))),
                "close": prices,
                "volume": rng.integers(1000, 100000, n_bars),
            }
        )
