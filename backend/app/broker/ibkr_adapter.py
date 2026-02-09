import asyncio
from datetime import datetime

import pandas as pd
import structlog
from ib_insync import IB, Contract, MarketOrder, LimitOrder, StopOrder, Trade

from app.broker.base import (
    AccountSummary,
    BrokerAdapter,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    Portfolio,
    Position,
)
from app.core.exceptions import BrokerConnectionError, BrokerOrderError

logger = structlog.get_logger()


class IBKRAdapter(BrokerAdapter):
    """Interactive Brokers adapter using ib_insync."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib = IB()
        self._market_data_callbacks: dict[str, callable] = {}

    async def connect(self) -> None:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._ib.connect(self._host, self._port, clientId=self._client_id),
            )
            logger.info("ibkr.connected", host=self._host, port=self._port)
        except Exception as e:
            raise BrokerConnectionError(f"Failed to connect to IBKR: {e}") from e

    async def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()
            logger.info("ibkr.disconnected")

    async def is_connected(self) -> bool:
        return self._ib.isConnected()

    async def place_order(self, order: OrderRequest) -> OrderResult:
        try:
            contract = self._make_contract(order.symbol)
            ib_order = self._make_order(order)

            trade: Trade = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._ib.placeOrder(contract, ib_order)
            )

            logger.info(
                "ibkr.order_placed",
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                order_id=trade.order.orderId,
            )

            return OrderResult(
                order_id=str(trade.order.orderId),
                status=trade.orderStatus.status,
                message=f"Order placed: {trade.order.orderId}",
            )
        except Exception as e:
            raise BrokerOrderError(f"Failed to place order: {e}") from e

    async def cancel_order(self, order_id: str) -> bool:
        try:
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    logger.info("ibkr.order_cancelled", order_id=order_id)
                    return True
            return False
        except Exception as e:
            raise BrokerOrderError(f"Failed to cancel order: {e}") from e

    async def get_order_status(self, order_id: str) -> OrderResult:
        for trade in self._ib.trades():
            if str(trade.order.orderId) == order_id:
                return OrderResult(
                    order_id=order_id,
                    status=trade.orderStatus.status,
                    filled_price=trade.orderStatus.avgFillPrice or None,
                    filled_quantity=int(trade.orderStatus.filled) if trade.orderStatus.filled else None,
                )
        return OrderResult(order_id=order_id, status="UNKNOWN")

    async def get_positions(self) -> list[Position]:
        positions = await asyncio.get_event_loop().run_in_executor(None, self._ib.positions)
        result = []
        for pos in positions:
            if pos.position != 0:
                result.append(
                    Position(
                        symbol=pos.contract.symbol,
                        quantity=int(pos.position),
                        avg_cost=pos.avgCost,
                        market_price=pos.avgCost,  # Updated via market data
                        market_value=pos.position * pos.avgCost,
                        unrealized_pnl=0.0,
                    )
                )
        return result

    async def get_portfolio(self) -> Portfolio:
        summary = await self.get_account_summary()
        positions = await self.get_positions()
        return Portfolio(account_summary=summary, positions=positions)

    async def get_account_summary(self) -> AccountSummary:
        account_values = await asyncio.get_event_loop().run_in_executor(
            None, self._ib.accountSummary
        )

        values = {}
        for av in account_values:
            values[av.tag] = float(av.value) if av.value else 0.0

        return AccountSummary(
            total_value=values.get("NetLiquidation", 0.0),
            cash=values.get("TotalCashValue", 0.0),
            buying_power=values.get("BuyingPower", 0.0),
            unrealized_pnl=values.get("UnrealizedPnL", 0.0),
            realized_pnl=values.get("RealizedPnL", 0.0),
        )

    async def subscribe_market_data(self, symbols: list[str], callback: callable) -> None:
        for symbol in symbols:
            contract = self._make_contract(symbol)
            self._ib.reqMktData(contract)
            self._market_data_callbacks[symbol] = callback
            logger.info("ibkr.subscribed_market_data", symbol=symbol)

    async def unsubscribe_market_data(self, symbols: list[str]) -> None:
        for symbol in symbols:
            contract = self._make_contract(symbol)
            self._ib.cancelMktData(contract)
            self._market_data_callbacks.pop(symbol, None)

    async def get_historical_data(
        self,
        symbol: str,
        duration: str = "30 D",
        bar_size: str = "1 hour",
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        contract = self._make_contract(symbol)
        end_dt = end_date or datetime.now()

        bars = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            ),
        )

        if not bars:
            return pd.DataFrame()

        data = [
            {
                "timestamp": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        return pd.DataFrame(data)

    def _make_contract(self, symbol: str) -> Contract:
        """Create an IBKR Contract. Assumes US stock by default."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def _make_order(self, order: OrderRequest):
        """Convert OrderRequest to an ib_insync order object."""
        action = "BUY" if order.side == OrderSide.BUY else "SELL"

        if order.order_type == OrderType.MARKET:
            return MarketOrder(action, order.quantity)
        elif order.order_type == OrderType.LIMIT:
            return LimitOrder(action, order.quantity, order.limit_price)
        elif order.order_type == OrderType.STOP:
            return StopOrder(action, order.quantity, order.stop_price)
        else:
            raise BrokerOrderError(f"Unsupported order type: {order.order_type}")
