import asyncio
import threading
from datetime import datetime

import nest_asyncio
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
from app.risk.market_hours import parse_symbol_for_ibkr

logger = structlog.get_logger()

# Reverse mapping: IBKR primaryExchange → symbol suffix
_IBKR_EXCHANGE_TO_SUFFIX = {
    "AEB": ".AS",
    "SBF": ".PA",
    "BVME": ".BR",
    "IBIS": ".DE",
    "LSE": ".L",
}


class IBKRAdapter(BrokerAdapter):
    """Interactive Brokers adapter using ib_insync.

    Runs ib_insync on a dedicated background thread with its own event loop
    to avoid conflicts with uvicorn's asyncio loop.  Automatically reconnects
    on disconnection using exponential backoff.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib: IB | None = None
        self._ib_loop: asyncio.AbstractEventLoop | None = None
        self._ib_thread: threading.Thread | None = None
        self._market_data_callbacks: dict[str, callable] = {}
        self._auto_reconnect = True
        self._reconnecting = False

    def _start_ib_loop(self):
        """Start a dedicated event loop thread for ib_insync."""
        self._ib_loop = asyncio.new_event_loop()
        nest_asyncio.apply(self._ib_loop)

        def _run():
            asyncio.set_event_loop(self._ib_loop)
            self._ib_loop.run_forever()

        self._ib_thread = threading.Thread(
            target=_run,
            daemon=True,
            name="ib-insync-loop",
        )
        self._ib_thread.start()
        logger.info("ibkr.loop_started")

    async def _run(self, coro):
        """Schedule a coroutine on ib_insync's dedicated loop, await from FastAPI's loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._ib_loop)
        return await asyncio.wrap_future(future)

    async def _sync(self, fn, *args):
        """Run a sync ib_insync function on the dedicated loop thread."""
        async def _wrapper():
            return fn(*args)
        return await self._run(_wrapper())

    async def connect(self) -> None:
        try:
            self._start_ib_loop()

            async def _do_connect():
                self._ib = IB()
                logger.info(
                    "ibkr.connecting",
                    host=self._host,
                    port=self._port,
                    client_id=self._client_id,
                )
                # Retry with increasing timeout — IB Gateway can be slow to handshake
                last_err = None
                for attempt in range(1, 4):
                    try:
                        await self._ib.connectAsync(
                            self._host, self._port,
                            clientId=self._client_id,
                            timeout=20,
                        )
                        # Use delayed market data (type 3) — free, 15-min lag.
                        # Avoids Error 10089 when no real-time subscription.
                        self._ib.reqMarketDataType(3)
                        return  # success
                    except Exception as e:
                        last_err = e
                        logger.warning(
                            "ibkr.connect_retry",
                            attempt=attempt,
                            error=str(e),
                        )
                        if attempt < 3:
                            await asyncio.sleep(3)
                            self._ib = IB()  # fresh IB instance for retry
                raise last_err

            await self._run(_do_connect())
            self._wire_disconnect_handler()
            logger.info("ibkr.connected", host=self._host, port=self._port)
        except Exception as e:
            raise BrokerConnectionError(f"Failed to connect to IBKR: {e}") from e

    def _wire_disconnect_handler(self) -> None:
        """Subscribe to ib_insync disconnect event for auto-reconnect."""
        if self._ib is None:
            return

        def _on_disconnect():
            logger.warning("ibkr.disconnected_event", auto_reconnect=self._auto_reconnect)
            if self._auto_reconnect and not self._reconnecting:
                asyncio.run_coroutine_threadsafe(
                    self._auto_reconnect_loop(), self._ib_loop
                )

        # Clear existing handlers to avoid duplicates on re-connect
        self._ib.disconnectedEvent.clear()
        self._ib.disconnectedEvent += _on_disconnect

    async def _auto_reconnect_loop(self) -> None:
        """Reconnect with exponential backoff (5s → 10s → 20s → ... max 300s)."""
        if self._reconnecting:
            return
        self._reconnecting = True
        delay = 5
        max_delay = 300
        attempt = 0

        try:
            while self._auto_reconnect:
                attempt += 1
                logger.info("ibkr.auto_reconnect", attempt=attempt, delay=delay)
                await asyncio.sleep(delay)

                try:
                    self._ib = IB()
                    await self._ib.connectAsync(
                        self._host, self._port,
                        clientId=self._client_id,
                        timeout=30,
                    )
                    self._ib.reqMarketDataType(3)
                    self._wire_disconnect_handler()
                    logger.info("ibkr.auto_reconnected", attempt=attempt)

                    # Re-subscribe market data if we had subscriptions
                    if self._market_data_callbacks:
                        symbols = list(self._market_data_callbacks.keys())
                        for symbol in symbols:
                            try:
                                contract = self._make_contract(symbol)
                                self._ib.reqMktData(contract)
                            except Exception:
                                logger.warning("ibkr.resubscribe_failed", symbol=symbol)

                    # Send recovery alert
                    try:
                        from app.monitoring.alerts import send_alert
                        await send_alert(
                            "Broker Reconnected",
                            f"IBKR connection restored after {attempt} attempt(s).",
                        )
                    except Exception:
                        pass

                    return  # Success
                except Exception as e:
                    logger.warning("ibkr.auto_reconnect_failed", attempt=attempt, error=str(e))
                    delay = min(delay * 2, max_delay)
        finally:
            self._reconnecting = False

    async def disconnect(self) -> None:
        self._auto_reconnect = False  # Prevent reconnect during intentional shutdown
        if self._ib and self._ib.isConnected():
            await self._sync(self._ib.disconnect)
            logger.info("ibkr.disconnected")
        if self._ib_loop and self._ib_loop.is_running():
            self._ib_loop.call_soon_threadsafe(self._ib_loop.stop)
            if self._ib_thread:
                self._ib_thread.join(timeout=5)

    async def is_connected(self) -> bool:
        if self._ib is None:
            return False
        return self._ib.isConnected()

    async def place_order(self, order: OrderRequest) -> OrderResult:
        try:
            contract = self._make_contract(order.symbol)
            ib_order = self._make_order(order)

            trade: Trade = await self._sync(self._ib.placeOrder, contract, ib_order)

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
            async def _do_cancel():
                for trade in self._ib.openTrades():
                    if str(trade.order.orderId) == order_id:
                        self._ib.cancelOrder(trade.order)
                        return True
                return False

            result = await self._run(_do_cancel())
            if result:
                logger.info("ibkr.order_cancelled", order_id=order_id)
            return result
        except Exception as e:
            raise BrokerOrderError(f"Failed to cancel order: {e}") from e

    async def get_order_status(self, order_id: str) -> OrderResult:
        async def _do_get():
            for trade in self._ib.trades():
                if str(trade.order.orderId) == order_id:
                    return OrderResult(
                        order_id=order_id,
                        status=trade.orderStatus.status,
                        filled_price=trade.orderStatus.avgFillPrice or None,
                        filled_quantity=int(trade.orderStatus.filled) if trade.orderStatus.filled else None,
                    )
            return OrderResult(order_id=order_id, status="UNKNOWN")

        return await self._run(_do_get())

    async def get_positions(self) -> list[Position]:
        portfolio_items = await self._sync(self._ib.portfolio)
        result = []
        for item in portfolio_items:
            if item.position != 0:
                symbol = self._reverse_map_symbol(item.contract)
                result.append(
                    Position(
                        symbol=symbol,
                        quantity=int(item.position),
                        avg_cost=item.averageCost,
                        market_price=item.marketPrice,
                        market_value=item.marketValue,
                        unrealized_pnl=item.unrealizedPNL,
                        realized_pnl=item.realizedPNL,
                    )
                )
        return result

    async def get_portfolio(self) -> Portfolio:
        summary = await self.get_account_summary()
        positions = await self.get_positions()
        return Portfolio(account_summary=summary, positions=positions)

    async def get_account_summary(self) -> AccountSummary:
        account_values = await self._sync(self._ib.accountSummary)

        values = {}
        for av in account_values:
            try:
                values[av.tag] = float(av.value)
            except (ValueError, TypeError):
                pass

        return AccountSummary(
            total_value=values.get("NetLiquidation", 0.0),
            cash=values.get("TotalCashValue", 0.0),
            buying_power=values.get("BuyingPower", 0.0),
            unrealized_pnl=values.get("UnrealizedPnL", 0.0),
            realized_pnl=values.get("RealizedPnL", 0.0),
        )

    async def subscribe_market_data(self, symbols: list[str], callback: callable) -> None:
        async def _do_subscribe():
            for symbol in symbols:
                contract = self._make_contract(symbol)
                self._ib.reqMktData(contract)
                self._market_data_callbacks[symbol] = callback

            # Wire up the pending-tickers event so prices flow to the callback
            def _on_pending_tickers(tickers):
                for ticker in tickers:
                    sym = ticker.contract.symbol
                    # Reverse-map bare symbol back to suffixed symbol
                    matched_symbol = None
                    for s in self._market_data_callbacks:
                        if s == sym or s.split(".")[0].upper() == sym.upper():
                            matched_symbol = s
                            break
                    if matched_symbol is None:
                        continue
                    price = ticker.marketPrice()
                    if price and price > 0 and price == price:  # filters NaN
                        cb = self._market_data_callbacks.get(matched_symbol)
                        if cb:
                            import asyncio
                            asyncio.ensure_future(cb({
                                "symbol": matched_symbol,
                                "price": float(price),
                                "volume": int(ticker.volume) if ticker.volume and ticker.volume == ticker.volume else 0,
                            }))

            self._ib.pendingTickersEvent += _on_pending_tickers

        await self._run(_do_subscribe())
        for symbol in symbols:
            logger.info("ibkr.subscribed_market_data", symbol=symbol)

    async def unsubscribe_market_data(self, symbols: list[str]) -> None:
        for symbol in symbols:
            contract = self._make_contract(symbol)
            await self._sync(self._ib.cancelMktData, contract)
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

        async def _do_request():
            return await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

        bars = await self._run(_do_request())

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
        """Create an IBKR Contract with correct currency/exchange for US and EU stocks."""
        bare_symbol, currency, primary_exchange = parse_symbol_for_ibkr(symbol)
        contract = Contract()
        contract.symbol = bare_symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = currency
        if primary_exchange:
            contract.primaryExchange = primary_exchange
        return contract

    def _reverse_map_symbol(self, contract) -> str:
        """Map IBKR contract back to suffixed symbol (e.g. ASML on AEB → ASML.AS)."""
        exchange = getattr(contract, 'primaryExchange', '') or getattr(contract, 'exchange', '')
        suffix = _IBKR_EXCHANGE_TO_SUFFIX.get(exchange, "")
        return f"{contract.symbol}{suffix}"

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
