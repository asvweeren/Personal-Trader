"""Order lifecycle management: submission, tracking, partial fills, and cancellation."""

import asyncio
from datetime import datetime, timezone, timedelta

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import BrokerAdapter, OrderRequest, OrderResult, OrderSide, OrderType
from app.config import settings
from app.core.event_bus import event_bus, ORDER_PLACED, ORDER_FILLED, ORDER_CANCELLED
from app.models.order import Order, OrderStatus
from app.models.order import OrderType as DBOrderType

logger = structlog.get_logger()

# Statuses that indicate an order is still active
_ACTIVE_STATUSES = {OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}


class OrderManager:
    """Manages the lifecycle of orders from creation to fill/cancel."""

    # Auto-cancel orders pending longer than this
    ORDER_TIMEOUT = timedelta(minutes=10)

    def __init__(self, broker: BrokerAdapter, db: AsyncSession):
        self._broker = broker
        self._db = db
        # Track pending orders by broker_order_id for polling
        self._pending_orders: dict[str, Order] = {}
        # Secondary index: symbol -> set of broker_order_ids (for cancel by symbol)
        self._pending_by_symbol: dict[str, set[str]] = {}
        # Track submission time for timeout detection
        self._submitted_at: dict[str, datetime] = {}

    @property
    def pending_count(self) -> int:
        return len(self._pending_orders)

    async def submit_order(
        self,
        trade_id: int,
        symbol: str,
        side: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        expected_price: float | None = None,
    ) -> OrderResult:
        """Submit an order to the broker and record it in the database."""
        if quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {quantity}")

        order_request = OrderRequest(
            symbol=symbol,
            side=OrderSide(side),
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        result = await self._broker.place_order(order_request)

        # Calculate slippage if we have both expected and filled price
        slippage = None
        if expected_price is not None and result.filled_price is not None:
            slippage = result.filled_price - expected_price
            # Alert on high slippage
            slippage_pct = abs(slippage) / expected_price * 100
            if slippage_pct > settings.max_slippage_pct:
                logger.warning(
                    "order.high_slippage",
                    symbol=symbol,
                    slippage=round(slippage, 4),
                    slippage_pct=round(slippage_pct, 2),
                    expected=expected_price,
                    filled=result.filled_price,
                )
                try:
                    from app.monitoring.alerts import send_alert
                    asyncio.get_event_loop().create_task(send_alert(
                        "High Slippage Alert",
                        f"{symbol}: {slippage_pct:.2f}% slippage\n"
                        f"Expected: {expected_price:.2f} | Filled: {result.filled_price:.2f}\n"
                        f"Slippage: {slippage:+.4f}",
                    ))
                except Exception:
                    pass

        db_order = Order(
            trade_id=trade_id,
            broker_order_id=result.order_id,
            symbol=symbol,
            side=side,
            order_type=DBOrderType(order_type.value),
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            filled_price=result.filled_price,
            filled_quantity=result.filled_quantity,
            status=self._map_status(result.status),
            expected_price=expected_price,
            slippage=slippage,
        )

        mapped = self._map_status(result.status)
        if mapped == OrderStatus.FILLED:
            db_order.filled_at = datetime.now(timezone.utc)

        self._db.add(db_order)
        await self._db.flush()

        event_data = {
            "order_id": result.order_id,
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "status": result.status,
        }
        await event_bus.publish(ORDER_PLACED, event_data)

        if mapped == OrderStatus.FILLED:
            event_data["filled_price"] = result.filled_price
            event_data["filled_quantity"] = result.filled_quantity
            await event_bus.publish(ORDER_FILLED, event_data)
        elif mapped in _ACTIVE_STATUSES:
            self._pending_orders[result.order_id] = db_order
            self._pending_by_symbol.setdefault(symbol, set()).add(result.order_id)
            self._submitted_at[result.order_id] = datetime.now(timezone.utc)

        logger.info(
            "order.submitted",
            order_id=result.order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type.value,
            status=result.status,
        )

        return result

    async def poll_pending_orders(self) -> list[dict]:
        """Check status of all pending orders and update accordingly.

        Returns list of fill info dicts for orders that were filled, each containing:
        trade_id, order_id, symbol, side, filled_price, filled_quantity.
        """
        if not self._pending_orders:
            return []

        # Auto-cancel timed-out orders
        now = datetime.now(timezone.utc)
        timed_out = [
            bid for bid, sub_at in self._submitted_at.items()
            if now - sub_at > self.ORDER_TIMEOUT and bid in self._pending_orders
        ]
        for broker_id in timed_out:
            logger.warning(
                "order.timeout_cancel",
                order_id=broker_id,
                symbol=self._pending_orders[broker_id].symbol,
                pending_seconds=int((now - self._submitted_at[broker_id]).total_seconds()),
            )
            try:
                await self.cancel_order(broker_id)
            except Exception:
                logger.exception("order.timeout_cancel_error", order_id=broker_id)

        filled = []
        to_remove = []

        for broker_id, db_order in self._pending_orders.items():
            try:
                result = await self._broker.get_order_status(broker_id)
                new_status = self._map_status(result.status)

                if new_status == db_order.status:
                    continue

                db_order.status = new_status

                if new_status == OrderStatus.FILLED:
                    db_order.filled_price = result.filled_price
                    db_order.filled_quantity = result.filled_quantity
                    db_order.filled_at = datetime.now(timezone.utc)
                    if db_order.expected_price is not None and result.filled_price is not None:
                        db_order.slippage = result.filled_price - db_order.expected_price
                    to_remove.append(broker_id)
                    filled.append({
                        "trade_id": db_order.trade_id,
                        "order_id": broker_id,
                        "symbol": db_order.symbol,
                        "side": db_order.side,
                        "filled_price": result.filled_price,
                        "filled_quantity": result.filled_quantity,
                    })
                    await event_bus.publish(ORDER_FILLED, {
                        "order_id": broker_id,
                        "trade_id": db_order.trade_id,
                        "symbol": db_order.symbol,
                        "side": db_order.side,
                        "filled_price": result.filled_price,
                        "filled_quantity": result.filled_quantity,
                    })
                    logger.info(
                        "order.filled",
                        order_id=broker_id,
                        symbol=db_order.symbol,
                        price=result.filled_price,
                    )

                elif new_status == OrderStatus.PARTIALLY_FILLED:
                    db_order.filled_quantity = result.filled_quantity
                    db_order.filled_price = result.filled_price
                    logger.info(
                        "order.partial_fill",
                        order_id=broker_id,
                        symbol=db_order.symbol,
                        filled=result.filled_quantity,
                        total=db_order.quantity,
                    )

                elif new_status in (OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.ERROR):
                    to_remove.append(broker_id)
                    await event_bus.publish(ORDER_CANCELLED, {
                        "order_id": broker_id,
                        "trade_id": db_order.trade_id,
                        "symbol": db_order.symbol,
                        "reason": result.message,
                    })
                    logger.info(
                        "order.terminal",
                        order_id=broker_id,
                        symbol=db_order.symbol,
                        status=result.status,
                    )

            except Exception:
                logger.exception("order.poll_error", order_id=broker_id)

        for broker_id in to_remove:
            order = self._pending_orders.pop(broker_id, None)
            self._submitted_at.pop(broker_id, None)
            if order:
                sym_set = self._pending_by_symbol.get(order.symbol)
                if sym_set:
                    sym_set.discard(broker_id)
                    if not sym_set:
                        del self._pending_by_symbol[order.symbol]

        if filled:
            await self._db.flush()

        return filled

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an order with the broker."""
        if broker_order_id not in self._pending_orders:
            return False
        success = await self._broker.cancel_order(broker_order_id)
        if success:
            db_order = self._pending_orders.pop(broker_order_id, None)
            self._submitted_at.pop(broker_order_id, None)
            if db_order:
                db_order.status = OrderStatus.CANCELLED
                sym_set = self._pending_by_symbol.get(db_order.symbol)
                if sym_set:
                    sym_set.discard(broker_order_id)
                    if not sym_set:
                        del self._pending_by_symbol[db_order.symbol]
                await self._db.flush()
            await event_bus.publish(ORDER_CANCELLED, {"order_id": broker_order_id})
            logger.info("order.cancelled", order_id=broker_order_id)
        return success

    async def cancel_all_pending(self) -> int:
        """Cancel all pending orders. Returns count of successfully cancelled orders."""
        cancelled = 0
        broker_ids = list(self._pending_orders.keys())
        for broker_id in broker_ids:
            try:
                if await self.cancel_order(broker_id):
                    cancelled += 1
            except Exception:
                logger.exception("order.cancel_error", order_id=broker_id)
        logger.info("order.cancel_all", total=len(broker_ids), cancelled=cancelled)
        return cancelled

    async def cancel_orders_for_symbol(
        self, symbol: str, side: str | None = None
    ) -> int:
        """Cancel all pending orders for a symbol, optionally filtered by side.

        Args:
            symbol: The symbol to cancel orders for.
            side: If provided, only cancel orders with this side (e.g. "SELL").

        Returns:
            Number of successfully cancelled orders.
        """
        broker_ids = list(self._pending_by_symbol.get(symbol, set()))
        if not broker_ids:
            return 0

        cancelled = 0
        for broker_id in broker_ids:
            db_order = self._pending_orders.get(broker_id)
            if db_order and side and db_order.side != side:
                continue
            try:
                if await self.cancel_order(broker_id):
                    cancelled += 1
            except Exception:
                logger.exception(
                    "order.cancel_for_symbol_error",
                    order_id=broker_id,
                    symbol=symbol,
                )
        if cancelled:
            logger.info(
                "order.cancelled_for_symbol",
                symbol=symbol,
                side=side,
                cancelled=cancelled,
            )
        return cancelled

    def _map_status(self, broker_status: str) -> OrderStatus:
        status_map = {
            # Standard uppercase statuses
            "FILLED": OrderStatus.FILLED,
            "SUBMITTED": OrderStatus.SUBMITTED,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            # IBKR ib_insync statuses (mixed case)
            "PendingSubmit": OrderStatus.SUBMITTED,
            "PreSubmitted": OrderStatus.SUBMITTED,
            "Submitted": OrderStatus.SUBMITTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.CANCELLED,
            "ApiCancelled": OrderStatus.CANCELLED,
        }
        return status_map.get(broker_status, OrderStatus.ERROR)
