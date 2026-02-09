"""Order lifecycle management: submission, tracking, partial fills, and cancellation."""

import asyncio
from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import BrokerAdapter, OrderRequest, OrderResult, OrderSide, OrderType
from app.core.event_bus import event_bus, ORDER_PLACED, ORDER_FILLED, ORDER_CANCELLED
from app.models.order import Order, OrderStatus
from app.models.order import OrderType as DBOrderType

logger = structlog.get_logger()

# Statuses that indicate an order is still active
_ACTIVE_STATUSES = {OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}


class OrderManager:
    """Manages the lifecycle of orders from creation to fill/cancel."""

    def __init__(self, broker: BrokerAdapter, db: AsyncSession):
        self._broker = broker
        self._db = db
        # Track pending orders by broker_order_id for polling
        self._pending_orders: dict[str, Order] = {}

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
    ) -> OrderResult:
        """Submit an order to the broker and record it in the database."""
        order_request = OrderRequest(
            symbol=symbol,
            side=OrderSide(side),
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        result = await self._broker.place_order(order_request)

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
        )

        if result.status == "FILLED":
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

        if result.status == "FILLED":
            event_data["filled_price"] = result.filled_price
            event_data["filled_quantity"] = result.filled_quantity
            await event_bus.publish(ORDER_FILLED, event_data)
        elif result.status in ("SUBMITTED", "PARTIALLY_FILLED"):
            self._pending_orders[result.order_id] = db_order

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

    async def poll_pending_orders(self) -> list[OrderResult]:
        """Check status of all pending orders and update accordingly."""
        if not self._pending_orders:
            return []

        filled = []
        to_remove = []

        for broker_id, db_order in self._pending_orders.items():
            try:
                result = await self._broker.get_order_status(broker_id)
                new_status = self._map_status(result.status)

                if new_status == db_order.status:
                    continue

                db_order.status = new_status

                if result.status == "FILLED":
                    db_order.filled_price = result.filled_price
                    db_order.filled_quantity = result.filled_quantity
                    db_order.filled_at = datetime.now(timezone.utc)
                    to_remove.append(broker_id)
                    filled.append(result)
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

                elif result.status == "PARTIALLY_FILLED":
                    db_order.filled_quantity = result.filled_quantity
                    db_order.filled_price = result.filled_price
                    logger.info(
                        "order.partial_fill",
                        order_id=broker_id,
                        symbol=db_order.symbol,
                        filled=result.filled_quantity,
                        total=db_order.quantity,
                    )

                elif result.status in ("CANCELLED", "REJECTED", "ERROR"):
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
            self._pending_orders.pop(broker_id, None)

        if filled:
            await self._db.flush()

        return filled

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an order with the broker."""
        success = await self._broker.cancel_order(broker_order_id)
        if success:
            db_order = self._pending_orders.pop(broker_order_id, None)
            if db_order:
                db_order.status = OrderStatus.CANCELLED
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

    def _map_status(self, broker_status: str) -> OrderStatus:
        status_map = {
            "FILLED": OrderStatus.FILLED,
            "SUBMITTED": OrderStatus.SUBMITTED,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
        }
        return status_map.get(broker_status, OrderStatus.ERROR)
