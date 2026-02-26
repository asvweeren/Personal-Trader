import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

import structlog

logger = structlog.get_logger()

Callback = Callable[..., Coroutine[Any, Any, None]]


class EventBus:
    """Async publish/subscribe event bus for decoupled communication between modules."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callback]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callback) -> None:
        self._subscribers[event_type].append(callback)
        logger.debug("event_bus.subscribe", event_type=event_type, callback=callback.__name__)

    def unsubscribe(self, event_type: str, callback: Callback) -> None:
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    async def publish(self, event_type: str, data: Any = None) -> None:
        subscribers = self._subscribers.get(event_type, [])
        if not subscribers:
            return

        logger.debug("event_bus.publish", event_type=event_type, subscriber_count=len(subscribers))

        tasks = [self._safe_call(cb, event_type, data) for cb in subscribers]
        await asyncio.gather(*tasks)

    async def _safe_call(self, callback: Callback, event_type: str, data: Any) -> None:
        try:
            await callback(data)
        except Exception:
            logger.exception(
                "event_bus.callback_error",
                event_type=event_type,
                callback=callback.__name__,
            )


# Singleton event bus
event_bus = EventBus()

# Event type constants
SIGNAL_GENERATED = "signal.generated"
ORDER_PLACED = "order.placed"
ORDER_FILLED = "order.filled"
ORDER_CANCELLED = "order.cancelled"
POSITION_OPENED = "position.opened"
POSITION_CLOSED = "position.closed"
RISK_LIMIT_HIT = "risk.limit_hit"
RISK_DAILY_STOP = "risk.daily_stop"
PORTFOLIO_UPDATED = "portfolio.updated"
MARKET_DATA_UPDATE = "market_data.update"
RECONCILIATION_UPDATE = "reconciliation.update"
SYSTEM_ERROR = "system.error"
