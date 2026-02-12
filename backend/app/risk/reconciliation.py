"""Position reconciliation: ensures internal state matches broker reality."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import Portfolio
from app.models.trade import Trade, TradeStatus
from app.monitoring.alerts import send_alert

logger = structlog.get_logger()


@dataclass
class ReconciliationResult:
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    matches: list[str] = field(default_factory=list)
    mismatches: list[dict] = field(default_factory=list)
    orphaned_broker: list[dict] = field(default_factory=list)
    orphaned_internal: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.mismatches and not self.orphaned_broker and not self.orphaned_internal

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "is_clean": self.is_clean,
            "matches": self.matches,
            "mismatches": self.mismatches,
            "orphaned_broker": self.orphaned_broker,
            "orphaned_internal": self.orphaned_internal,
            "match_count": len(self.matches),
            "mismatch_count": len(self.mismatches),
            "orphaned_broker_count": len(self.orphaned_broker),
            "orphaned_internal_count": len(self.orphaned_internal),
        }


async def reconcile(
    engine_trades: dict[str, Trade],
    portfolio: Portfolio,
) -> ReconciliationResult:
    """Compare engine's internal open trades with broker's actual positions.

    Returns a ReconciliationResult describing matches, mismatches, and orphans.
    """
    result = ReconciliationResult()

    # Build broker positions map: symbol -> quantity
    broker_positions: dict[str, int] = {}
    for pos in portfolio.positions:
        if pos.quantity > 0:
            broker_positions[pos.symbol] = pos.quantity

    internal_symbols = set(engine_trades.keys())
    broker_symbols = set(broker_positions.keys())

    # Matching symbols — check quantities
    for symbol in internal_symbols & broker_symbols:
        internal_qty = engine_trades[symbol].quantity
        broker_qty = broker_positions[symbol]
        if internal_qty == broker_qty:
            result.matches.append(symbol)
        else:
            result.mismatches.append({
                "symbol": symbol,
                "internal_qty": internal_qty,
                "broker_qty": broker_qty,
                "action": "sync_to_broker",
            })

    # Orphaned in broker (broker has position, engine doesn't know about it)
    for symbol in broker_symbols - internal_symbols:
        result.orphaned_broker.append({
            "symbol": symbol,
            "broker_qty": broker_positions[symbol],
            "action": "add_to_internal",
        })

    # Orphaned in engine (engine thinks it has a position, broker doesn't)
    for symbol in internal_symbols - broker_symbols:
        result.orphaned_internal.append(symbol)

    return result


async def auto_fix(
    result: ReconciliationResult,
    engine,
    db: AsyncSession,
) -> list[str]:
    """Attempt to fix reconciliation issues by syncing internal state to broker reality.

    Returns a list of actions taken.
    """
    actions: list[str] = []

    # Fix quantity mismatches: trust broker
    for mismatch in result.mismatches:
        symbol = mismatch["symbol"]
        broker_qty = mismatch["broker_qty"]
        trade = engine._open_trades.get(symbol)
        if trade:
            old_qty = trade.quantity
            trade.quantity = broker_qty
            actions.append(
                f"Synced {symbol} quantity: {old_qty} -> {broker_qty}"
            )
            logger.warning(
                "reconciliation.quantity_fixed",
                symbol=symbol,
                old=old_qty,
                new=broker_qty,
            )

    # Orphaned internal: mark as closed (broker doesn't have them)
    for symbol in result.orphaned_internal:
        trade = engine._open_trades.pop(symbol, None)
        if trade:
            trade.status = TradeStatus.CLOSED
            trade.closed_at = datetime.now(timezone.utc)
            actions.append(f"Closed orphaned internal trade: {symbol}")
            logger.warning(
                "reconciliation.orphaned_internal_closed",
                symbol=symbol,
                trade_id=trade.id,
            )

    if actions:
        try:
            await db.flush()
        except Exception:
            logger.exception("reconciliation.flush_error")

    return actions


# Module-level storage for the latest result
_last_result: ReconciliationResult | None = None


def get_last_result() -> ReconciliationResult | None:
    return _last_result


def set_last_result(result: ReconciliationResult) -> None:
    global _last_result
    _last_result = result
