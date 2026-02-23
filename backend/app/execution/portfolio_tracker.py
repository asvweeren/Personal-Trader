"""Portfolio state management: position tracking, P&L calculation, and snapshots."""

from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import BrokerAdapter, Portfolio
from app.config import settings
from app.core.event_bus import event_bus, PORTFOLIO_UPDATED, POSITION_CLOSED
from app.models.portfolio_snapshot import PortfolioSnapshot
from app.models.trade import Trade, TradeStatus
from app.monitoring.alerts import send_alert
from app.monitoring.performance import PerformanceTracker

logger = structlog.get_logger()


class PortfolioTracker:
    """Tracks portfolio state and persists snapshots."""

    def __init__(
        self,
        broker: BrokerAdapter,
        db: AsyncSession,
        performance: PerformanceTracker | None = None,
    ):
        self._broker = broker
        self._db = db
        self._performance = performance
        self._last_portfolio: Portfolio | None = None
        self._daily_start_value: float | None = None

    async def get_current(self) -> Portfolio:
        """Get current portfolio from broker."""
        portfolio = await self._broker.get_portfolio()
        self._last_portfolio = portfolio

        if self._performance:
            self._performance.update_unrealized(
                portfolio.account_summary.unrealized_pnl
            )

        return portfolio

    async def initialize_daily(self) -> float:
        """Initialize daily tracking. Returns start-of-day portfolio value."""
        portfolio = await self.get_current()
        self._daily_start_value = portfolio.account_summary.total_value

        if self._performance:
            self._performance.reset_daily()

        logger.info(
            "portfolio.daily_init",
            value=self._daily_start_value,
        )
        return self._daily_start_value

    def get_daily_pnl(self) -> float:
        """Calculate P&L since start of day.

        Uses realized trades P&L + unrealized P&L change when performance
        tracker is available, falls back to total_value difference otherwise.
        """
        if self._daily_start_value is None or self._last_portfolio is None:
            return 0.0

        # Prefer performance-based calculation (handles deposits/withdrawals)
        if self._performance:
            realized_today = self._performance.realized_pnl
            unrealized = self._last_portfolio.account_summary.unrealized_pnl
            return realized_today + unrealized

        return self._last_portfolio.account_summary.total_value - self._daily_start_value

    async def record_trade_close(
        self,
        trade: Trade,
        exit_price: float,
        commission: float = 0.0,
    ) -> float:
        """Record a closed trade and calculate realized P&L."""
        if trade.entry_price is None:
            logger.warning("portfolio.no_entry_price", trade_id=trade.id)
            return 0.0

        if trade.side.value == "BUY":
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity

        pnl -= commission

        trade.exit_price = exit_price
        trade.realized_pnl = round(pnl, 2)
        trade.commission = commission
        trade.status = TradeStatus.CLOSED
        trade.closed_at = datetime.now(timezone.utc)

        # Calculate hold duration
        hold_duration_minutes = 0.0
        if trade.created_at and trade.closed_at:
            delta = trade.closed_at - trade.created_at
            hold_duration_minutes = delta.total_seconds() / 60

        if self._performance:
            self._performance.record_trade(
                pnl,
                commission,
                strategy_name=trade.strategy_name,
                hold_duration_minutes=hold_duration_minutes,
            )
            if self._performance.should_alert_consecutive_losses(
                settings.consecutive_loss_alert_threshold
            ):
                await send_alert(
                    "Consecutive Loss Alert",
                    f"{self._performance.consecutive_losses} consecutive losing trades.\n"
                    f"Total realized P&L: {self._performance.realized_pnl:+.2f}\n"
                    f"Win rate: {self._performance.win_rate:.1f}%",
                    critical=True,
                )

        await event_bus.publish(POSITION_CLOSED, {
            "trade_id": trade.id,
            "symbol": trade.symbol,
            "pnl": round(pnl, 2),
            "exit_price": exit_price,
        })

        logger.info(
            "portfolio.trade_closed",
            trade_id=trade.id,
            symbol=trade.symbol,
            pnl=round(pnl, 2),
            entry=trade.entry_price,
            exit=exit_price,
        )

        return pnl

    async def take_snapshot(self) -> PortfolioSnapshot:
        """Take a snapshot of the current portfolio and persist to DB."""
        portfolio = await self.get_current()

        positions_detail = [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "market_price": p.market_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in portfolio.positions
        ]

        daily_pnl = self.get_daily_pnl()

        snapshot = PortfolioSnapshot(
            total_value=portfolio.account_summary.total_value,
            cash=portfolio.account_summary.cash,
            positions_value=sum(p.market_value for p in portfolio.positions),
            unrealized_pnl=portfolio.account_summary.unrealized_pnl,
            realized_pnl=portfolio.account_summary.realized_pnl,
            daily_pnl=round(daily_pnl, 2),
            positions_detail=positions_detail,
        )

        self._db.add(snapshot)
        await self._db.flush()

        await event_bus.publish(PORTFOLIO_UPDATED, {
            "total_value": snapshot.total_value,
            "cash": snapshot.cash,
            "positions": len(portfolio.positions),
            "daily_pnl": snapshot.daily_pnl,
            "unrealized_pnl": snapshot.unrealized_pnl,
        })

        logger.info(
            "portfolio.snapshot",
            total_value=snapshot.total_value,
            cash=snapshot.cash,
            positions=len(portfolio.positions),
            daily_pnl=snapshot.daily_pnl,
        )

        return snapshot

    def get_status(self) -> dict:
        """Return current portfolio tracker status."""
        if self._last_portfolio is None:
            return {"initialized": False}

        return {
            "initialized": True,
            "total_value": self._last_portfolio.account_summary.total_value,
            "cash": self._last_portfolio.account_summary.cash,
            "position_count": len(self._last_portfolio.positions),
            "daily_start_value": self._daily_start_value,
            "daily_pnl": self.get_daily_pnl(),
        }
