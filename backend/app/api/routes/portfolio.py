import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import BrokerAdapter
from app.config import settings
from app.dependencies import get_broker, get_db, get_performance_tracker
from app.models.portfolio_snapshot import PortfolioSnapshot
from app.monitoring.performance import PerformanceTracker

logger = structlog.get_logger()
router = APIRouter()


@router.get("/portfolio")
async def get_portfolio(broker: BrokerAdapter = Depends(get_broker)):
    try:
        portfolio = await broker.get_portfolio()
        return {
            "total_value": portfolio.account_summary.total_value,
            "cash": portfolio.account_summary.cash,
            "buying_power": portfolio.account_summary.buying_power,
            "unrealized_pnl": portfolio.account_summary.unrealized_pnl,
            "realized_pnl": portfolio.account_summary.realized_pnl,
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "market_price": p.market_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for p in portfolio.positions
            ],
        }
    except Exception:
        logger.warning("portfolio.broker_unavailable")
        return {
            "total_value": settings.initial_capital,
            "cash": settings.initial_capital,
            "buying_power": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "positions": [],
            "broker_connected": False,
        }


@router.get("/positions")
async def get_positions(broker: BrokerAdapter = Depends(get_broker)):
    try:
        positions = await broker.get_positions()
        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "market_price": p.market_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in positions
        ]
    except Exception:
        logger.warning("positions.broker_unavailable")
        return []


@router.get("/performance")
async def get_performance(
    broker: BrokerAdapter = Depends(get_broker),
    tracker: PerformanceTracker = Depends(get_performance_tracker),
):
    data = tracker.to_dict()
    # Override with actual broker values for accuracy
    try:
        portfolio = await broker.get_portfolio()
        actual_value = portfolio.account_summary.total_value
        initial = tracker.initial_capital
        data["total_value"] = round(actual_value, 2)
        data["unrealized_pnl"] = round(
            portfolio.account_summary.unrealized_pnl, 2
        )
        data["realized_pnl"] = round(
            portfolio.account_summary.realized_pnl, 2
        )
        if initial > 0:
            data["total_return_pct"] = round(
                (actual_value - initial) / initial * 100, 2
            )
            data["daily_pnl"] = round(
                actual_value - tracker.daily_start_value, 2
            )
            data["daily_return_pct"] = round(
                (actual_value - tracker.daily_start_value)
                / tracker.daily_start_value * 100, 2
            ) if tracker.daily_start_value > 0 else 0.0
        # Update drawdown from actual value
        if actual_value > tracker.peak_value:
            tracker.peak_value = actual_value
        dd = (
            (tracker.peak_value - actual_value)
            / tracker.peak_value * 100
            if tracker.peak_value > 0 else 0.0
        )
        data["max_drawdown"] = round(max(dd, tracker.max_drawdown), 2)
    except Exception:
        logger.warning("performance.broker_unavailable_for_metrics")
    return data


@router.get("/portfolio/snapshots")
async def get_snapshots(
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(PortfolioSnapshot)
        .order_by(desc(PortfolioSnapshot.timestamp))
        .limit(limit)
    )
    snapshots = result.scalars().all()

    return [
        {
            "id": s.id,
            "total_value": s.total_value,
            "cash": s.cash,
            "positions_value": s.positions_value,
            "unrealized_pnl": s.unrealized_pnl,
            "realized_pnl": s.realized_pnl,
            "daily_pnl": s.daily_pnl,
            "timestamp": s.timestamp.isoformat() if s.timestamp else None,
        }
        for s in reversed(snapshots)  # Chronological order
    ]
