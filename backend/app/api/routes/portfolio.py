import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
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
async def get_performance(tracker: PerformanceTracker = Depends(get_performance_tracker)):
    return tracker.to_dict()


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
