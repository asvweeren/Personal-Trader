import asyncio

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import BrokerAdapter
from app.dependencies import get_broker, get_db, get_risk_manager
from app.models.risk_event import RiskEvent
from app.risk.manager import RiskManager

router = APIRouter()


class RiskLimitsUpdate(BaseModel):
    max_daily_loss_pct: float | None = None
    max_position_pct: float | None = None
    max_open_positions: int | None = None
    min_cash_reserve_pct: float | None = None


@router.get("/risk/metrics")
async def get_risk_metrics(
    risk_manager: RiskManager = Depends(get_risk_manager),
    broker: BrokerAdapter = Depends(get_broker),
):
    try:
        portfolio = await broker.get_portfolio()
        health = await risk_manager.check_portfolio_health(portfolio)

        # Calculate VaR if positions exist
        var_data = {}
        if portfolio.positions:
            try:
                from app.dependencies import get_data_pipeline
                from app.risk.var_calculator import VaRCalculator
                pipeline = get_data_pipeline()
                var_calc = VaRCalculator()
                var_result = await asyncio.wait_for(
                    var_calc.calculate_portfolio_var(
                        portfolio, pipeline._market_data
                    ),
                    timeout=3.0,
                )
                var_data = var_result.to_dict()
            except (asyncio.TimeoutError, Exception):
                pass

        return {
            "health": {
                "healthy": health.healthy,
                "checks": health.checks,
                "warnings": health.warnings,
                "daily_loss_pct": health.daily_loss_pct,
                "cash_reserve_pct": health.cash_reserve_pct,
                "position_count": health.position_count,
                "max_drawdown_pct": health.max_drawdown_pct,
                "sector_exposure": health.sector_exposure,
                "largest_position_pct": health.largest_position_pct,
                "market_open": health.market_open,
                "var_95": var_data.get("var_95", 0.0),
                "var_99": var_data.get("var_99", 0.0),
                "cvar_95": var_data.get("cvar_95", 0.0),
            },
            "limits": risk_manager.get_limits(),
            "daily_loss_triggered": risk_manager.daily_loss_triggered,
            "var": var_data,
        }
    except Exception:
        return {
            "health": {
                "healthy": False,
                "checks": {"broker_connected": False},
                "warnings": ["Broker not connected"],
                "daily_loss_pct": 0.0,
                "cash_reserve_pct": 100.0,
                "position_count": 0,
                "max_drawdown_pct": 0.0,
                "sector_exposure": {},
                "largest_position_pct": 0.0,
                "market_open": False,
            },
            "limits": risk_manager.get_limits(),
            "daily_loss_triggered": risk_manager.daily_loss_triggered,
            "broker_connected": False,
        }


@router.put("/risk/limits")
async def update_risk_limits(
    update: RiskLimitsUpdate,
    risk_manager: RiskManager = Depends(get_risk_manager),
):
    if update.max_daily_loss_pct is not None:
        risk_manager.max_daily_loss_pct = update.max_daily_loss_pct
    if update.max_position_pct is not None:
        risk_manager.max_position_pct = update.max_position_pct
    if update.max_open_positions is not None:
        risk_manager.max_open_positions = update.max_open_positions
    if update.min_cash_reserve_pct is not None:
        risk_manager.min_cash_reserve_pct = update.min_cash_reserve_pct
    return risk_manager.get_limits()


@router.get("/risk/events")
async def get_risk_events(
    limit: int = Query(default=50, le=200),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(RiskEvent)
        .order_by(desc(RiskEvent.timestamp))
        .limit(limit)
    )
    events = result.scalars().all()
    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "severity": e.severity,
            "symbol": e.symbol,
            "description": e.description,
            "action_taken": e.action_taken,
            "portfolio_value": e.portfolio_value,
            "daily_loss_pct": e.daily_loss_pct,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
        }
        for e in events
    ]
