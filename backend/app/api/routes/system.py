from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import BrokerAdapter
from app.config import settings
from app.data.pipeline import DataPipeline
from app.execution.engine import TradingEngine
from app.dependencies import get_broker, get_data_pipeline, get_trading_engine, get_db
from app.models.strategy_config import StrategyConfig
from app.monitoring.alerts import send_alert
from app.monitoring.performance import PerformanceTracker
from app.dependencies import get_performance_tracker

router = APIRouter()


@router.get("/system/health")
async def health_check(
    broker: BrokerAdapter = Depends(get_broker),
    db: AsyncSession = Depends(get_db),
):
    try:
        broker_connected = await broker.is_connected()
    except Exception:
        broker_connected = False

    engine_status = "unknown"
    try:
        engine = get_trading_engine()
        engine_status = engine.state.value
    except RuntimeError:
        engine_status = "not_initialized"

    # Database check
    db_connected = False
    try:
        await db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        pass

    # Redis check
    redis_connected = False
    try:
        r = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        redis_connected = True
    except Exception:
        pass

    all_ok = broker_connected and db_connected and redis_connected
    return {
        "status": "healthy" if all_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "broker": {"status": "connected" if broker_connected else "disconnected"},
            "database": {"status": "connected" if db_connected else "disconnected"},
            "redis": {"status": "connected" if redis_connected else "disconnected"},
            "trading_engine": {"status": engine_status},
            "environment": settings.app_env,
            "paper_trading": settings.ibkr_paper_trading,
        },
    }


@router.get("/system/pipeline")
async def pipeline_status(pipeline: DataPipeline = Depends(get_data_pipeline)):
    return pipeline.get_status()


@router.get("/system/engine")
async def engine_status():
    try:
        engine = get_trading_engine()
        return engine.get_status()
    except RuntimeError:
        return {
            "state": "NOT_INITIALIZED",
            "trading_enabled": False,
            "cycle_count": 0,
            "last_cycle_at": None,
            "open_trades": 0,
            "pending_orders": 0,
            "symbols": [],
            "strategies": [],
            "reconnect_attempts": 0,
        }


class TradingToggle(BaseModel):
    enabled: bool


@router.put("/system/engine/trading")
async def toggle_trading(body: TradingToggle, db: AsyncSession = Depends(get_db)):
    try:
        engine = get_trading_engine()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")

    engine.trading_enabled = body.enabled

    # Persist to DB
    result = await db.execute(select(StrategyConfig).limit(1))
    config = result.scalar_one_or_none()
    if config:
        config.trading_enabled = body.enabled
        await db.commit()

    return {"trading_enabled": engine.trading_enabled, "state": engine.state.value}


@router.post("/system/test-alert")
async def test_alert():
    """Send a test alert via all configured channels."""
    await send_alert(
        title="Test Alert",
        message="Dit is een test alert vanuit het AI Trading systeem. Als je dit ontvangt, werken de alerts correct.",
        critical=True,
    )
    return {"status": "sent", "channels": {"telegram": bool(settings.telegram_bot_token), "email": bool(settings.smtp_user)}}


@router.get("/system/reconciliation")
async def get_reconciliation():
    """Get the latest position reconciliation result."""
    from app.risk.reconciliation import get_last_result
    result = get_last_result()
    if result is None:
        return {"status": "no_data", "message": "No reconciliation has run yet"}
    return result.to_dict()


@router.get("/system/regime")
async def get_regime():
    """Get the current market regime detection result."""
    try:
        engine = get_trading_engine()
        regime = engine._regime_detector.current_regime
        if regime:
            return regime.to_dict()
        return {"regime": "unknown", "message": "No regime detected yet"}
    except RuntimeError:
        return {"regime": "unknown", "message": "Engine not initialized"}


@router.get("/system/calendar")
async def get_calendar():
    """Get upcoming economic events."""
    from app.data.economic_calendar import get_economic_calendar
    calendar = get_economic_calendar()
    events = await calendar.fetch_events(days_ahead=7)
    return {
        "events": [e.to_dict() for e in events],
        "total": len(events),
    }
