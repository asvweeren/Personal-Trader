from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
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
async def health_check(broker: BrokerAdapter = Depends(get_broker)):
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

    return {
        "status": "healthy" if broker_connected else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "broker": {"status": "connected" if broker_connected else "disconnected"},
            "database": {"status": "connected"},
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
