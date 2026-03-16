import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.auth import get_current_user
from app.api.auth import router as auth_router
from app.api.routes import (
    backtest,
    models,
    portfolio,
    risk,
    screener,
    strategy,
    system,
    trades,
    validation,
)
from app.api.websocket import broadcast_update
from app.api.websocket import router as ws_router
from app.config import settings
from app.core.event_bus import (
    ORDER_FILLED,
    PORTFOLIO_UPDATED,
    POSITION_CLOSED,
    RISK_DAILY_STOP,
    SIGNAL_GENERATED,
    event_bus,
)
from app.core.scheduler import (
    schedule_broker_watchdog,
    schedule_daily_reset,
    schedule_daily_screener,
    schedule_daily_validation_report,
    schedule_data_pipeline,
    schedule_economic_calendar,
    schedule_eod_safety_close,
    schedule_heartbeat,
    schedule_snapshot_cleanup,
    schedule_trading_engine,
    schedule_weekly_model_retrain,
    start_scheduler,
    stop_scheduler,
)
from sqlalchemy import select

from app.dependencies import (
    get_broker,
    get_data_pipeline,
    get_performance_tracker,
    get_startup_symbols,
    init_trading_engine,
)
from app.models.database import async_session as session_factory
from app.models.trade import Trade, TradeStatus
from app.monitoring.logger import setup_logging

logger = structlog.get_logger()


async def _ws_forward(data):
    """Forward event bus events to WebSocket clients."""
    await broadcast_update(data.get("_event_type", "update"), data)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(debug=settings.debug)
    logger.info(
        "app.starting",
        env=settings.app_env,
        paper_trading=settings.ibkr_paper_trading,
    )

    # Subscribe event bus to WS broadcasting
    for evt in (
        SIGNAL_GENERATED, ORDER_FILLED, PORTFOLIO_UPDATED,
        RISK_DAILY_STOP, POSITION_CLOSED,
    ):
        event_bus.subscribe(evt, _ws_forward)

    # Start scheduler
    start_scheduler()
    schedule_daily_validation_report()
    schedule_daily_screener()
    schedule_heartbeat()
    schedule_snapshot_cleanup()
    schedule_economic_calendar()
    schedule_weekly_model_retrain()

    # Try to start broker, pipeline and trading engine (each step non-fatal)
    pipeline = None
    engine = None
    db = None
    broker = None

    # Step 1: Connect broker (with timeout so app starts even if IBKR is down)
    try:
        broker = get_broker()
        await asyncio.wait_for(broker.connect(), timeout=90)
        logger.info("broker.connected")
    except asyncio.TimeoutError:
        logger.warning("startup.broker_timeout", hint="IBKR connect timed out after 90s")
        broker = get_broker()  # keep broker object for watchdog reconnect
    except Exception as e:
        logger.warning("startup.broker_unavailable", error=str(e))
        logger.info("app.running_without_broker", hint="Configure IBKR credentials and restart")

    # Always schedule the broker watchdog (reconnects even when engine is not running)
    if broker:
        schedule_broker_watchdog(broker)

    # Step 2: Start data pipeline (requires broker)
    if broker and await broker.is_connected():
        try:
            symbols = await get_startup_symbols()
            pipeline = get_data_pipeline()
            await asyncio.wait_for(pipeline.start(symbols), timeout=60)
            logger.info("pipeline.started", symbols=len(symbols))
            schedule_data_pipeline(pipeline)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("startup.pipeline_error", error=str(e))

    # Step 3: Start trading engine (requires broker)
    if broker and await broker.is_connected():
        try:
            db = session_factory()
            engine = await init_trading_engine(db)
            await asyncio.wait_for(engine.start(), timeout=60)
            logger.info("engine.started")
            schedule_trading_engine(engine)
            schedule_daily_reset(engine)
            schedule_eod_safety_close(engine)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("startup.engine_error", error=str(e))

    # Step 4: Restore performance metrics from historical trades
    try:
        async with session_factory() as restore_db:
            result = await restore_db.execute(
                select(Trade).where(Trade.status == TradeStatus.CLOSED)
            )
            closed_trades = result.scalars().all()
            if closed_trades:
                tracker = get_performance_tracker()
                trade_dicts = [
                    {
                        "realized_pnl": t.realized_pnl,
                        "commission": t.commission,
                        "strategy_name": t.strategy_name,
                        "created_at": t.created_at,
                        "closed_at": t.closed_at,
                    }
                    for t in closed_trades
                ]
                tracker.restore_from_trades(trade_dicts)

                # Set daily_start_value to current total value (not initial_capital)
                if broker and await broker.is_connected():
                    try:
                        portfolio = await broker.get_portfolio()
                        tracker.daily_start_value = portfolio.account_summary.total_value
                        tracker.peak_value = max(tracker.peak_value, tracker.total_value)
                    except Exception:
                        tracker.daily_start_value = tracker.total_value
                else:
                    tracker.daily_start_value = tracker.total_value
    except Exception as e:
        logger.warning("startup.performance_restore_error", error=str(e))

    yield

    # Shutdown
    stop_scheduler()
    if engine:
        try:
            await engine.stop()
        except Exception:
            pass
    if pipeline:
        try:
            await pipeline.stop()
        except Exception:
            pass
    if broker:
        try:
            await broker.disconnect()
        except Exception:
            pass
    if db:
        await db.close()
    logger.info("app.shutdown")


app = FastAPI(
    title="AI Trader",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trader.edgedigital.nl",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth routes (no authentication required)
app.include_router(auth_router, prefix="/api", tags=["auth"])

# REST API routes (authentication required)
_auth = [Depends(get_current_user)]
app.include_router(portfolio.router, prefix="/api", tags=["portfolio"], dependencies=_auth)
app.include_router(trades.router, prefix="/api", tags=["trades"], dependencies=_auth)
app.include_router(strategy.router, prefix="/api", tags=["strategy"], dependencies=_auth)
app.include_router(risk.router, prefix="/api", tags=["risk"], dependencies=_auth)
app.include_router(backtest.router, prefix="/api", tags=["backtest"], dependencies=_auth)
app.include_router(system.router, prefix="/api", tags=["system"], dependencies=_auth)
app.include_router(validation.router, prefix="/api", tags=["validation"], dependencies=_auth)
app.include_router(screener.router, prefix="/api", tags=["screener"], dependencies=_auth)
app.include_router(models.router, prefix="/api", tags=["models"], dependencies=_auth)

# WebSocket (token validated inside the handler)
app.include_router(ws_router)
