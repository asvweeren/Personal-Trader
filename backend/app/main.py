from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import portfolio, trades, strategy, risk, backtest, system, validation
from app.api.websocket import router as ws_router, broadcast_update
from app.config import settings
from app.core.event_bus import (
    event_bus,
    SIGNAL_GENERATED,
    ORDER_FILLED,
    PORTFOLIO_UPDATED,
    RISK_DAILY_STOP,
    POSITION_CLOSED,
)
from app.core.scheduler import (
    start_scheduler,
    stop_scheduler,
    schedule_data_pipeline,
    schedule_trading_engine,
    schedule_daily_validation_report,
)
from app.dependencies import get_data_pipeline, init_trading_engine
from app.models.database import async_session as session_factory
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
    for evt in (SIGNAL_GENERATED, ORDER_FILLED, PORTFOLIO_UPDATED, RISK_DAILY_STOP, POSITION_CLOSED):
        event_bus.subscribe(evt, _ws_forward)

    # Start scheduler
    start_scheduler()
    schedule_daily_validation_report()

    # Try to start data pipeline and trading engine (non-fatal if broker unavailable)
    pipeline = None
    engine = None
    db = None
    try:
        pipeline = get_data_pipeline()
        await pipeline.start(settings.symbols_list)
        logger.info("pipeline.started")

        db = session_factory()
        engine = await init_trading_engine(db)
        await engine.start()
        logger.info("engine.started")

        schedule_data_pipeline(pipeline)
        schedule_trading_engine(engine)
    except Exception as e:
        logger.warning("startup.broker_unavailable", error=str(e))
        logger.info("app.running_without_broker", hint="Configure IBKR credentials and restart")

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
    if db:
        await db.close()
    logger.info("app.shutdown")


app = FastAPI(
    title="AI Trader",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API routes
app.include_router(portfolio.router, prefix="/api", tags=["portfolio"])
app.include_router(trades.router, prefix="/api", tags=["trades"])
app.include_router(strategy.router, prefix="/api", tags=["strategy"])
app.include_router(risk.router, prefix="/api", tags=["risk"])
app.include_router(backtest.router, prefix="/api", tags=["backtest"])
app.include_router(system.router, prefix="/api", tags=["system"])
app.include_router(validation.router, prefix="/api", tags=["validation"])

# WebSocket
app.include_router(ws_router)
