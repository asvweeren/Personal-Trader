from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import structlog

logger = structlog.get_logger()

scheduler = AsyncIOScheduler()


def start_scheduler() -> None:
    if not scheduler.running:
        scheduler.start()
        logger.info("scheduler.started")


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("scheduler.stopped")


def schedule_data_pipeline(pipeline) -> None:
    """Register data pipeline jobs on the scheduler."""

    # Refresh technical features every 5 minutes during market hours
    scheduler.add_job(
        pipeline.refresh_features,
        IntervalTrigger(minutes=5),
        id="refresh_features",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="refresh_features", interval="5min")

    # Refresh sentiment every 15 minutes
    scheduler.add_job(
        pipeline.refresh_sentiment,
        IntervalTrigger(minutes=15),
        id="refresh_sentiment",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="refresh_sentiment", interval="15min")

    # Refresh historical data daily at 08:00 UTC (before EU market open)
    scheduler.add_job(
        pipeline.refresh_historical_data,
        CronTrigger(hour=8, minute=0),
        id="refresh_historical",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="refresh_historical", trigger="cron(08:00)")


def schedule_trading_engine(engine) -> None:
    """Register trading engine cycle on the scheduler."""
    scheduler.add_job(
        engine.run_cycle,
        IntervalTrigger(minutes=5),
        id="trading_cycle",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="trading_cycle", interval="5min")


def schedule_heartbeat() -> None:
    """Send periodic system status over WebSocket so clients stay informed."""

    async def send_heartbeat():
        from datetime import datetime, timezone
        from app.api.websocket import broadcast_update
        from app.dependencies import get_broker

        try:
            broker = get_broker()
            try:
                connected = await broker.is_connected()
            except Exception:
                connected = False

            engine_state = "not_initialized"
            try:
                from app.dependencies import get_trading_engine
                engine = get_trading_engine()
                engine_state = engine.state.value
            except RuntimeError:
                pass

            await broadcast_update("system.heartbeat", {
                "broker_connected": connected,
                "engine_state": engine_state,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass  # Don't let heartbeat failures break the scheduler

    scheduler.add_job(
        send_heartbeat,
        IntervalTrigger(seconds=30),
        id="ws_heartbeat",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="ws_heartbeat", interval="30s")


def schedule_daily_validation_report() -> None:
    """Schedule the daily paper-trading validation report.

    Runs at 21:00 UTC (after US market close) to summarise
    the day's paper-trading performance.
    """
    from app.monitoring.daily_reporter import DailyReporter

    scheduler.add_job(
        DailyReporter.scheduled_daily_report,
        CronTrigger(hour=21, minute=0),
        id="daily_validation_report",
        replace_existing=True,
        max_instances=1,
    )
    logger.info(
        "scheduler.job_added",
        job="daily_validation_report",
        trigger="cron(21:00)",
    )
