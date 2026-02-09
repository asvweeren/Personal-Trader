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
