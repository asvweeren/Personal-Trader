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
        misfire_grace_time=3600,
    )
    logger.info(
        "scheduler.job_added",
        job="daily_validation_report",
        trigger="cron(21:00)",
        misfire_grace_time=3600,
    )


def schedule_snapshot_cleanup() -> None:
    """Clean up old portfolio snapshots to prevent unbounded growth.

    Runs daily at 02:00 UTC:
    - Keeps all snapshots from the last 7 days
    - Keeps 1 snapshot per day for days 7-90
    - Deletes everything older than 90 days
    """

    async def cleanup_snapshots():
        from datetime import datetime, timezone, timedelta
        from sqlalchemy import text
        from app.models.database import async_session

        try:
            async with async_session() as session:
                now = datetime.now(timezone.utc)
                cutoff_90d = now - timedelta(days=90)
                cutoff_7d = now - timedelta(days=7)

                # 1. Delete everything older than 90 days
                result_old = await session.execute(
                    text("DELETE FROM portfolio_snapshots WHERE timestamp < :cutoff"),
                    {"cutoff": cutoff_90d},
                )
                deleted_old = result_old.rowcount

                # 2. Downsample 7-90 day range: keep only the last snapshot per day
                result_dup = await session.execute(
                    text("""
                        DELETE FROM portfolio_snapshots
                        WHERE timestamp >= :cutoff_90d
                          AND timestamp < :cutoff_7d
                          AND id NOT IN (
                            SELECT DISTINCT ON (timestamp::date) id
                            FROM portfolio_snapshots
                            WHERE timestamp >= :cutoff_90d
                              AND timestamp < :cutoff_7d
                            ORDER BY timestamp::date, timestamp DESC
                          )
                    """),
                    {"cutoff_90d": cutoff_90d, "cutoff_7d": cutoff_7d},
                )
                deleted_dup = result_dup.rowcount

                await session.commit()

                total = deleted_old + deleted_dup
                if total > 0:
                    logger.info(
                        "snapshots.cleanup",
                        deleted_old=deleted_old,
                        deleted_duplicates=deleted_dup,
                        total_deleted=total,
                    )

        except Exception:
            logger.exception("snapshots.cleanup_error")

    scheduler.add_job(
        cleanup_snapshots,
        CronTrigger(hour=2, minute=0),
        id="snapshot_cleanup",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="snapshot_cleanup", trigger="cron(02:00)")


def schedule_broker_watchdog(broker) -> None:
    """Monitor broker connection and trigger reconnect if needed.

    Runs every 30 seconds, independent of the trading cycle.
    The broker adapter handles the actual reconnect with exponential backoff;
    this job ensures the connection state is checked even when the engine
    is not running cycles (e.g. outside market hours).
    """

    async def broker_watchdog():
        try:
            connected = await broker.is_connected()
            if not connected:
                logger.warning("watchdog.broker_offline")
                try:
                    await broker.connect()
                    logger.info("watchdog.broker_reconnected")
                except Exception:
                    pass  # Adapter auto-reconnect handles retries
        except Exception:
            pass  # Don't let watchdog failures break the scheduler

    scheduler.add_job(
        broker_watchdog,
        IntervalTrigger(seconds=30),
        id="broker_watchdog",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="broker_watchdog", interval="30s")


def schedule_daily_reset(engine) -> None:
    """Reset daily counters at 08:30 UTC Mon-Fri (before EU market open)."""

    async def daily_reset():
        try:
            engine.reset_daily()
        except Exception:
            logger.exception("scheduler.daily_reset_error")

    scheduler.add_job(
        daily_reset,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=30),
        id="daily_reset",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="daily_reset", trigger="cron(mon-fri 08:30)")


def schedule_eod_safety_close(engine) -> None:
    """Backup EOD close check every minute to catch positions between 5-min cycles.

    The main trading cycle runs every 5 minutes, but EOD close needs
    tighter timing to avoid overnight positions.
    """

    async def eod_safety_check():
        from app.risk.market_hours import is_any_market_open
        from app.config import settings

        try:
            if engine.state.value != "RUNNING" or not engine.trading_enabled:
                return
            if not engine._open_trades:
                return

            # Only run the check if any market is still open
            symbols = list(engine._open_trades.keys())
            if not is_any_market_open(symbols):
                return

            # Get fresh prices and run the EOD close check
            snapshot = await engine._market_data.get_snapshot(symbols)
            await engine._check_eod_close(snapshot.prices)
            await engine._db.commit()
        except Exception:
            logger.exception("scheduler.eod_safety_check_error")

    scheduler.add_job(
        eod_safety_check,
        IntervalTrigger(minutes=1),
        id="eod_safety_close",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="eod_safety_close", interval="1min")


def schedule_weekly_model_retrain() -> None:
    """Retrain the XGBoost model weekly (Sunday 02:00 UTC).

    Downloads 90 days of data via yfinance, trains a candidate model,
    and hot-swaps it if accuracy is >= 95% of the current model.
    """

    async def retrain_model():
        import json
        from pathlib import Path

        import pandas as pd

        from app.config import settings
        from app.monitoring.alerts import send_alert

        try:
            import yfinance as yf
        except ImportError:
            logger.error("scheduler.retrain_yfinance_missing")
            return

        from app.strategy.ml_strategy import MLStrategy, MODEL_DIR

        logger.info("scheduler.retrain_started")

        try:
            # 1. Download 90 days of data for all symbols
            symbols = settings.symbols_list
            frames = []
            for sym in symbols:
                try:
                    df = yf.download(sym, period="90d", interval="1d", progress=False)
                    if df.empty:
                        continue
                    df = df.copy()
                    df["symbol"] = sym
                    frames.append(df)
                except Exception:
                    logger.warning("scheduler.retrain_download_error", symbol=sym)

            if not frames:
                logger.error("scheduler.retrain_no_data")
                await send_alert(
                    "Model Retrain Failed",
                    "No historical data could be downloaded.",
                )
                return

            historical_data = pd.concat(frames)

            # 2. Train candidate
            strategy = MLStrategy()
            result = await strategy.train_candidate(historical_data)

            if not result.success:
                logger.error("scheduler.retrain_failed", message=result.message)
                await send_alert(
                    "Model Retrain Failed",
                    f"Training error: {result.message}",
                )
                return

            candidate_accuracy = result.metrics.get("test_accuracy", 0.0)

            # 3. Compare with current model
            meta_path = MODEL_DIR / "xgboost_model.json"
            current_accuracy = 0.0
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        current_meta = json.load(f)
                    current_accuracy = current_meta.get("test_accuracy", 0.0)
                except Exception:
                    pass

            threshold = current_accuracy * 0.95  # Must be >= 95% of current

            # 4. Swap if candidate meets threshold
            if candidate_accuracy >= threshold:
                swapped = strategy.hot_swap_model()
                if swapped:
                    msg = (
                        f"Model retrained and deployed.\n"
                        f"Old accuracy: {current_accuracy:.4f}\n"
                        f"New accuracy: {candidate_accuracy:.4f}"
                    )
                    logger.info("scheduler.retrain_swapped", old=current_accuracy, new=candidate_accuracy)
                    await send_alert("Model Retrain Success", msg)
                else:
                    await send_alert(
                        "Model Retrain Warning",
                        "Candidate trained but hot-swap failed. Rollback applied.",
                    )
            else:
                msg = (
                    f"Candidate rejected (below threshold).\n"
                    f"Current: {current_accuracy:.4f}\n"
                    f"Candidate: {candidate_accuracy:.4f}\n"
                    f"Threshold (95%): {threshold:.4f}"
                )
                logger.info("scheduler.retrain_rejected", current=current_accuracy, candidate=candidate_accuracy)
                await send_alert("Model Retrain Skipped", msg)

        except Exception:
            logger.exception("scheduler.retrain_error")
            try:
                await send_alert(
                    "Model Retrain Error",
                    "Unexpected error during weekly model retrain. Check logs.",
                    critical=True,
                )
            except Exception:
                pass

    scheduler.add_job(
        retrain_model,
        CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="weekly_model_retrain",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=7200,
    )
    logger.info(
        "scheduler.job_added",
        job="weekly_model_retrain",
        trigger="cron(sun 02:00)",
        misfire_grace_time=7200,
    )
