import asyncio
import time
import traceback
from datetime import UTC

import structlog
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = structlog.get_logger()

scheduler = AsyncIOScheduler()

# Jobs where failure is critical (affects trading)
_CRITICAL_JOBS = {"trading_cycle", "eod_safety_close", "daily_reset", "broker_watchdog"}

# High-frequency monitoring jobs — missed executions are expected and not alertable
_HIGH_FREQ_JOBS = {"broker_watchdog", "ws_heartbeat", "eod_safety_close"}


def _on_job_event(event) -> None:
    """Handle scheduler job errors and missed executions."""
    job_id = getattr(event, "job_id", "unknown")
    is_critical = job_id in _CRITICAL_JOBS

    if hasattr(event, "exception") and event.exception:
        tb = "".join(traceback.format_exception(
            type(event.exception),
            event.exception,
            event.exception.__traceback__,
        ))
        logger.error(
            "scheduler.job_error",
            job_id=job_id,
            error=str(event.exception),
            traceback=tb[:500],
            critical=is_critical,
        )
        try:
            from app.monitoring.alerts import send_alert
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(send_alert(
                    "Scheduler Job Failed" if is_critical else "Scheduler Job Warning",
                    f"Job: {job_id}\nError: {str(event.exception)[:300]}",
                    critical=is_critical,
                ))
        except Exception:
            logger.warning("scheduler.alert_send_failed", job_id=job_id)
    else:
        logger.warning("scheduler.job_missed", job_id=job_id, critical=is_critical)
        # Don't spam alerts for high-frequency monitoring jobs — missed windows
        # are expected during heavy event-loop load and are not actionable.
        if job_id in _HIGH_FREQ_JOBS:
            return
        try:
            from app.monitoring.alerts import send_alert
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(send_alert(
                    "Scheduler Job Missed",
                    f"Job '{job_id}' missed its scheduled execution window.",
                    critical=is_critical,
                ))
        except Exception:
            logger.warning("scheduler.alert_send_failed", job_id=job_id)


def start_scheduler() -> None:
    if not scheduler.running:
        scheduler.add_listener(_on_job_event, EVENT_JOB_ERROR | EVENT_JOB_MISSED)
        scheduler.start()
        logger.info("scheduler.started")


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("scheduler.stopped")


def schedule_data_pipeline(pipeline) -> None:
    """Register data pipeline jobs on the scheduler."""

    # Refresh technical features at cycle interval (matches trading cycle)
    from app.config import settings as cfg
    feat_interval = cfg.cycle_interval_minutes
    scheduler.add_job(
        pipeline.refresh_features,
        IntervalTrigger(minutes=feat_interval),
        id="refresh_features",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="refresh_features", interval=f"{feat_interval}min")

    # Refresh sentiment every 60 minutes (cost-optimized: ~15 symbols, market hours only)
    scheduler.add_job(
        pipeline.refresh_sentiment,
        IntervalTrigger(minutes=60),
        id="refresh_sentiment",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="refresh_sentiment", interval="60min")

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
    from app.config import settings as cfg
    cycle_interval = cfg.cycle_interval_minutes
    scheduler.add_job(
        engine.run_cycle,
        IntervalTrigger(minutes=cycle_interval),
        id="trading_cycle",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="trading_cycle", interval=f"{cycle_interval}min")


def schedule_heartbeat() -> None:
    """Send periodic system status over WebSocket so clients stay informed."""

    async def send_heartbeat():
        from datetime import datetime

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
                "timestamp": datetime.now(UTC).isoformat(),
            })
        except Exception:
            logger.warning("scheduler.heartbeat_error", exc_info=True)

    scheduler.add_job(
        send_heartbeat,
        IntervalTrigger(seconds=30),
        id="ws_heartbeat",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=60,
    )
    logger.info("scheduler.job_added", job="ws_heartbeat", interval="30s")


def schedule_economic_calendar() -> None:
    """Refresh the economic event calendar daily at 06:00 UTC."""

    async def refresh_calendar():
        try:
            from app.data.economic_calendar import get_economic_calendar
            calendar = get_economic_calendar()
            events = await calendar.fetch_events(days_ahead=7)
            logger.info("calendar.refreshed", events=len(events))
        except Exception:
            logger.exception("calendar.refresh_error")

    scheduler.add_job(
        refresh_calendar,
        CronTrigger(hour=6, minute=0),
        id="refresh_economic_calendar",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduler.job_added", job="refresh_economic_calendar", trigger="cron(06:00)")


def schedule_daily_screener() -> None:
    """Run the stock screener daily at 07:50 UTC Mon-Fri (before EU open).

    Selects top candidates from a broad universe and updates the engine/pipeline
    symbols so the system trades the best opportunities each day.
    """

    async def run_daily_screener():
        from datetime import date as date_type

        from app.config import settings as cfg
        from app.data.screener import StockScreener
        from app.models.database import async_session as session_factory
        from app.models.screening_result import ScreeningResult

        if not cfg.screener_enabled:
            logger.info("screener.disabled")
            return

        logger.info("screener.daily_run_starting")
        screener = StockScreener()

        try:
            data = await screener.run_screening()
        except Exception:
            logger.exception("screener.daily_run_failed")
            return

        # Persist to DB
        try:
            async with session_factory() as session:
                row = ScreeningResult(
                    screening_date=date_type.today(),
                    total_scanned=data["total_scanned"],
                    candidates=data["candidates"],
                    config=data["config"],
                )
                session.add(row)
                await session.commit()
        except Exception:
            logger.exception("screener.db_save_failed")

        # Update engine + pipeline with new symbols (filter EU when disabled)
        if data["candidates"]:
            from app.dependencies import should_skip_eu
            symbols = [c["symbol"] for c in data["candidates"] if not should_skip_eu(c["symbol"])]
            eu_filtered = len(data["candidates"]) - len(symbols)
            if eu_filtered:
                logger.info("screener.eu_symbols_filtered", removed=eu_filtered, remaining=len(symbols))
            try:
                from app.dependencies import get_trading_engine
                engine = get_trading_engine()
                engine.update_symbols(symbols)
                logger.info("screener.engine_updated", symbols=len(symbols))
            except RuntimeError:
                logger.debug("screener.engine_not_initialized")
            try:
                from app.dependencies import get_data_pipeline
                pipeline = get_data_pipeline()
                await pipeline.update_symbols(symbols)
                logger.info("screener.pipeline_updated", symbols=len(symbols))
            except Exception:
                logger.warning("screener.pipeline_update_failed")

    scheduler.add_job(
        run_daily_screener,
        CronTrigger(day_of_week="mon-fri", hour=7, minute=50),
        id="daily_screener",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    logger.info(
        "scheduler.job_added",
        job="daily_screener",
        trigger="cron(mon-fri 07:50)",
    )


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
        from datetime import datetime, timedelta

        from sqlalchemy import text

        from app.models.database import async_session

        try:
            async with async_session() as session:
                now = datetime.now(UTC)
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
    """Monitor broker connection, trigger reconnect, and lazy-start engine.

    Runs every 30 seconds, independent of the trading cycle.
    The broker adapter handles the actual reconnect with exponential backoff;
    this job ensures the connection state is checked even when the engine
    is not running cycles (e.g. outside market hours).

    When the broker comes online and the engine hasn't been initialized yet,
    the watchdog will start the data pipeline and trading engine automatically.
    """
    _initializing = {"active": False}
    _broker_down_since: dict[str, float | None] = {"ts": None}
    _alert_sent: dict[str, bool] = {"offline": False}

    async def broker_watchdog():
        try:
            # Bound the connectivity probe: a half-dead IBKR socket can make
            # is_connected() hang forever, which — with max_instances=1 — would
            # wedge every future watchdog run and leave a dropped connection
            # unrecovered (root cause of the silent multi-hour trading outages).
            try:
                connected = await asyncio.wait_for(broker.is_connected(), timeout=15)
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning("watchdog.is_connected_timeout")
                connected = False
            if not connected:
                now = time.monotonic()
                if _broker_down_since["ts"] is None:
                    _broker_down_since["ts"] = now

                reconnecting = getattr(broker, "_reconnecting", False)
                down_seconds = now - _broker_down_since["ts"]

                # Force-reset stuck reconnect loop after 3 minutes
                if reconnecting and down_seconds > 180:
                    logger.warning(
                        "watchdog.force_reconnect",
                        down_seconds=int(down_seconds),
                        message="Resetting stuck _reconnecting flag",
                    )
                    broker._reconnecting = False
                    reconnecting = False

                # Send one-time alert after 5 minutes offline
                if down_seconds > 300 and not _alert_sent["offline"]:
                    try:
                        from app.monitoring.alerts import send_alert
                        mins = int(down_seconds // 60)
                        asyncio.create_task(send_alert(
                            "Broker Offline",
                            f"IBKR offline sinds {mins} min. Auto-recovery actief.",
                        ))
                    except Exception:
                        logger.warning("watchdog.offline_alert_failed", exc_info=True)
                    _alert_sent["offline"] = True

                if not reconnecting:
                    logger.warning("watchdog.broker_offline", down_seconds=int(down_seconds))
                    try:
                        # IBKR connect can legitimately take ~30s; cap it so a
                        # stuck handshake cannot occupy the watchdog slot forever.
                        await asyncio.wait_for(broker.connect(), timeout=90)
                        logger.info("watchdog.broker_reconnected")
                        _broker_down_since["ts"] = None
                    except (TimeoutError, asyncio.TimeoutError):
                        logger.warning("watchdog.reconnect_timeout")
                    except Exception:
                        logger.warning("watchdog.reconnect_failed", exc_info=True)
                else:
                    logger.warning(
                        "watchdog.broker_offline",
                        reconnect_active=True,
                        down_seconds=int(down_seconds),
                    )
                return

            # Broker is connected — reset down timer
            if _broker_down_since["ts"] is not None:
                down_seconds = time.monotonic() - _broker_down_since["ts"]
                logger.info(
                    "watchdog.broker_recovered",
                    down_seconds=int(down_seconds),
                )
                # Send recovery alert if we previously sent an offline alert
                if _alert_sent["offline"]:
                    try:
                        from app.monitoring.alerts import send_alert
                        mins = int(down_seconds // 60)
                        asyncio.create_task(send_alert(
                            "Broker Hersteld",
                            f"IBKR hersteld na {mins} min offline.",
                        ))
                    except Exception:
                        logger.warning("watchdog.recovery_alert_failed", exc_info=True)
                    _alert_sent["offline"] = False
                _broker_down_since["ts"] = None

            # Broker is connected — check if engine needs lazy initialization
            if _initializing["active"]:
                return
            try:
                from app.dependencies import get_trading_engine
                get_trading_engine()  # Raises RuntimeError if not initialized
                return  # Engine already running, nothing to do
            except RuntimeError:
                pass  # Engine not initialized — start it

            # Run lazy init as a detached, guarded task rather than awaiting it
            # inside the watchdog job. Engine/pipeline startup awaits slow broker
            # RPCs; holding the (max_instances=1) watchdog slot open for its full
            # duration blocks every subsequent connectivity check. Detaching it
            # keeps the reconnect loop responsive; the timeout resets the guard so
            # a hung init retries on a later cycle instead of latching forever.
            _initializing["active"] = True

            async def _guarded_lazy_init():
                try:
                    await asyncio.wait_for(_lazy_init_engine(), timeout=240)
                except (TimeoutError, asyncio.TimeoutError):
                    logger.warning("watchdog.lazy_init_timeout")
                except Exception:
                    logger.warning("watchdog.lazy_init_error", exc_info=True)
                finally:
                    _initializing["active"] = False

            asyncio.create_task(_guarded_lazy_init())

        except Exception:
            logger.warning("scheduler.watchdog_error", exc_info=True)

    async def _lazy_init_engine():
        """Initialize pipeline + engine after broker becomes available."""
        from app.dependencies import get_data_pipeline, get_startup_symbols, init_trading_engine
        from app.models.database import async_session as session_factory

        logger.info("watchdog.lazy_init_starting")

        # Start data pipeline
        try:
            symbols = await get_startup_symbols()
            pipeline = get_data_pipeline()
            await pipeline.start(symbols)
            logger.info("watchdog.pipeline_started", symbols=len(symbols))
            schedule_data_pipeline(pipeline)
        except Exception as e:
            logger.warning("watchdog.pipeline_error", error=str(e))

        # Start trading engine
        try:
            db = session_factory()
            engine = await init_trading_engine(db)
            await engine.start()
            logger.info("watchdog.engine_started")
            schedule_trading_engine(engine)
            schedule_daily_reset(engine)
            schedule_eod_safety_close(engine)
        except Exception as e:
            logger.warning("watchdog.engine_error", error=str(e))

    scheduler.add_job(
        broker_watchdog,
        IntervalTrigger(seconds=10),
        id="broker_watchdog",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=30,
    )
    logger.info("scheduler.job_added", job="broker_watchdog", interval="10s")


def schedule_daily_reset(engine) -> None:
    """Reset daily counters at 08:30 UTC Mon-Fri (before EU market open)."""

    async def daily_reset():
        try:
            engine.reset_daily()
            # Update ensemble weights based on accumulated performance
            for strategy in engine._strategies:
                if hasattr(strategy, "update_weights_from_history"):
                    try:
                        new_weights = strategy.update_weights_from_history(engine._performance)
                        logger.info("scheduler.ensemble_weights_updated", weights=new_weights)
                    except Exception:
                        logger.debug("scheduler.ensemble_weight_update_skipped", exc_info=True)
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
        from app.config import settings as cfg
        from app.risk.market_hours import is_any_market_open

        try:
            if engine.state.value != "RUNNING" or not engine.trading_enabled:
                return

            # Continuous daily-loss halt — runs even in swing mode (EOD disabled)
            # and even with no open trades, since it also blocks new entries.
            try:
                await engine.check_daily_loss_halt()
            except Exception:
                logger.exception("scheduler.daily_loss_check_error")

            # Skip EOD close if disabled (swing trading mode)
            if not cfg.eod_close_enabled:
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
        misfire_grace_time=120,
    )
    logger.info("scheduler.job_added", job="eod_safety_close", interval="1min")


def schedule_weekly_model_retrain() -> None:
    """Retrain the XGBoost model weekly (Sunday 02:00 UTC).

    Downloads 90 days of data via yfinance, trains a candidate model,
    and hot-swaps it if accuracy is >= 95% of the current model.
    """

    async def retrain_model():
        import json

        import pandas as pd

        from app.config import settings
        from app.monitoring.alerts import send_alert

        try:
            import yfinance as yf
        except ImportError:
            logger.error("scheduler.retrain_yfinance_missing")
            return

        from app.strategy.ml_strategy import MODEL_DIR, MLStrategy

        logger.info("scheduler.retrain_started")

        try:
            # 1. Download 1 year of data for all symbols (more data = less overfitting)
            symbols = settings.symbols_list
            frames = []
            for sym in symbols:
                try:
                    df = yf.download(sym, period="1y", interval="1d", progress=False)
                    if df.empty:
                        continue
                    df = df.copy()
                    # Normalize yfinance column names to lowercase OHLCV
                    df = df.rename(columns={
                        "Open": "open", "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume",
                    })
                    keep = ["open", "high", "low", "close", "volume"]
                    df = df[[c for c in keep if c in df.columns]]
                    df.index.name = "timestamp"
                    df = df.reset_index()
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
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

            # Compute features PER SYMBOL to prevent rolling indicators
            # from crossing symbol boundaries, then concatenate.
            from app.data.indicators import compute_features
            from app.strategy.feature_pipeline import create_binary_target

            feature_dfs = []
            for frame in frames:
                feat_df = compute_features(frame)
                feat_df["target"] = create_binary_target(
                    feat_df, forward_periods=5,
                    buy_threshold=0.01,
                )
                feat_df = feat_df.dropna()
                if len(feat_df) >= 60:
                    feature_dfs.append(feat_df)

            if not feature_dfs:
                logger.error("scheduler.retrain_no_features")
                await send_alert(
                    "Model Retrain Failed",
                    "No symbols had enough data after feature computation.",
                )
                return

            historical_data = pd.concat(feature_dfs, ignore_index=True)
            historical_data = historical_data.sort_values("timestamp").reset_index(drop=True)
            logger.info(
                "scheduler.retrain_features",
                symbols=len(feature_dfs),
                samples=len(historical_data),
            )

            # 2. Train candidate (features already computed per-symbol)
            strategy = MLStrategy()
            result = await strategy.train_candidate(historical_data, features_precomputed=True)

            if not result.success:
                logger.error("scheduler.retrain_failed", message=result.message)
                await send_alert(
                    "Model Retrain Failed",
                    f"Training error: {result.message}",
                )
                return

            # Use profit_score for binary models, test_accuracy as fallback
            candidate_score = result.metrics.get(
                "test_profit_score",
                result.metrics.get("test_accuracy", 0.0),
            )

            # 3. Compare with current model
            meta_path = MODEL_DIR / "xgboost_model.json"
            current_score = 0.0
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        current_meta = json.load(f)
                    current_score = current_meta.get(
                        "test_profit_score",
                        current_meta.get("test_accuracy", 0.0),
                    )
                except Exception:
                    pass

            threshold = current_score * 0.95  # Must be >= 95% of current

            # 4. Swap if candidate meets threshold
            if candidate_score >= threshold:
                swapped = strategy.hot_swap_model()
                if swapped:
                    msg = (
                        f"Model retrained and deployed.\n"
                        f"Old score: {current_score:.4f}\n"
                        f"New score: {candidate_score:.4f}"
                    )
                    logger.info(
                        "scheduler.retrain_swapped",
                        old=current_score,
                        new=candidate_score,
                    )
                    await send_alert("Model Retrain Success", msg)
                else:
                    await send_alert(
                        "Model Retrain Warning",
                        "Candidate trained but hot-swap failed. Rollback applied.",
                    )
            else:
                msg = (
                    f"Candidate rejected (below threshold).\n"
                    f"Current: {current_score:.4f}\n"
                    f"Candidate: {candidate_score:.4f}\n"
                    f"Threshold (95%): {threshold:.4f}"
                )
                logger.info(
                    "scheduler.retrain_rejected",
                    current=current_score,
                    candidate=candidate_score,
                )
                await send_alert("Model Retrain Skipped", msg)

            # ── LSTM retrain ──────────────────────────────────────────
            try:
                from app.strategy.nn_strategy import NNStrategy
                nn = NNStrategy()
                nn_result = await nn.train(historical_data)
                if nn_result.success:
                    logger.info("scheduler.lstm_retrained", metrics=nn_result.metrics)
                    await send_alert(
                        "LSTM Retrain Success",
                        nn_result.message,
                    )
                else:
                    logger.warning("scheduler.lstm_retrain_failed", msg=nn_result.message)
            except ImportError:
                logger.info("scheduler.lstm_skip_no_torch")
            except Exception:
                logger.exception("scheduler.lstm_retrain_error")

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

    # ── Monthly hyperparameter optimization (1st of month, 03:00 UTC) ──
    async def monthly_hyperopt():
        """Run extended hyperparameter search with broader param ranges."""
        import pandas as pd

        from app.monitoring.alerts import send_alert
        from app.strategy.ml_strategy import MLStrategy

        try:
            import yfinance as yf
        except ImportError:
            logger.error("scheduler.hyperopt_yfinance_missing")
            return

        from app.config import settings
        from app.data.indicators import compute_features
        from app.strategy.feature_pipeline import create_binary_target

        logger.info("scheduler.hyperopt_started")

        try:
            symbols = settings.symbols_list
            feature_dfs = []
            for sym in symbols:
                try:
                    df = yf.download(sym, period="2y", interval="1d", progress=False)
                    if df.empty:
                        continue
                    df = df.copy()
                    df = df.rename(columns={
                        "Open": "open", "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume",
                    })
                    keep = ["open", "high", "low", "close", "volume"]
                    df = df[[c for c in keep if c in df.columns]]
                    df.index.name = "timestamp"
                    df = df.reset_index()
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    feat_df = compute_features(df)
                    feat_df["target"] = create_binary_target(
                        feat_df, forward_periods=5, buy_threshold=0.01,
                    )
                    feat_df = feat_df.dropna()
                    if len(feat_df) >= 60:
                        feature_dfs.append(feat_df)
                except Exception:
                    continue

            if not feature_dfs:
                logger.error("scheduler.hyperopt_no_data")
                return

            historical_data = pd.concat(feature_dfs, ignore_index=True)
            historical_data = historical_data.sort_values("timestamp").reset_index(drop=True)

            # Train with extended data (2y instead of 1y)
            strategy = MLStrategy()
            result = await strategy.train_candidate(historical_data, features_precomputed=True)

            if result.success:
                candidate_score = result.metrics.get(
                    "test_profit_score", result.metrics.get("test_accuracy", 0.0),
                )
                msg = (
                    f"Monthly hyperopt complete.\n"
                    f"Score: {candidate_score:.4f}\n"
                    f"Params: {result.metrics.get('best_params', {})}"
                )
                logger.info("scheduler.hyperopt_done", score=candidate_score)

                # Auto-swap if better than current
                import json
                from app.strategy.ml_strategy import MODEL_DIR
                meta_path = MODEL_DIR / "xgboost_model.json"
                current_score = 0.0
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            current_meta = json.load(f)
                        current_score = current_meta.get(
                            "test_profit_score",
                            current_meta.get("test_accuracy", 0.0),
                        )
                    except Exception:
                        pass

                if candidate_score > current_score:
                    swapped = strategy.hot_swap_model()
                    if swapped:
                        msg += f"\nSwapped! (was {current_score:.4f})"
                    else:
                        msg += "\nSwap failed, rollback applied."
                else:
                    msg += f"\nKept current ({current_score:.4f})"

                await send_alert("Monthly Hyperopt", msg)
            else:
                logger.warning("scheduler.hyperopt_failed", message=result.message)

        except Exception:
            logger.exception("scheduler.hyperopt_error")

    scheduler.add_job(
        monthly_hyperopt,
        CronTrigger(day=1, hour=3, minute=0),
        id="monthly_hyperopt",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=7200,
    )
    logger.info(
        "scheduler.job_added",
        job="monthly_hyperopt",
        trigger="cron(1st 03:00)",
        misfire_grace_time=7200,
    )
