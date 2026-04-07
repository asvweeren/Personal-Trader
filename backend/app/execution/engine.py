"""Main trading orchestrator - the heart of the system."""

import asyncio
from datetime import UTC, datetime
from enum import StrEnum

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.broker.base import BrokerAdapter, OrderType, Portfolio
from app.config import settings
from app.core.event_bus import RECONCILIATION_UPDATE, RISK_DAILY_STOP, SIGNAL_GENERATED, SYSTEM_ERROR, event_bus
from app.data.correlation import get_correlation_matrix
from app.data.market_data import MarketDataService
from app.execution.order_manager import OrderManager
from app.execution.portfolio_tracker import PortfolioTracker
from app.execution.smart_execution import SmartExecutor
from app.models.order import Order as DBOrder, OrderStatus
from app.models.signal import Signal as DBSignal
from app.models.signal import SignalAction as DBSignalAction
from app.models.trade import Trade, TradeSide, TradeStatus
from app.monitoring.alerts import send_alert
from app.monitoring.performance import PerformanceTracker
from app.risk.manager import RiskManager
from app.risk.market_hours import minutes_until_close_for_symbol
from app.risk.position_sizer import (
    calculate_progressive_trailing_stop,
    calculate_take_profit,
    calculate_trailing_stop,
)
from app.risk.reconciliation import (
    auto_fix,
    reconcile,
    set_last_result,
)
from app.strategy.adaptive import AdaptiveManager
from app.strategy.base import SignalAction, Strategy, TradingSignal
from app.strategy.multi_timeframe import MultiTimeframeFilter
from app.strategy.regime import RegimeDetector

logger = structlog.get_logger()


class EngineState(StrEnum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class TradingEngine:
    """Orchestrates the trading loop: data -> strategy -> risk -> execution."""

    def __init__(
        self,
        broker: BrokerAdapter,
        strategies: list[Strategy],
        risk_manager: RiskManager,
        market_data: MarketDataService,
        performance: PerformanceTracker,
        db: AsyncSession,
        symbols: list[str],
        trading_enabled: bool = False,
        session_factory: async_sessionmaker | None = None,
    ):
        self._broker = broker
        self._strategies = strategies
        self._risk_manager = risk_manager
        self._market_data = market_data
        self._performance = performance
        self._db = db
        self._session_factory = session_factory
        self._symbols = symbols
        self._trading_enabled = trading_enabled
        self._order_manager = OrderManager(broker, db)
        self._portfolio_tracker = PortfolioTracker(broker, db, performance)
        self._smart_executor = SmartExecutor(broker, self._order_manager)
        self._regime_detector = RegimeDetector()
        self._mtf_filter = MultiTimeframeFilter()
        self._adaptive = AdaptiveManager()
        self._state = EngineState.STOPPED
        self._cycle_count = 0
        self._last_cycle_at: datetime | None = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        # Track open trades by symbol for close logic
        self._open_trades: dict[str, Trade] = {}
        # Track last close time per symbol for re-entry cooldown
        self._last_close_time: dict[str, datetime] = {}
        # Track last trailing stop update price to prevent whipsaw
        self._last_stop_update_price: dict[str, float] = {}
        # Track trades that need stop-loss placement retry (IBKR race condition)
        self._pending_stop_retries: set[str] = set()
        # Track per-symbol daily trade count to prevent over-trading
        self._daily_symbol_trades: dict[str, int] = {}
        # Cache snapshot features for ATR lookups in _execute_buy
        self._snapshot_features: dict[str, dict] = {}
        # Track symbols with pending EOD sell to prevent duplicate sells → short positions
        self._eod_sell_pending: set[str] = set()

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def trading_enabled(self) -> bool:
        return self._trading_enabled

    @trading_enabled.setter
    def trading_enabled(self, value: bool) -> None:
        self._trading_enabled = value
        logger.info("engine.trading_toggled", enabled=value)

    async def _await_market_fill(self, result: "OrderResult") -> "OrderResult":
        """Poll briefly for a MARKET order fill (IBKR returns PendingSubmit initially)."""
        mapped = self._order_manager._map_status(result.status)
        if mapped == OrderStatus.FILLED or not result.order_id:
            return result
        for _ in range(6):  # Up to 3 seconds
            await asyncio.sleep(0.5)
            try:
                poll = await self._broker.get_order_status(result.order_id)
                if self._order_manager._map_status(poll.status) == OrderStatus.FILLED:
                    return poll
            except Exception:
                break
        return result

    def update_symbols(self, symbols: list[str]) -> None:
        old = self._symbols
        self._symbols = symbols
        logger.info("engine.symbols_updated", old=old, new=symbols)

    async def start(self) -> None:
        """Initialize the trading engine."""
        self._state = EngineState.STARTING

        try:
            if not await self._broker.is_connected():
                await self._broker.connect()

            # Initialize daily tracking (fallback to config capital if broker errors)
            try:
                start_value = await self._portfolio_tracker.initialize_daily()
            except Exception as e:
                logger.warning("engine.daily_init_fallback", error=str(e)[:200])
                start_value = settings.initial_capital
            self._risk_manager.set_daily_start_value(start_value)

            # Load open trades from DB
            await self._load_open_trades()

            # Reconcile stale SUBMITTED orders from before restart
            await self._reconcile_stale_orders()

            self._state = EngineState.RUNNING
            self._reconnect_attempts = 0
            logger.info(
                "engine.started",
                symbols=self._symbols,
                trading=self._trading_enabled,
                open_trades=len(self._open_trades),
            )
        except Exception:
            self._state = EngineState.ERROR
            logger.exception("engine.start_error")
            raise

    async def stop(self) -> None:
        """Gracefully stop the trading engine."""
        self._state = EngineState.STOPPING

        # Cancel all pending orders
        cancelled = await self._order_manager.cancel_all_pending()
        if cancelled > 0:
            logger.info("engine.shutdown_cancelled_orders", count=cancelled)

        # Final portfolio snapshot
        try:
            await self._portfolio_tracker.take_snapshot()
        except Exception:
            logger.exception("engine.shutdown_snapshot_error")

        # Commit any pending DB changes
        try:
            await self._db.commit()
        except Exception:
            logger.exception("engine.shutdown_commit_error")

        self._state = EngineState.STOPPED
        logger.info("engine.stopped", cycles=self._cycle_count)

    async def run_cycle(self) -> None:
        """Run one complete trading cycle: data -> signals -> risk -> orders.

        Wrapped in a 4-minute timeout to prevent a hung cycle from blocking
        all subsequent cycles (APScheduler max_instances=1).
        """
        if self._state != EngineState.RUNNING:
            return

        try:
            await asyncio.wait_for(self._run_cycle_inner(), timeout=240)
        except TimeoutError:
            logger.error(
                "engine.cycle_timeout",
                timeout_seconds=240,
                cycle=self._cycle_count,
            )
            await event_bus.publish(SYSTEM_ERROR, {
                "component": "trading_engine",
                "error": "Cycle timed out after 240s",
            })
            self._cycle_count += 1
            self._last_cycle_at = datetime.now(UTC)
        except Exception:
            logger.exception("engine.cycle_error")
            await event_bus.publish(SYSTEM_ERROR, {
                "component": "trading_engine",
                "error": "Cycle failed",
            })

    async def _refresh_session(self) -> None:
        """Create a fresh DB session per cycle to prevent stale data and memory leaks."""
        if self._session_factory is None:
            return
        try:
            await self._db.close()
        except Exception:
            pass
        self._db = self._session_factory()
        self._order_manager._db = self._db
        self._portfolio_tracker._db = self._db

    async def _run_cycle_inner(self) -> None:
        """Inner cycle logic, called with a timeout by run_cycle()."""
        try:
            # Refresh DB session each cycle to prevent stale connections / memory leaks
            await self._refresh_session()

            # Check broker connectivity
            if not await self._ensure_connected():
                return

            # 1. Poll pending orders for status updates
            filled_orders = await self._order_manager.poll_pending_orders()
            for fill in filled_orders:
                await self._handle_order_fill(fill)

            # 2. Get market data (only for symbols whose markets are open)
            from app.risk.market_hours import get_exchange_for_symbol, is_market_open
            open_symbols = [
                s for s in self._symbols
                if is_market_open(
                    get_exchange_for_symbol(s),
                    include_extended=settings.extended_hours_enabled,
                )
            ]
            # Always include symbols with open trades so we can manage stops/exits
            for sym in list(self._open_trades.keys()):
                if sym not in open_symbols:
                    open_symbols.append(sym)
            if not open_symbols:
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(UTC)
                return
            snapshot = await self._market_data.get_snapshot(open_symbols)
            # Extract latest features from computed DataFrames for ATR lookups
            for sym, feat_df in snapshot.computed_features_df.items():
                if feat_df is not None and not feat_df.empty:
                    try:
                        row = feat_df.iloc[-1]
                        self._snapshot_features[sym] = {
                            k: float(v) for k, v in row.items()
                            if isinstance(v, (int, float)) and v == v  # skip NaN
                        }
                    except Exception:
                        pass

            # 2a. Staleness check: skip cycle if data is too old
            data_age = (datetime.now(UTC) - snapshot.timestamp).total_seconds()
            if data_age > 300:
                logger.warning(
                    "engine.stale_data_skip",
                    age_seconds=round(data_age),
                    threshold=300,
                )
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(UTC)
                return

            # 2b. Reconciliation every 3rd cycle (~15 min)
            if self._cycle_count > 0 and self._cycle_count % 3 == 0:
                try:
                    portfolio_for_recon = await self._portfolio_tracker.get_current()
                    recon_result = await reconcile(self._open_trades, portfolio_for_recon)
                    set_last_result(recon_result)
                    await event_bus.publish(RECONCILIATION_UPDATE, recon_result.to_dict())
                    if not recon_result.is_clean:
                        actions = await auto_fix(recon_result, self, self._db)
                        if actions:
                            await send_alert(
                                "Position Reconciliation",
                                "Mismatches detected and fixed:\n" + "\n".join(actions),
                            )
                        logger.warning("reconciliation.mismatches", result=recon_result.to_dict())
                    else:
                        logger.info("reconciliation.ok", matches=len(recon_result.matches))
                except Exception:
                    logger.debug("reconciliation.error", exc_info=True)

                # Also close DB trades that are OPEN but have no broker position
                # (handles duplicates and trades orphaned across restarts)
                try:
                    await self._close_orphaned_db_trades(portfolio_for_recon)
                except Exception:
                    logger.debug("reconciliation.db_cleanup_error", exc_info=True)

            # 2c. Detect market regime
            try:
                regime_state = self._regime_detector.detect(snapshot)
            except Exception:
                logger.debug("regime.detection_error", exc_info=True)
                regime_state = None

            # 3. EOD close — force-sell positions near market close (only if enabled)
            if settings.eod_close_enabled:
                await self._check_eod_close(snapshot.prices)

            # 3a. Max hold days — force-close positions held too many trading days
            if settings.max_hold_days > 0:
                await self._check_max_hold_days(snapshot.prices)

            # 3b. Stale position exit — close positions held too long with minimal P&L
            await self._check_stale_positions(snapshot.prices)

            # 4. Take-profit check — sell positions that hit their target
            await self._check_take_profits(snapshot.prices)

            # 4b. Retry failed stop-loss placements (IBKR race condition)
            await self._retry_pending_stops()

            # 5. Check trailing stops on open positions (progressive)
            await self._check_trailing_stops(snapshot.prices)

            # 6. Generate signals from all strategies
            #    Pass regime state to ML strategy for regime-aware thresholds
            if regime_state is not None:
                for strategy in self._strategies:
                    if hasattr(strategy, "get_regime_threshold"):
                        strategy._current_regime = regime_state
                    # Also propagate to sub-strategies in ensemble
                    if hasattr(strategy, "_strategies"):
                        for sub in strategy._strategies:
                            if hasattr(sub, "get_regime_threshold"):
                                sub._current_regime = regime_state

            all_signals: list[TradingSignal] = []
            # Day trading mode: momentum (primary) + ML (confirmation).
            # Skip sentiment (too slow for intraday) and ensemble.
            _skip_strategies = {"ensemble", "sentiment", "nn_lstm"}
            strategies_to_run = [
                s for s in self._strategies if s.name not in _skip_strategies
            ]
            for strategy in strategies_to_run:
                try:
                    signals = await strategy.generate_signals(snapshot)
                    all_signals.extend(signals)
                except Exception:
                    logger.exception("engine.strategy_error", strategy=strategy.name)

            if not all_signals:
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(UTC)
                return

            # 6a. Deduplicate: keep highest-confidence signal per symbol per action
            best_signals: dict[tuple[str, str], TradingSignal] = {}
            for sig in all_signals:
                key = (sig.symbol, sig.action.value)
                existing = best_signals.get(key)
                if existing is None or sig.confidence > existing.confidence:
                    best_signals[key] = sig
            all_signals = list(best_signals.values())

            # 6b. Time-of-day confidence adjustment
            # Market open (first 30 min) and close (last 30 min) are noisier
            now_utc = datetime.now(UTC)
            hour_utc = now_utc.hour
            # US market: 14:30-21:00 UTC, EU: 08:00-16:30 UTC
            # Penalize first/last 30 min of US session (14:30-15:00, 20:30-21:00)
            tod_factor = 1.0
            if (hour_utc == 14 and now_utc.minute < 60) or (hour_utc == 20 and now_utc.minute >= 30):
                tod_factor = 0.85  # 15% penalty during volatile open/close
            elif hour_utc in (15, 16, 17, 18, 19):
                tod_factor = 1.0  # Core hours — full confidence

            # 6c. Adaptive threshold + multi-timeframe confirmation
            regime_name = getattr(getattr(regime_state, "regime", None), "value", "unknown")
            confirmed_signals: list[TradingSignal] = []
            for signal in all_signals:
                if signal.action == SignalAction.HOLD:
                    confirmed_signals.append(signal)
                    continue

                # Apply adaptive per-symbol + per-regime threshold adjustment
                if signal.action == SignalAction.BUY:
                    adaptive_threshold = self._adaptive.get_adjusted_threshold(
                        settings.confidence_threshold, signal.symbol, regime_name,
                    )
                    if signal.confidence < adaptive_threshold:
                        logger.debug(
                            "engine.adaptive_threshold_skip",
                            symbol=signal.symbol,
                            confidence=round(signal.confidence, 3),
                            adaptive_threshold=round(adaptive_threshold, 3),
                        )
                        continue

                # Apply time-of-day adjustment
                if tod_factor < 1.0:
                    signal = TradingSignal(
                        symbol=signal.symbol,
                        action=signal.action,
                        confidence=signal.confidence * tod_factor,
                        strategy_name=signal.strategy_name,
                        features_snapshot=signal.features_snapshot,
                        metadata=signal.metadata,
                    )

                try:
                    # Use pre-computed features from snapshot (already has 1Y daily data)
                    precomputed = snapshot.computed_features_df.get(signal.symbol)
                    if precomputed is not None and len(precomputed) >= 20:
                        daily_feat = precomputed.iloc[-1].to_dict()
                    else:
                        daily_feat = {}
                    hourly_feat = signal.features_snapshot or {}
                    signal = self._mtf_filter.confirm_signal(signal, hourly_feat, daily_feat)
                except Exception:
                    pass  # Keep original signal on error
                confirmed_signals.append(signal)
            all_signals = confirmed_signals

            # 7. Get current portfolio for risk checks
            portfolio = await self._portfolio_tracker.get_current()

            # 7b. Compute correlation matrix for position sizing
            try:
                correlation_matrix = await get_correlation_matrix(
                    self._symbols, self._market_data, snapshot=snapshot
                )
            except Exception:
                logger.debug("engine.correlation_compute_failed", exc_info=True)
                correlation_matrix = None

            # 8. Evaluate each signal through risk management
            #    Limit BUY executions per cycle to prevent timeout (each BUY takes ~30s)
            buys_this_cycle = 0
            max_buys_per_cycle = 2  # Max 2 BUY orders per 5-min cycle
            for signal in all_signals:
                if signal.action == SignalAction.HOLD:
                    continue

                await event_bus.publish(SIGNAL_GENERATED, {
                    "symbol": signal.symbol,
                    "action": signal.action.value,
                    "confidence": signal.confidence,
                    "strategy": signal.strategy_name,
                })

                # Persist signal to DB
                db_signal = DBSignal(
                    strategy_name=signal.strategy_name,
                    symbol=signal.symbol,
                    action=DBSignalAction(signal.action.value),
                    confidence=signal.confidence,
                    features_snapshot=signal.features_snapshot,
                    metadata_=signal.metadata,
                )
                self._db.add(db_signal)
                await self._db.flush()

                # Get estimated price
                price = snapshot.prices.get(signal.symbol, 0.0)
                if price <= 0:
                    continue

                # Skip blacklisted symbols (except SELL to close existing positions)
                if signal.action == SignalAction.BUY and (
                    signal.symbol in settings.symbol_blacklist_set
                    or signal.symbol in self._performance.get_underperforming_symbols()
                    or self._adaptive.should_skip_symbol(signal.symbol)
                ):
                    logger.info(
                        "engine.signal_skipped_blacklist",
                        symbol=signal.symbol,
                    )
                    continue

                # Gate 1: SPY/QQQ momentum — don't BUY if broad market is weak
                if signal.action == SignalAction.BUY and signal.symbol not in ("SPY", "QQQ", "IWM", "DIA"):
                    market_bearish = False
                    for index_sym in ("SPY", "QQQ"):
                        idx_features = self._snapshot_features.get(index_sym, {})
                        idx_price = snapshot.prices.get(index_sym, 0)
                        idx_sma50 = idx_features.get("sma_50", 0)
                        idx_sma20 = idx_features.get("sma_20", 0)
                        if idx_price > 0 and idx_sma50 > 0 and idx_price < idx_sma50:
                            market_bearish = True
                        if idx_price > 0 and idx_sma20 > 0 and idx_price < idx_sma20:
                            market_bearish = True
                        # Gate 1b: Skip if SPY/QQQ intraday return < -1% (crash protection)
                        idx_momentum = idx_features.get("momentum_1d", 0)
                        if idx_momentum < -0.01:
                            market_bearish = True
                        # Gate 1c: Skip if ATR/price ratio > 3% (high volatility day)
                        idx_atr = idx_features.get("atr_14", 0)
                        if idx_price > 0 and idx_atr > 0 and (idx_atr / idx_price) > 0.03:
                            market_bearish = True
                    if market_bearish:
                        logger.info(
                            "engine.signal_skipped_market_bearish",
                            symbol=signal.symbol,
                        )
                        continue

                # Gate 2: Opening range — no BUY in first N minutes after market open
                if signal.action == SignalAction.BUY and settings.opening_range_minutes > 0:
                    mins_since_open = self._minutes_since_market_open(signal.symbol)
                    if mins_since_open is not None and mins_since_open < settings.opening_range_minutes:
                        logger.info(
                            "engine.signal_skipped_opening_range",
                            symbol=signal.symbol,
                            mins_since_open=round(mins_since_open, 1),
                        )
                        continue

                # Gate 3: Volume filter — skip if today's volume too low
                if signal.action == SignalAction.BUY and settings.min_relative_volume > 0:
                    rel_vol = self._get_relative_volume(signal.symbol, snapshot)
                    if rel_vol is not None and rel_vol < settings.min_relative_volume:
                        logger.info(
                            "engine.signal_skipped_low_volume",
                            symbol=signal.symbol,
                            relative_volume=round(rel_vol, 2),
                        )
                        continue

                # Gate 4: Intraday trend filter — stock must show short-term strength
                if signal.action == SignalAction.BUY:
                    sym_features = self._snapshot_features.get(signal.symbol, {})
                    sym_price = snapshot.prices.get(signal.symbol, 0)
                    sym_ema10 = sym_features.get("ema_10", 0)
                    sym_rsi = sym_features.get("rsi_14", 0)
                    sym_vwap = sym_features.get("vwap", 0)

                    # Stock must be above VWAP or EMA10 (intraday buying pressure)
                    # Only apply if features are available (skip in test/mock mode)
                    above_vwap = sym_vwap > 0 and sym_price > sym_vwap
                    above_ema = sym_ema10 > 0 and sym_price > sym_ema10
                    has_features = sym_vwap > 0 or sym_ema10 > 0
                    if has_features and sym_price > 0 and not above_vwap and not above_ema:
                        logger.info(
                            "engine.signal_skipped_no_intraday_strength",
                            symbol=signal.symbol,
                            price=round(sym_price, 2),
                            vwap=round(sym_vwap, 2) if sym_vwap else 0,
                            ema10=round(sym_ema10, 2) if sym_ema10 else 0,
                        )
                        continue

                    # RSI filter: don't buy overbought (>75) or in freefall (<30)
                    if sym_rsi > 0 and (sym_rsi > 75 or sym_rsi < 30):
                        logger.info(
                            "engine.signal_skipped_rsi",
                            symbol=signal.symbol,
                            rsi=round(sym_rsi, 1),
                        )
                        continue

                # Handle SELL signals via position close
                if signal.action == SignalAction.SELL:
                    await self._handle_sell_signal(signal, price, db_signal.id)
                    continue

                # Skip duplicate BUY if we already have an open/pending trade for this symbol
                if signal.symbol in self._open_trades:
                    logger.info(
                        "engine.signal_skipped_existing",
                        symbol=signal.symbol,
                        reason="Already have open/pending trade",
                    )
                    continue

                # Re-entry cooldown: don't re-buy a symbol too soon after closing
                last_close = self._last_close_time.get(signal.symbol)
                if last_close and settings.reentry_cooldown_minutes > 0:
                    mins_since = (
                        datetime.now(UTC) - last_close
                    ).total_seconds() / 60
                    if mins_since < settings.reentry_cooldown_minutes:
                        logger.info(
                            "engine.signal_skipped_cooldown",
                            symbol=signal.symbol,
                            mins_since_close=round(mins_since, 1),
                            cooldown=settings.reentry_cooldown_minutes,
                        )
                        continue

                # Per-symbol daily trade limit: prevent over-trading same symbol
                symbol_trades_today = self._daily_symbol_trades.get(signal.symbol, 0)
                if symbol_trades_today >= settings.max_trades_per_symbol_per_day:
                    logger.info(
                        "engine.signal_skipped_daily_limit",
                        symbol=signal.symbol,
                        trades_today=symbol_trades_today,
                        limit=settings.max_trades_per_symbol_per_day,
                    )
                    continue

                # Gate 4: Max new positions per day (swing trading capital management)
                if settings.max_new_positions_per_day > 0:
                    total_new_today = sum(self._daily_symbol_trades.values())
                    if total_new_today >= settings.max_new_positions_per_day:
                        logger.info(
                            "engine.signal_skipped_daily_position_limit",
                            symbol=signal.symbol,
                            new_today=total_new_today,
                            limit=settings.max_new_positions_per_day,
                        )
                        continue

                # For BUY signals, run risk evaluation
                decision = await self._risk_manager.evaluate_signal(
                    signal, portfolio, price,
                    correlation_matrix=correlation_matrix,
                    regime=regime_state,
                )

                # Re-broadcast signal with AI modifier from risk evaluation
                if decision.ai_modifier != 1.0:
                    await event_bus.publish(SIGNAL_GENERATED, {
                        "symbol": signal.symbol,
                        "action": signal.action.value,
                        "confidence": signal.confidence,
                        "strategy": signal.strategy_name,
                        "ai_modifier": decision.ai_modifier,
                    })

                if not decision.approved:
                    logger.info(
                        "engine.signal_rejected",
                        symbol=signal.symbol,
                        reason=decision.reason,
                    )
                    if self._risk_manager.daily_loss_triggered:
                        await event_bus.publish(RISK_DAILY_STOP, {})
                        await send_alert(
                            "Daily Loss Limit Hit",
                            "Trading has been halted due to daily loss limit.",
                            critical=True,
                        )
                    continue

                # Execute trade if trading is enabled
                if not self._trading_enabled:
                    logger.info(
                        "engine.signal_approved_no_execute",
                        symbol=signal.symbol,
                        action=signal.action.value,
                        quantity=decision.adjusted_quantity,
                        reason="Trading disabled",
                    )
                    continue

                qty = decision.adjusted_quantity or 0
                if qty <= 0:
                    logger.info(
                        "engine.signal_skipped_zero_quantity",
                        symbol=signal.symbol,
                        reason="Position sizer returned 0 shares",
                    )
                    continue

                # Limit BUY executions per cycle to prevent 240s timeout
                if buys_this_cycle >= max_buys_per_cycle:
                    logger.info(
                        "engine.signal_skipped_cycle_limit",
                        symbol=signal.symbol,
                        buys_this_cycle=buys_this_cycle,
                    )
                    continue
                await self._execute_buy(signal, qty, price, db_signal.id)
                buys_this_cycle += 1

            await self._db.commit()

            # 9. Take portfolio snapshot
            await self._portfolio_tracker.take_snapshot()
            await self._db.commit()

            # 10. Update ensemble weights and save adaptive state every 50 cycles (~4 hours)
            if self._cycle_count > 0 and self._cycle_count % 50 == 0:
                for strategy in self._strategies:
                    if hasattr(strategy, "update_weights_from_history"):
                        try:
                            strategy.update_weights_from_history(self._performance)
                        except Exception:
                            logger.debug("engine.ensemble_weight_update_failed", exc_info=True)
                try:
                    self._adaptive.save_state()
                except Exception:
                    logger.debug("engine.adaptive_save_failed", exc_info=True)

            self._cycle_count += 1
            self._last_cycle_at = datetime.now(UTC)

        except Exception:
            logger.exception("engine.cycle_inner_error")
            self._cycle_count += 1
            self._last_cycle_at = datetime.now(UTC)
            raise  # Re-raise so run_cycle() wrapper can log/publish

    async def _ensure_connected(self) -> bool:
        """Check broker connection and attempt reconnect if needed.

        Never gives up — the broker adapter handles auto-reconnect with
        exponential backoff.  The engine just skips cycles while disconnected.
        """
        if await self._broker.is_connected():
            if self._reconnect_attempts > 0:
                logger.info("engine.reconnected", after_attempts=self._reconnect_attempts)
                await send_alert(
                    "Engine Reconnected",
                    f"Trading engine restored after {self._reconnect_attempts} cycle(s) offline.",
                )
            self._reconnect_attempts = 0
            if self._state == EngineState.ERROR:
                self._state = EngineState.RUNNING
            return True

        self._reconnect_attempts += 1

        # Alert on first disconnect and then every 10 cycles (~50 min)
        if self._reconnect_attempts == 1 or self._reconnect_attempts % 10 == 0:
            logger.warning(
                "engine.broker_offline",
                attempt=self._reconnect_attempts,
            )
            if self._reconnect_attempts == 1:
                await send_alert(
                    "Broker Disconnected",
                    "IBKR connection lost. Auto-reconnect is active — "
                    "trading will resume when connection is restored.",
                )

        # Try reconnect from the engine side as well (belt and suspenders)
        try:
            await self._broker.connect()
            self._reconnect_attempts = 0
            if self._state == EngineState.ERROR:
                self._state = EngineState.RUNNING
            logger.info("engine.reconnected")
            return True
        except Exception:
            logger.debug("engine.reconnect_pending", attempt=self._reconnect_attempts)
            return False

    async def _handle_sell_signal(
        self, signal: TradingSignal, price: float, signal_id: int
    ) -> None:
        """Handle a SELL signal by closing the open trade for the symbol."""
        open_trade = self._open_trades.get(signal.symbol)
        if not open_trade:
            logger.info("engine.no_position_to_sell", symbol=signal.symbol)
            return

        if not self._trading_enabled:
            logger.info(
                "engine.sell_signal_no_execute",
                symbol=signal.symbol,
                reason="Trading disabled",
            )
            return

        # Enforce minimum hold time before allowing signal-based close
        if open_trade.created_at and settings.min_hold_minutes > 0:
            held_minutes = (
                datetime.now(UTC) - open_trade.created_at
            ).total_seconds() / 60
            if held_minutes < settings.min_hold_minutes:
                logger.info(
                    "engine.sell_signal_too_early",
                    symbol=signal.symbol,
                    held_minutes=round(held_minutes, 1),
                    min_hold=settings.min_hold_minutes,
                )
                return

        # Cancel any pending SELL orders (e.g. stop-loss) before placing new sell
        await self._cancel_pending_sells(signal.symbol)
        # Verify no pending sells remain to avoid accidental short
        remaining = len(self._order_manager._pending_by_symbol.get(signal.symbol, set()))
        if remaining > 0:
            logger.warning(
                "engine.sell_cancel_incomplete",
                symbol=signal.symbol,
                remaining_orders=remaining,
            )
            return

        # Place sell order for the full position
        result = await self._order_manager.submit_order(
            trade_id=open_trade.id,
            symbol=signal.symbol,
            side="SELL",
            quantity=open_trade.quantity,
            order_type=OrderType.MARKET,
            expected_price=price,
        )
        result = await self._await_market_fill(result)

        mapped_status = self._order_manager._map_status(result.status)
        if mapped_status == OrderStatus.FILLED and result.filled_price:
            await self._portfolio_tracker.record_trade_close(
                open_trade, result.filled_price
            )
            self._open_trades.pop(signal.symbol, None)
            self._eod_sell_pending.discard(signal.symbol)
            self._last_close_time[signal.symbol] = datetime.now(UTC)
            pnl = open_trade.realized_pnl or 0.0
            hold_mins = ""
            if open_trade.created_at:
                delta = datetime.now(UTC) - open_trade.created_at
                hold_mins = f" | Hold: {delta.total_seconds() / 60:.0f}min"
            logger.info(
                "engine.position_closed",
                symbol=signal.symbol,
                exit_price=result.filled_price,
                pnl=pnl,
            )
            # Record outcome for dynamic strategy weight adjustment
            for strategy in self._strategies:
                if hasattr(strategy, "record_outcome"):
                    strategy.record_outcome(signal.symbol, open_trade.strategy_name, pnl > 0)
            self._record_adaptive_outcome(open_trade)
            await send_alert(
                "Trade Closed",
                f"{signal.symbol}: SOLD {open_trade.quantity} @ {result.filled_price:.2f}\n"
                f"P&L: {pnl:+.2f}{hold_mins}\n"
                f"Exit reason: signal",
            )

    async def _cancel_pending_sells(self, symbol: str) -> int:
        """Cancel all pending SELL orders for a symbol before placing a new sell.

        This prevents the old stop-loss from triggering after a position is
        already closed, which would create an unintended short position.
        Cancels both in-memory tracked orders AND broker-side orders as safety net.
        """
        cancelled = await self._order_manager.cancel_orders_for_symbol(symbol, side="SELL")
        # Safety net: also cancel directly at broker in case orders aren't tracked in memory
        # (e.g. after restart, or orders that were tracked but already removed from pending)
        try:
            broker_cancelled = await self._broker.cancel_open_orders_for_symbol(symbol)
            if broker_cancelled:
                logger.info(
                    "engine.broker_cancel_safety_net",
                    symbol=symbol,
                    broker_cancelled=broker_cancelled,
                )
                cancelled += broker_cancelled
        except Exception:
            logger.debug("engine.broker_cancel_safety_net_failed", symbol=symbol)
        if cancelled:
            logger.info(
                "engine.cancelled_pending_sells",
                symbol=symbol,
                count=cancelled,
            )
        return cancelled

    async def _verify_position_or_record_close(
        self, symbol: str, trade: "Trade"
    ) -> bool:
        """Check if a position still exists at IBKR before replacing its stop.

        If the broker position is gone (qty=0), the stop-loss was filled at IBKR
        but the engine missed the fill event (race condition during cancel/replace).
        In that case, record the trade closure using the last known market price.

        Returns True if the position still exists, False if it was closed.
        """
        try:
            portfolio = await self._broker.get_portfolio()
            for p in portfolio.positions:
                if p.symbol == symbol and int(p.quantity) > 0:
                    return True  # Position still exists, safe to replace stop

            # Position is GONE at IBKR — the stop was filled but we missed it
            logger.warning(
                "engine.stop_fill_detected_via_position_check",
                symbol=symbol,
                trade_id=trade.id,
                entry_price=trade.entry_price,
                stop_loss=trade.stop_loss,
            )

            # Use last market price from portfolio, or the stop-loss price as fill estimate
            fill_price = trade.stop_loss or trade.entry_price or 0.0
            for p in portfolio.positions:
                if p.symbol == symbol:
                    fill_price = p.market_price or fill_price
                    break

            # Clean up any stale pending orders for this symbol
            self._order_manager._pending_by_symbol.pop(symbol, None)
            # Remove stale entries from _pending_orders
            stale_ids = [
                bid for bid, o in self._order_manager._pending_orders.items()
                if o.symbol == symbol
            ]
            for bid in stale_ids:
                self._order_manager._pending_orders.pop(bid, None)
                self._order_manager._submitted_at.pop(bid, None)

            # Record trade closure
            await self._portfolio_tracker.record_trade_close(trade, fill_price)
            self._open_trades.pop(symbol, None)
            self._eod_sell_pending.discard(symbol)
            self._last_close_time[symbol] = datetime.now(UTC)
            self._pending_stop_retries.discard(symbol)
            self._last_stop_update_price.pop(symbol, None)

            pnl = trade.realized_pnl or 0.0
            hold_mins = ""
            if trade.created_at:
                delta = datetime.now(UTC) - trade.created_at
                hold_mins = f" | Hold: {delta.total_seconds() / 60:.0f}min"

            # Record outcome for adaptive strategy weights
            for strategy in self._strategies:
                if hasattr(strategy, "record_outcome"):
                    strategy.record_outcome(symbol, trade.strategy_name, pnl > 0)
            self._record_adaptive_outcome(trade)

            await self._db.flush()
            await send_alert(
                "Stop-Loss Filled (detected)",
                f"{symbol}: Stop filled @ ~{fill_price:.2f}\n"
                f"P&L: {pnl:+.2f}{hold_mins}\n"
                f"Note: fill detected via position check",
            )
            return False  # Position was closed

        except Exception:
            logger.exception(
                "engine.position_verify_error",
                symbol=symbol,
            )
            return True  # Assume position still exists on error (safer)

    async def _close_orphaned_db_trades(self, portfolio: Portfolio) -> None:
        """Close DB trades marked OPEN that have no corresponding broker position.

        This catches trades orphaned by:
        - Duplicate trades for the same symbol (dict key overwrites in _open_trades)
        - Stop-loss fills during engine downtime
        - Trades that fell out of _open_trades after restart
        """
        from sqlalchemy import select

        broker_symbols = {p.symbol for p in portfolio.positions if p.quantity > 0}

        result = await self._db.execute(
            select(Trade).where(Trade.status == TradeStatus.OPEN)
        )
        db_open_trades = result.scalars().all()

        closed_count = 0
        for trade in db_open_trades:
            if trade.symbol not in broker_symbols:
                trade.status = TradeStatus.CLOSED
                trade.exit_price = trade.stop_loss or trade.entry_price
                trade.realized_pnl = (
                    (trade.exit_price - trade.entry_price) * trade.quantity
                    if trade.exit_price and trade.entry_price
                    else 0.0
                )
                trade.closed_at = datetime.now(UTC)
                # Also remove from in-memory tracking if present
                if self._open_trades.get(trade.symbol) and self._open_trades[trade.symbol].id == trade.id:
                    self._open_trades.pop(trade.symbol, None)
                closed_count += 1
                logger.warning(
                    "reconciliation.db_orphan_closed",
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    entry_price=trade.entry_price,
                    est_exit=trade.exit_price,
                    est_pnl=trade.realized_pnl,
                )

        if closed_count:
            await self._db.flush()
            await send_alert(
                "Orphaned Trades Cleaned",
                f"Closed {closed_count} DB trades with no broker position",
            )

    def _get_atr(self, symbol: str) -> float | None:
        """Get ATR value for a symbol from snapshot features."""
        try:
            features = self._snapshot_features.get(symbol)
            if features:
                atr = features.get("atr_14")
                if atr and atr > 0:
                    return float(atr)
        except Exception:
            pass
        return None

    async def _place_stop_verified(
        self, trade_id: int, symbol: str, quantity: int, stop_price: float
    ) -> bool:
        """Place a stop-loss order and verify IBKR accepted it.

        Returns True if the stop was placed and confirmed, False otherwise.
        On failure, adds symbol to retry queue.
        """
        try:
            result = await self._order_manager.submit_order(
                trade_id=trade_id,
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                order_type=OrderType.STOP,
                stop_price=stop_price,
            )
            # Poll once after brief delay to verify broker accepted the stop
            if result.order_id:
                await asyncio.sleep(1)
                try:
                    status = await self._broker.get_order_status(result.order_id)
                    mapped = self._order_manager._map_status(status.status)
                    if mapped in (OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.ERROR):
                        logger.error(
                            "engine.stop_rejected_by_broker",
                            symbol=symbol,
                            stop_price=stop_price,
                            broker_status=status.status,
                        )
                        self._pending_stop_retries.add(symbol)
                        return False
                except Exception:
                    pass  # Verification failed, but order may still be active
            logger.info(
                "engine.stop_placed",
                symbol=symbol,
                stop_price=stop_price,
            )
            return True
        except Exception:
            logger.exception(
                "engine.stop_placement_failed",
                symbol=symbol,
                stop_price=stop_price,
            )
            self._pending_stop_retries.add(symbol)
            return False

    def _calculate_atr_stop(self, filled_price: float, symbol: str) -> float:
        """Calculate stop price using ATR if available, else fallback to 3%."""
        atr_val = self._get_atr(symbol)
        if atr_val and atr_val > 0:
            # atr_14 is now a ratio (atr/close), convert back to raw ATR
            atr_raw = atr_val * filled_price if atr_val < 1.0 else atr_val
            stop_price = round(filled_price - settings.atr_stop_multiplier * atr_raw, 2)
            min_stop = round(filled_price * (1 - settings.min_stop_loss_pct / 100), 2)
            stop_price = max(stop_price, min_stop)
        else:
            stop_price = round(filled_price * 0.97, 2)
        return stop_price

    def _get_avg_volume(self, symbol: str) -> float:
        """Get average daily volume for smart execution algo selection."""
        try:
            cache = getattr(self._market_data, "_historical_cache", {})
            for key, df in cache.items():
                if key.startswith(f"{symbol}_") and df is not None and not df.empty:
                    if "volume" in df.columns:
                        return float(df["volume"].tail(20).mean())
        except Exception:
            pass
        return 0.0

    def _minutes_since_market_open(self, symbol: str) -> float | None:
        """Return minutes since market open for this symbol's exchange."""
        try:
            from app.risk.market_hours import (
                EXCHANGE_SESSIONS,
                get_exchange_for_symbol,
            )
            from zoneinfo import ZoneInfo

            exchange = get_exchange_for_symbol(symbol)
            session = EXCHANGE_SESSIONS.get(exchange)
            if not session:
                return None
            tz = ZoneInfo(session.timezone)
            now_local = datetime.now(UTC).astimezone(tz)
            market_open = datetime.combine(now_local.date(), session.open_time, tzinfo=tz)
            delta = (now_local - market_open).total_seconds() / 60
            return delta if delta >= 0 else None
        except Exception:
            return None

    def _get_relative_volume(self, symbol: str, snapshot) -> float | None:
        """Get today's volume relative to 20-day average (1.0 = normal)."""
        try:
            features = self._snapshot_features.get(symbol, {})
            # Check if we have volume data from computed features
            today_vol = features.get("volume", 0)
            avg_vol = features.get("volume_sma_20", 0)
            if avg_vol and avg_vol > 0 and today_vol > 0:
                return today_vol / avg_vol
            # Fallback: check OHLCV data directly
            df = snapshot.ohlcv.get(symbol)
            if df is not None and not df.empty and "volume" in df.columns:
                today_vol = float(df["volume"].iloc[-1])
                if len(df) >= 20:
                    avg_vol = float(df["volume"].tail(20).mean())
                    if avg_vol > 0:
                        return today_vol / avg_vol
        except Exception:
            pass
        return None

    async def _execute_buy(
        self, signal: TradingSignal, quantity: int, price: float, signal_id: int
    ) -> None:
        """Execute a BUY signal with retry logic and alerts."""
        # R:R gate: check risk/reward ratio before executing
        stop_price_est = self._calculate_atr_stop(price, signal.symbol)
        atr_val = self._get_atr(signal.symbol)
        tp_est = calculate_take_profit(price, signal.symbol, atr_val)
        risk = price - stop_price_est
        reward = tp_est - price
        if risk > 0 and (reward / risk) < 2.0:
            logger.info(
                "engine.insufficient_rr",
                symbol=signal.symbol,
                rr=round(reward / risk, 2),
                price=price,
                stop=stop_price_est,
                tp=tp_est,
            )
            return

        # Create trade record
        trade = Trade(
            symbol=signal.symbol,
            side=TradeSide.BUY,
            quantity=quantity,
            status=TradeStatus.PENDING,
            strategy_name=signal.strategy_name,
            signal_id=signal_id,
        )
        self._db.add(trade)
        await self._db.flush()

        max_retries = settings.order_max_retries
        timeout_secs = settings.order_fill_timeout_seconds
        filled = False

        for attempt in range(1, max_retries + 1):
            # Place the order
            result = await self._order_manager.submit_order(
                trade_id=trade.id,
                symbol=signal.symbol,
                side="BUY",
                quantity=quantity,
                order_type=OrderType.MARKET,
                expected_price=price,
            )

            mapped_status = self._order_manager._map_status(result.status)

            # Wait for fill with configurable timeout
            if mapped_status in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED):
                for _ in range(timeout_secs):
                    await asyncio.sleep(1)
                    try:
                        updated = await self._broker.get_order_status(result.order_id)
                        mapped_status = self._order_manager._map_status(updated.status)
                        if mapped_status == OrderStatus.FILLED:
                            result = updated
                            break
                        if mapped_status in (
                            OrderStatus.CANCELLED,
                            OrderStatus.REJECTED,
                            OrderStatus.ERROR,
                        ):
                            result = updated
                            break
                    except Exception:
                        break

            if mapped_status == OrderStatus.FILLED:
                filled = True
                break

            # Handle partial fill: accept what we got and continue
            if mapped_status == OrderStatus.PARTIALLY_FILLED and result.filled_quantity:
                try:
                    await self._broker.cancel_order(result.order_id)
                except Exception:
                    pass
                # Use the partial fill — update quantity to what was actually filled
                quantity = result.filled_quantity
                filled = True
                logger.info(
                    "engine.partial_fill_accepted",
                    symbol=signal.symbol,
                    requested=trade.quantity,
                    filled=quantity,
                )
                break

            # Not filled — cancel pending order before retry
            if mapped_status == OrderStatus.SUBMITTED:
                try:
                    await self._broker.cancel_order(result.order_id)
                except Exception:
                    pass

            if mapped_status in (OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.ERROR):
                logger.info(
                    "engine.buy_attempt_failed",
                    symbol=signal.symbol,
                    attempt=attempt,
                    broker_status=result.status,
                )
            else:
                logger.warning(
                    "engine.buy_attempt_timeout",
                    symbol=signal.symbol,
                    attempt=attempt,
                    timeout=timeout_secs,
                )

            if attempt < max_retries:
                await asyncio.sleep(2)

        if filled:
            trade.status = TradeStatus.OPEN
            trade.quantity = quantity  # Update to actual filled quantity
            trade.entry_price = result.filled_price
            self._open_trades[signal.symbol] = trade
            self._daily_symbol_trades[signal.symbol] = self._daily_symbol_trades.get(signal.symbol, 0) + 1
            logger.info(
                "engine.trade_opened",
                symbol=signal.symbol,
                quantity=quantity,
                price=result.filled_price,
            )

            # Place stop-loss (ATR-based with min floor, adjusted for slippage)
            # Brief pause to let IBKR register the filled position before placing stop
            await asyncio.sleep(2)
            if result.filled_price:
                stop_price = self._calculate_atr_stop(result.filled_price, signal.symbol)

                # Slippage correction: tighten stop if we got a worse entry
                slippage = result.filled_price - price if price > 0 else 0
                if slippage > 0:
                    # Bought higher than expected — full adjustment to maintain original risk
                    stop_price = round(stop_price + slippage, 2)

                trade.stop_loss = stop_price
                await self._place_stop_verified(
                    trade.id, signal.symbol, quantity, stop_price
                )

                # Set take-profit target (ATR-based with min floor, adjusted for slippage)
                atr_val = self._get_atr(signal.symbol)
                tp = calculate_take_profit(result.filled_price, signal.symbol, atr_val)
                # If we got a worse entry, shift TP up to maintain R:R ratio
                if slippage > 0:
                    tp = round(tp + slippage, 2)
                trade.take_profit = tp

            # Alert: Trade Opened
            alert_msg = (
                f"{signal.symbol}: BUY {quantity} @ {result.filled_price:.2f}\n"
                f"Strategy: {signal.strategy_name} (confidence: {signal.confidence:.0%})"
            )
            if trade.stop_loss and trade.take_profit:
                alert_msg += f"\nStop: {trade.stop_loss:.2f} | Target: {trade.take_profit:.2f}"
            await send_alert("Trade Opened", alert_msg)
        else:
            trade.status = TradeStatus.CANCELLED
            logger.warning(
                "engine.trade_cancelled_after_retries",
                symbol=signal.symbol,
                attempts=max_retries,
            )
            await send_alert(
                "Trade Cancelled",
                f"{signal.symbol}: BUY {quantity} failed after {max_retries} attempt(s)\n"
                f"Strategy: {signal.strategy_name} | Last status: {result.status}",
            )

        await self._db.flush()

    async def _handle_order_fill(self, fill: dict) -> None:
        """Update trade record when a pending order is filled via polling."""
        trade_id = fill["trade_id"]
        symbol = fill["symbol"]
        side = fill["side"]
        filled_price = fill.get("filled_price")

        trade = self._open_trades.get(symbol)
        if not trade or trade.id != trade_id:
            # Try loading from DB
            try:
                result = await self._db.execute(
                    select(Trade).where(Trade.id == trade_id)
                )
                trade = result.scalar_one_or_none()
            except Exception:
                logger.exception("engine.fill_load_error", trade_id=trade_id)
                return

        if not trade:
            logger.warning("engine.fill_no_trade", trade_id=trade_id, symbol=symbol)
            return

        if side == "BUY" and trade.status in (TradeStatus.PENDING, TradeStatus.OPEN):
            if trade.status == TradeStatus.PENDING:
                trade.status = TradeStatus.OPEN
                trade.entry_price = filled_price
                self._open_trades[symbol] = trade
                logger.info(
                    "engine.trade_opened_on_fill",
                    symbol=symbol,
                    price=filled_price,
                )
                # Place stop-loss for the newly filled buy (ATR-based)
                if filled_price:
                    stop_price = self._calculate_atr_stop(filled_price, symbol)
                    trade.stop_loss = stop_price
                    await self._place_stop_verified(
                        trade.id, symbol, trade.quantity, stop_price
                    )

                    # Set take-profit target
                    atr_val = self._get_atr(symbol)
                    trade.take_profit = calculate_take_profit(
                        filled_price, symbol, atr_val
                    )
        elif side == "SELL":
            # Cancel any remaining pending SELL orders (e.g. stop-loss) for this symbol
            await self._cancel_pending_sells(symbol)
            if filled_price:
                await self._portfolio_tracker.record_trade_close(trade, filled_price)
            self._open_trades.pop(symbol, None)
            self._eod_sell_pending.discard(symbol)
            self._last_close_time[symbol] = datetime.now(UTC)
            pnl = trade.realized_pnl or 0.0
            hold_mins = ""
            if trade.created_at:
                delta = datetime.now(UTC) - trade.created_at
                hold_mins = f" | Hold: {delta.total_seconds() / 60:.0f}min"
            logger.info(
                "engine.position_closed_on_fill",
                symbol=symbol,
                exit_price=filled_price,
            )
            # Record outcome for dynamic strategy weight adjustment
            for strategy in self._strategies:
                if hasattr(strategy, "record_outcome"):
                    strategy.record_outcome(symbol, trade.strategy_name, pnl > 0)
            self._record_adaptive_outcome(trade)
            await send_alert(
                "Trade Closed",
                f"{symbol}: SOLD {trade.quantity} @ {filled_price:.2f}\n"
                f"P&L: {pnl:+.2f}{hold_mins}",
            )

        await self._db.flush()

    async def _retry_pending_stops(self) -> None:
        """Retry placing stop-loss orders that failed due to IBKR race condition.

        After a BUY fill, IBKR sometimes hasn't registered the position yet,
        causing the immediate STOP SELL order to be rejected. This retries
        those placements in the next cycle when the position is confirmed.
        """
        if not self._pending_stop_retries:
            return

        for symbol in list(self._pending_stop_retries):
            trade = self._open_trades.get(symbol)
            if not trade or not trade.stop_loss or trade.stop_loss <= 0:
                self._pending_stop_retries.discard(symbol)
                continue

            # Cancel any stale stop orders before retrying
            try:
                await self._broker.cancel_open_orders_for_symbol(symbol)
                await asyncio.sleep(0.5)
            except Exception:
                pass

            success = await self._place_stop_verified(
                trade.id, symbol, trade.quantity, trade.stop_loss
            )
            if success:
                self._pending_stop_retries.discard(symbol)

    async def _check_trailing_stops(self, prices: dict[str, float]) -> None:
        """Update trailing stops for open positions using progressive tiers."""
        tiers = settings.trailing_stop_tiers_parsed
        for symbol, trade in list(self._open_trades.items()):
            if trade.entry_price is None or trade.stop_loss is None:
                continue

            current_price = prices.get(symbol)
            if current_price is None:
                continue

            atr_val = self._get_atr(trade.symbol)

            # Breakeven stop: move stop to entry once position is up enough
            gain_pct = (current_price - trade.entry_price) / trade.entry_price * 100
            if (
                settings.breakeven_stop_trigger_pct > 0
                and gain_pct >= settings.breakeven_stop_trigger_pct
                and trade.stop_loss < trade.entry_price
            ):
                # Move stop to entry + small buffer (0.1%) to cover commissions
                breakeven_price = round(trade.entry_price * 1.001, 2)
                if breakeven_price > trade.stop_loss:
                    # CRITICAL: Verify position still exists before replacing stop
                    if not await self._verify_position_or_record_close(symbol, trade):
                        continue  # Position was closed by stop fill — skip

                    old_stop = trade.stop_loss
                    trade.stop_loss = breakeven_price
                    self._last_stop_update_price[symbol] = current_price
                    try:
                        await self._cancel_pending_sells(symbol)
                        await asyncio.sleep(0.5)
                        await self._place_stop_verified(
                            trade.id, symbol, trade.quantity, breakeven_price
                        )
                        logger.info(
                            "engine.breakeven_stop_set",
                            symbol=symbol,
                            old_stop=old_stop,
                            breakeven=breakeven_price,
                            gain_pct=round(gain_pct, 2),
                        )
                    except Exception:
                        logger.exception("engine.breakeven_stop_failed", symbol=symbol)
                        self._pending_stop_retries.add(symbol)
                    continue  # Don't also run progressive trailing this cycle

            # Progressive trailing: tighter stops as profit grows
            new_stop = calculate_progressive_trailing_stop(
                entry_price=trade.entry_price,
                current_price=current_price,
                current_stop=trade.stop_loss,
                atr=atr_val,
                tiers=tiers,
            )

            # Fall back to standard trailing if progressive didn't tighten
            if new_stop <= trade.stop_loss:
                new_stop = calculate_trailing_stop(
                    entry_price=trade.entry_price,
                    current_price=current_price,
                    atr=atr_val,
                    trail_pct=settings.min_stop_loss_pct,
                )

            if new_stop > trade.stop_loss:
                # Whipsaw protection: only update if price moved >1% since last update
                last_update_price = self._last_stop_update_price.get(symbol, 0.0)
                if last_update_price > 0:
                    price_move = abs(current_price - last_update_price)
                    price_change_pct = price_move / last_update_price * 100
                    if price_change_pct < 1.0:
                        continue

                # CRITICAL: Verify position still exists before replacing stop
                if not await self._verify_position_or_record_close(symbol, trade):
                    continue  # Position was closed by stop fill — skip

                old_stop = trade.stop_loss
                trade.stop_loss = new_stop
                self._last_stop_update_price[symbol] = current_price

                # CRITICAL: Replace the stop order at IBKR, not just in DB
                try:
                    # Cancel old stop at both OrderManager and broker level
                    await self._cancel_pending_sells(symbol)
                    await asyncio.sleep(0.5)  # Let IBKR process cancellation

                    # Place new stop at updated price
                    success = await self._place_stop_verified(
                        trade.id, symbol, trade.quantity, new_stop
                    )
                    if success:
                        logger.info(
                            "engine.trailing_stop_replaced",
                            symbol=symbol,
                            old_stop=old_stop,
                            new_stop=new_stop,
                            current_price=current_price,
                        )
                except Exception:
                    logger.exception(
                        "engine.trailing_stop_replace_failed",
                        symbol=symbol,
                        new_stop=new_stop,
                    )
                    self._pending_stop_retries.add(symbol)

    async def _check_take_profits(self, prices: dict[str, float]) -> None:
        """Close positions that have reached their take-profit target.

        If partial_profit_enabled: sell 50% at first target, move stop to
        breakeven on the rest, and set a new higher target for the remainder.
        """
        if not self._trading_enabled:
            return

        for symbol, trade in list(self._open_trades.items()):
            if trade.take_profit is None or trade.entry_price is None:
                continue
            if trade.status != TradeStatus.OPEN:
                continue

            current_price = prices.get(symbol)
            if current_price is None:
                continue

            if current_price >= trade.take_profit:
                # Partial profit-taking: sell half, trail the rest
                partial_qty = trade.quantity // 2
                already_partial = trade.partial_taken

                if (
                    settings.partial_profit_enabled
                    and partial_qty >= 1
                    and trade.quantity >= 2
                    and not already_partial
                ):
                    logger.info(
                        "engine.partial_take_profit",
                        symbol=symbol,
                        sell_qty=partial_qty,
                        keep_qty=trade.quantity - partial_qty,
                        current_price=current_price,
                        take_profit=trade.take_profit,
                    )
                    # Cancel existing stop before partial sell
                    await self._cancel_pending_sells(symbol)
                    remaining = len(self._order_manager._pending_by_symbol.get(symbol, set()))
                    if remaining > 0:
                        logger.warning("engine.tp_cancel_incomplete", symbol=symbol)
                        continue

                    result = await self._order_manager.submit_order(
                        trade_id=trade.id,
                        symbol=symbol,
                        side="SELL",
                        quantity=partial_qty,
                        order_type=OrderType.MARKET,
                        expected_price=current_price,
                    )
                    result = await self._await_market_fill(result)
                    mapped_status = self._order_manager._map_status(result.status)
                    if mapped_status == OrderStatus.FILLED and result.filled_price:
                        # Update trade: reduce quantity, keep position open
                        trade.quantity -= partial_qty
                        trade.partial_taken = True
                        # Move stop to breakeven + buffer on remaining shares
                        breakeven = round(trade.entry_price * 1.001, 2)
                        trade.stop_loss = breakeven
                        # Set new take-profit 50% higher than original distance
                        original_distance = trade.take_profit - trade.entry_price
                        trade.take_profit = round(
                            trade.entry_price + original_distance * 1.5, 2
                        )
                        # Place new stop for reduced quantity
                        await asyncio.sleep(0.5)
                        await self._place_stop_verified(
                            trade.id, symbol, trade.quantity, breakeven
                        )
                        partial_pnl = (result.filled_price - trade.entry_price) * partial_qty
                        await send_alert(
                            "Partial Profit Taken",
                            f"{symbol}: sold {partial_qty} @ {result.filled_price:.2f} "
                            f"(+{partial_pnl:+.2f})\n"
                            f"Keeping {trade.quantity} shares, stop→breakeven, "
                            f"new target: {trade.take_profit:.2f}",
                        )
                    continue

                # Full take-profit (1 share positions, or second target hit)
                logger.info(
                    "engine.take_profit_triggered",
                    symbol=symbol,
                    current_price=current_price,
                    take_profit=trade.take_profit,
                    entry_price=trade.entry_price,
                )
                await self._cancel_pending_sells(symbol)
                remaining = len(self._order_manager._pending_by_symbol.get(symbol, set()))
                if remaining > 0:
                    logger.warning("engine.tp_cancel_incomplete", symbol=symbol)
                    continue

                result = await self._order_manager.submit_order(
                    trade_id=trade.id,
                    symbol=symbol,
                    side="SELL",
                    quantity=trade.quantity,
                    order_type=OrderType.MARKET,
                    expected_price=current_price,
                )
                result = await self._await_market_fill(result)
                mapped_status = self._order_manager._map_status(result.status)
                if mapped_status == OrderStatus.FILLED and result.filled_price:
                    await self._portfolio_tracker.record_trade_close(
                        trade, result.filled_price
                    )
                    self._open_trades.pop(symbol, None)
                    self._eod_sell_pending.discard(symbol)
                    self._last_close_time[symbol] = datetime.now(UTC)
                    self._record_adaptive_outcome(trade)
                    partial_note = " (2nd target)" if trade.partial_taken else ""
                    await send_alert(
                        "Take-Profit Hit",
                        f"{symbol}: sold {trade.quantity} @ {result.filled_price:.2f} "
                        f"(target {trade.take_profit:.2f}, entry {trade.entry_price:.2f}){partial_note}",
                    )

    async def _check_eod_close(self, prices: dict[str, float]) -> None:
        """Force-close all positions when market is about to close (day trading rule)."""
        if not self._trading_enabled:
            return

        now = datetime.now(UTC)

        for symbol, trade in list(self._open_trades.items()):
            if trade.status != TradeStatus.OPEN:
                continue

            # Skip if we already submitted an EOD sell for this symbol
            if symbol in self._eod_sell_pending:
                continue

            mins_left = minutes_until_close_for_symbol(
                symbol, now, include_extended=settings.extended_hours_enabled,
            )
            if mins_left is None:
                continue  # Market not open / already closed

            if mins_left <= settings.eod_close_minutes_before:
                current_price = prices.get(symbol, 0.0)
                logger.info(
                    "engine.eod_close_triggered",
                    symbol=symbol,
                    minutes_left=round(mins_left, 1),
                    current_price=current_price,
                )
                # Cancel pending SELL orders (stop-loss) before EOD close
                await self._cancel_pending_sells(symbol)
                # Wait briefly for IBKR to process the cancel
                await asyncio.sleep(1)

                # Mark as pending to prevent duplicate sells
                self._eod_sell_pending.add(symbol)

                result = await self._order_manager.submit_order(
                    trade_id=trade.id,
                    symbol=symbol,
                    side="SELL",
                    quantity=trade.quantity,
                    order_type=OrderType.MARKET,
                    expected_price=current_price,
                )
                result = await self._await_market_fill(result)
                mapped_status = self._order_manager._map_status(result.status)
                if mapped_status == OrderStatus.FILLED and result.filled_price:
                    await self._portfolio_tracker.record_trade_close(
                        trade, result.filled_price
                    )
                    self._open_trades.pop(symbol, None)
                    self._eod_sell_pending.discard(symbol)
                    self._last_close_time[symbol] = datetime.now(UTC)
                    pnl = trade.realized_pnl or 0.0
                    # Record outcome for dynamic strategy weight adjustment
                    for strategy in self._strategies:
                        if hasattr(strategy, "record_outcome"):
                            strategy.record_outcome(symbol, trade.strategy_name, pnl > 0)
                    self._record_adaptive_outcome(trade)
                    await send_alert(
                        "EOD Close",
                        f"{symbol}: closed at {result.filled_price:.2f} "
                        f"({mins_left:.0f} min before close, P&L: {pnl:+.2f})",
                    )

    async def _check_stale_positions(self, prices: dict[str, float]) -> None:
        """Close positions held too long with negligible P&L.

        Day trading positions that sit flat for hours tie up capital and
        often end up losing when volatility picks up. Better to cut them
        and redeploy capital.
        """
        if not self._trading_enabled or settings.stale_position_hours <= 0:
            return

        now = datetime.now(UTC)
        max_hold_seconds = settings.stale_position_hours * 3600

        for symbol, trade in list(self._open_trades.items()):
            if trade.status != TradeStatus.OPEN or trade.entry_price is None:
                continue
            if not trade.created_at:
                continue
            # Skip if pending EOD sell
            if symbol in self._eod_sell_pending:
                continue

            held_seconds = (now - trade.created_at).total_seconds()
            if held_seconds < max_hold_seconds:
                continue

            current_price = prices.get(symbol)
            if current_price is None or current_price <= 0:
                continue

            pnl_pct = abs(current_price - trade.entry_price) / trade.entry_price * 100
            if pnl_pct >= settings.stale_position_min_pnl_pct:
                continue  # Position has meaningful movement, keep it

            held_hours = held_seconds / 3600
            logger.info(
                "engine.stale_position_close",
                symbol=symbol,
                held_hours=round(held_hours, 1),
                pnl_pct=round(pnl_pct, 2),
            )

            await self._cancel_pending_sells(symbol)
            await asyncio.sleep(0.5)

            result = await self._order_manager.submit_order(
                trade_id=trade.id,
                symbol=symbol,
                side="SELL",
                quantity=trade.quantity,
                order_type=OrderType.MARKET,
                expected_price=current_price,
            )
            result = await self._await_market_fill(result)
            mapped_status = self._order_manager._map_status(result.status)
            if mapped_status == OrderStatus.FILLED and result.filled_price:
                await self._portfolio_tracker.record_trade_close(
                    trade, result.filled_price
                )
                self._open_trades.pop(symbol, None)
                self._eod_sell_pending.discard(symbol)
                self._last_close_time[symbol] = datetime.now(UTC)
                pnl = trade.realized_pnl or 0.0
                for strategy in self._strategies:
                    if hasattr(strategy, "record_outcome"):
                        strategy.record_outcome(symbol, trade.strategy_name, pnl > 0)
                self._record_adaptive_outcome(trade)
                await send_alert(
                    "Stale Position Closed",
                    f"{symbol}: closed at {result.filled_price:.2f} after "
                    f"{held_hours:.1f}h with {pnl_pct:.2f}% P&L ({pnl:+.2f})",
                )

    async def _check_max_hold_days(self, prices: dict[str, float]) -> None:
        """Force-close positions held longer than max_hold_days trading days.

        For swing trading, we want to exit positions that haven't hit take-profit
        or stop-loss within the expected holding window (matches ML model's
        forward_periods training target).
        """
        if not self._trading_enabled or settings.max_hold_days <= 0:
            return

        now = datetime.now(UTC)
        max_hold_seconds = settings.max_hold_days * 24 * 3600  # Calendar days as proxy

        for symbol, trade in list(self._open_trades.items()):
            if trade.status != TradeStatus.OPEN or trade.entry_price is None:
                continue
            if not trade.created_at:
                continue
            if symbol in self._eod_sell_pending:
                continue

            held_seconds = (now - trade.created_at).total_seconds()
            if held_seconds < max_hold_seconds:
                continue

            current_price = prices.get(symbol)
            if current_price is None or current_price <= 0:
                continue

            held_days = held_seconds / 86400
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price * 100
            logger.info(
                "engine.max_hold_close",
                symbol=symbol,
                held_days=round(held_days, 1),
                pnl_pct=round(pnl_pct, 2),
            )

            await self._cancel_pending_sells(symbol)
            await asyncio.sleep(0.5)

            result = await self._order_manager.submit_order(
                trade_id=trade.id,
                symbol=symbol,
                side="SELL",
                quantity=trade.quantity,
                order_type=OrderType.MARKET,
                expected_price=current_price,
            )
            result = await self._await_market_fill(result)
            mapped_status = self._order_manager._map_status(result.status)
            if mapped_status == OrderStatus.FILLED and result.filled_price:
                await self._portfolio_tracker.record_trade_close(
                    trade, result.filled_price
                )
                self._open_trades.pop(symbol, None)
                self._eod_sell_pending.discard(symbol)
                self._last_close_time[symbol] = datetime.now(UTC)
                pnl = trade.realized_pnl or 0.0
                for strategy in self._strategies:
                    if hasattr(strategy, "record_outcome"):
                        strategy.record_outcome(symbol, trade.strategy_name, pnl > 0)
                self._record_adaptive_outcome(trade)
                await send_alert(
                    "Max Hold Close",
                    f"{symbol}: closed at {result.filled_price:.2f} after "
                    f"{held_days:.1f} days (P&L: {pnl_pct:+.1f}%, ${pnl:+.2f})",
                )

    async def _load_open_trades(self) -> None:
        """Load open trades from database on startup for recovery.

        Only loads OPEN trades (with entry_price).  Stale PENDING trades
        (no entry_price, never filled) are cancelled to avoid blocking
        new trades for those symbols.

        Re-places stop-loss orders at IBKR for all loaded trades since
        broker-side orders may have been cancelled during restart.
        """
        try:
            result = await self._db.execute(
                select(Trade).where(Trade.status.in_([TradeStatus.OPEN, TradeStatus.PENDING]))
            )
            trades = result.scalars().all()
            loaded = 0
            cancelled = 0
            ghost_closed = 0
            stops_placed = 0

            # Get actual broker positions to detect ghost trades
            broker_positions = {}
            try:
                portfolio = await self._broker.get_portfolio()
                for pos in portfolio.positions:
                    broker_positions[pos.symbol] = pos.quantity
            except Exception:
                logger.warning("engine.load_trades_no_portfolio")

            for trade in trades:
                if trade.status == TradeStatus.OPEN and trade.entry_price is not None:
                    # Verify broker actually has a position for this trade
                    broker_qty = broker_positions.get(trade.symbol, 0)
                    if broker_positions and broker_qty == 0:
                        # Ghost trade: DB says open but broker has no position
                        trade.status = TradeStatus.CLOSED
                        trade.realized_pnl = trade.realized_pnl or 0.0
                        ghost_closed += 1
                        logger.warning(
                            "engine.ghost_trade_closed",
                            symbol=trade.symbol,
                            quantity=trade.quantity,
                            entry_price=trade.entry_price,
                        )
                        continue

                    self._open_trades[trade.symbol] = trade
                    loaded += 1

                    # Cancel any stale broker-side orders for this symbol
                    # to prevent duplicate stop-losses accumulating across restarts.
                    try:
                        stale = await self._broker.cancel_open_orders_for_symbol(trade.symbol)
                        if stale:
                            logger.info(
                                "engine.cancelled_stale_orders_on_load",
                                symbol=trade.symbol,
                                count=stale,
                            )
                    except Exception:
                        logger.warning(
                            "engine.cancel_stale_orders_failed",
                            symbol=trade.symbol,
                        )

                    # Re-place stop-loss at broker (may have been lost on restart)
                    if trade.stop_loss and trade.stop_loss > 0:
                        success = await self._place_stop_verified(
                            trade.id, trade.symbol, trade.quantity, trade.stop_loss
                        )
                        if success:
                            stops_placed += 1

                    # Set take-profit if missing
                    if not trade.take_profit and trade.entry_price:
                        atr_val = self._get_atr(trade.symbol)
                        tp = calculate_take_profit(trade.entry_price, trade.symbol, atr_val)
                        trade.take_profit = tp

                elif trade.status == TradeStatus.PENDING:
                    # Stale pending — never filled, cancel it
                    trade.status = TradeStatus.CANCELLED
                    cancelled += 1
            if loaded or cancelled or ghost_closed:
                logger.info(
                    "engine.loaded_open_trades",
                    count=loaded,
                    stale_cancelled=cancelled,
                    ghost_closed=ghost_closed,
                    stops_restored=stops_placed,
                )
                if cancelled or ghost_closed:
                    parts = []
                    if cancelled:
                        parts.append(f"{cancelled} stale PENDING")
                    if ghost_closed:
                        parts.append(f"{ghost_closed} ghost (no broker position)")
                    await send_alert(
                        "Stale Trades Cleaned",
                        f"Cleaned up on startup: {', '.join(parts)}",
                    )
            await self._db.flush()
        except Exception:
            logger.exception("engine.load_trades_error")

    async def _reconcile_stale_orders(self) -> None:
        """Reconcile SUBMITTED orders from before restart by checking IBKR status.

        After a restart, _pending_orders is empty so SUBMITTED orders in the DB
        would never be polled again.  This queries IBKR for the actual status of
        each stale order and updates the DB (recording fills or cancellations).
        """
        try:
            result = await self._db.execute(
                select(DBOrder).where(
                    DBOrder.status.in_([OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED])
                )
            )
            stale_orders = result.scalars().all()
            if not stale_orders:
                return

            updated = 0
            filled = 0
            for order in stale_orders:
                if not order.broker_order_id:
                    continue
                try:
                    broker_result = await self._broker.get_order_status(order.broker_order_id)
                    new_status = self._order_manager._map_status(broker_result.status)

                    if new_status == order.status:
                        continue

                    order.status = new_status
                    updated += 1

                    if new_status == OrderStatus.FILLED:
                        order.filled_price = broker_result.filled_price
                        order.filled_quantity = broker_result.filled_quantity
                        order.filled_at = datetime.now(UTC)
                        if order.expected_price and broker_result.filled_price:
                            order.slippage = broker_result.filled_price - order.expected_price
                        filled += 1

                        # Update trade exit price/P&L if this was a SELL fill
                        if order.side == "SELL" and broker_result.filled_price:
                            trade_result = await self._db.execute(
                                select(Trade).where(Trade.id == order.trade_id)
                            )
                            trade = trade_result.scalar_one_or_none()
                            if trade and not trade.exit_price:
                                trade.exit_price = broker_result.filled_price
                                if trade.entry_price:
                                    trade.realized_pnl = (
                                        (broker_result.filled_price - trade.entry_price)
                                        * order.quantity
                                    )
                                trade.closed_at = trade.closed_at or datetime.now(UTC)
                                logger.info(
                                    "engine.stale_sell_reconciled",
                                    symbol=order.symbol,
                                    exit_price=broker_result.filled_price,
                                    pnl=trade.realized_pnl,
                                )

                except Exception:
                    logger.debug(
                        "engine.stale_order_check_failed",
                        order_id=order.broker_order_id,
                        symbol=order.symbol,
                        exc_info=True,
                    )

            if updated:
                await self._db.flush()
                logger.info(
                    "engine.stale_orders_reconciled",
                    total=len(stale_orders),
                    updated=updated,
                    filled=filled,
                )
        except Exception:
            logger.exception("engine.stale_order_reconcile_error")

    def _record_adaptive_outcome(self, trade: Trade) -> None:
        """Record trade outcome to the adaptive learning system."""
        pnl = trade.realized_pnl or 0.0
        regime = getattr(getattr(self._regime_detector, "current_regime", None), "regime", None)
        regime_name = regime.value if regime else "unknown"
        hold_minutes = 0.0
        if trade.created_at:
            hold_minutes = (datetime.now(UTC) - trade.created_at).total_seconds() / 60
        # Get top features from the model for feature effectiveness tracking
        top_features = []
        for s in self._strategies:
            if hasattr(s, "_model_metadata"):
                importance = s._model_metadata.get("feature_importance", {})
                top_features = list(importance.keys())[:10]
                break
            if hasattr(s, "_strategies"):
                for sub in s._strategies:
                    if hasattr(sub, "_model_metadata"):
                        importance = sub._model_metadata.get("feature_importance", {})
                        top_features = list(importance.keys())[:10]
                        break
        self._adaptive.record_trade_outcome(
            symbol=trade.symbol,
            pnl=pnl,
            confidence=0.0,  # Not stored on trade, use 0
            strategy_name=trade.strategy_name or "unknown",
            regime=regime_name,
            hold_minutes=hold_minutes,
            top_features=top_features,
        )

    def reset_daily(self) -> None:
        """Reset daily counters. Called at start of each trading day."""
        self._risk_manager.set_daily_start_value(
            self._portfolio_tracker._last_portfolio.account_summary.total_value
            if self._portfolio_tracker._last_portfolio
            else 0.0
        )
        self._cycle_count = 0
        self._last_close_time.clear()
        self._last_stop_update_price.clear()
        self._daily_symbol_trades.clear()
        self._eod_sell_pending.clear()
        logger.info("engine.daily_reset")

    def get_status(self) -> dict:
        """Return current engine status for the API."""
        regime = self._regime_detector.current_regime
        # Collect AI sizing API costs from risk manager
        ai_advisor = getattr(self._risk_manager, "_ai_advisor", None)
        api_costs = None
        if ai_advisor:
            api_costs = {
                "calls": ai_advisor.call_count,
                "estimated_cost_usd": round(ai_advisor.estimated_cost_usd, 6),
            }
        return {
            "state": self._state.value,
            "trading_enabled": self._trading_enabled,
            "cycle_count": self._cycle_count,
            "last_cycle_at": (
                self._last_cycle_at.isoformat() if self._last_cycle_at else None
            ),
            "open_trades": len(self._open_trades),
            "pending_orders": self._order_manager.pending_count,
            "symbols": self._symbols,
            "strategies": [s.name for s in self._strategies],
            "reconnect_attempts": self._reconnect_attempts,
            "market_regime": regime.to_dict() if regime else None,
            "api_costs": api_costs,
            "adaptive_learning": {
                "total_outcomes": len(self._adaptive._outcomes),
                "symbols_tracked": len(self._adaptive._symbol_profiles),
                "regimes_tracked": len(self._adaptive._regime_profiles),
                "declining_features": self._adaptive.get_declining_features(),
            },
        }
