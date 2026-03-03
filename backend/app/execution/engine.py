"""Main trading orchestrator - the heart of the system."""

import asyncio
from datetime import UTC, datetime
from enum import StrEnum

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.broker.base import BrokerAdapter, OrderType
from app.config import settings
from app.core.event_bus import RECONCILIATION_UPDATE, RISK_DAILY_STOP, SIGNAL_GENERATED, SYSTEM_ERROR, event_bus
from app.data.correlation import get_correlation_matrix
from app.data.market_data import MarketDataService
from app.execution.order_manager import OrderManager
from app.execution.portfolio_tracker import PortfolioTracker
from app.execution.smart_execution import SmartExecutor
from app.models.order import OrderStatus
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

            # Initialize daily tracking
            start_value = await self._portfolio_tracker.initialize_daily()
            self._risk_manager.set_daily_start_value(start_value)

            # Load open trades from DB
            await self._load_open_trades()

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

            # 2. Get market data
            snapshot = await self._market_data.get_snapshot(self._symbols)
            # Cache snapshot features for ATR lookups in _execute_buy
            if snapshot.features:
                self._snapshot_features = snapshot.features

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

            # 2c. Detect market regime
            try:
                regime_state = self._regime_detector.detect(snapshot)
            except Exception:
                logger.debug("regime.detection_error", exc_info=True)
                regime_state = None

            # 3. EOD close — force-sell positions near market close (day trading)
            await self._check_eod_close(snapshot.prices)

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
            # If ensemble exists, run only that (it already calls sub-strategies
            # in parallel internally). Otherwise run all strategies.
            has_ensemble = any(s.name == "ensemble" for s in self._strategies)
            strategies_to_run = (
                [s for s in self._strategies if s.name == "ensemble"]
                if has_ensemble
                else self._strategies
            )
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

            # 6b. Multi-timeframe confirmation
            confirmed_signals: list[TradingSignal] = []
            for signal in all_signals:
                if signal.action == SignalAction.HOLD:
                    confirmed_signals.append(signal)
                    continue
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
                if signal.action == SignalAction.BUY and signal.symbol in settings.symbol_blacklist_set:
                    logger.info(
                        "engine.signal_skipped_blacklist",
                        symbol=signal.symbol,
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

                await self._execute_buy(
                    signal, decision.adjusted_quantity or 1, price, db_signal.id
                )

            await self._db.commit()

            # 9. Take portfolio snapshot
            await self._portfolio_tracker.take_snapshot()
            await self._db.commit()

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

        mapped_status = self._order_manager._map_status(result.status)
        if mapped_status == OrderStatus.FILLED and result.filled_price:
            await self._portfolio_tracker.record_trade_close(
                open_trade, result.filled_price
            )
            self._open_trades.pop(signal.symbol, None)
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
        """
        cancelled = await self._order_manager.cancel_orders_for_symbol(symbol, side="SELL")
        if cancelled:
            logger.info(
                "engine.cancelled_pending_sells",
                symbol=symbol,
                count=cancelled,
            )
        return cancelled

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
        if risk > 0 and (reward / risk) < 1.5:
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
            if result.filled_price:
                stop_price = self._calculate_atr_stop(result.filled_price, signal.symbol)

                # Slippage correction: tighten stop if we got a worse entry
                slippage = result.filled_price - price if price > 0 else 0
                if slippage > 0:
                    # Bought higher than expected — tighten stop proportionally
                    stop_price = round(stop_price + slippage * 0.5, 2)

                trade.stop_loss = stop_price
                try:
                    await self._order_manager.submit_order(
                        trade_id=trade.id,
                        symbol=signal.symbol,
                        side="SELL",
                        quantity=quantity,
                        order_type=OrderType.STOP,
                        stop_price=stop_price,
                    )
                except Exception:
                    logger.exception(
                        "engine.stop_loss_placement_failed",
                        symbol=signal.symbol,
                        stop_price=stop_price,
                    )
                    self._pending_stop_retries.add(signal.symbol)

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
                    try:
                        await self._order_manager.submit_order(
                            trade_id=trade.id,
                            symbol=symbol,
                            side="SELL",
                            quantity=trade.quantity,
                            order_type=OrderType.STOP,
                            stop_price=stop_price,
                        )
                    except Exception:
                        logger.exception(
                            "engine.stop_loss_on_fill_failed",
                            symbol=symbol,
                            stop_price=stop_price,
                        )
                        self._pending_stop_retries.add(symbol)

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

            try:
                await self._order_manager.submit_order(
                    trade_id=trade.id,
                    symbol=symbol,
                    side="SELL",
                    quantity=trade.quantity,
                    order_type=OrderType.STOP,
                    stop_price=trade.stop_loss,
                )
                self._pending_stop_retries.discard(symbol)
                logger.info(
                    "engine.stop_loss_retry_success",
                    symbol=symbol,
                    stop_price=trade.stop_loss,
                )
            except Exception:
                logger.warning(
                    "engine.stop_loss_retry_failed",
                    symbol=symbol,
                    stop_price=trade.stop_loss,
                )

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

                trade.stop_loss = new_stop
                self._last_stop_update_price[symbol] = current_price
                logger.debug(
                    "engine.trailing_stop_updated",
                    symbol=symbol,
                    new_stop=new_stop,
                    current_price=current_price,
                )

    async def _check_take_profits(self, prices: dict[str, float]) -> None:
        """Close positions that have reached their take-profit target."""
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
                logger.info(
                    "engine.take_profit_triggered",
                    symbol=symbol,
                    current_price=current_price,
                    take_profit=trade.take_profit,
                    entry_price=trade.entry_price,
                )
                # Cancel pending SELL orders (stop-loss) before closing
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
                mapped_status = self._order_manager._map_status(result.status)
                if mapped_status == OrderStatus.FILLED and result.filled_price:
                    await self._portfolio_tracker.record_trade_close(
                        trade, result.filled_price
                    )
                    self._open_trades.pop(symbol, None)
                    self._last_close_time[symbol] = datetime.now(UTC)
                    await send_alert(
                        "Take-Profit Hit",
                        f"{symbol}: sold at {result.filled_price:.2f} "
                        f"(target {trade.take_profit:.2f}, entry {trade.entry_price:.2f})",
                    )

    async def _check_eod_close(self, prices: dict[str, float]) -> None:
        """Force-close all positions when market is about to close (day trading rule)."""
        if not self._trading_enabled:
            return

        now = datetime.now(UTC)

        for symbol, trade in list(self._open_trades.items()):
            if trade.status != TradeStatus.OPEN:
                continue

            mins_left = minutes_until_close_for_symbol(symbol, now)
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
                remaining = len(self._order_manager._pending_by_symbol.get(symbol, set()))
                if remaining > 0:
                    logger.warning("engine.eod_cancel_incomplete", symbol=symbol)
                    continue

                result = await self._order_manager.submit_order(
                    trade_id=trade.id,
                    symbol=symbol,
                    side="SELL",
                    quantity=trade.quantity,
                    order_type=OrderType.MARKET,
                    expected_price=current_price,
                )
                mapped_status = self._order_manager._map_status(result.status)
                if mapped_status == OrderStatus.FILLED and result.filled_price:
                    await self._portfolio_tracker.record_trade_close(
                        trade, result.filled_price
                    )
                    self._open_trades.pop(symbol, None)
                    self._last_close_time[symbol] = datetime.now(UTC)
                    pnl = trade.realized_pnl or 0.0
                    # Record outcome for dynamic strategy weight adjustment
                    for strategy in self._strategies:
                        if hasattr(strategy, "record_outcome"):
                            strategy.record_outcome(symbol, trade.strategy_name, pnl > 0)
                    await send_alert(
                        "EOD Close",
                        f"{symbol}: closed at {result.filled_price:.2f} "
                        f"({mins_left:.0f} min before close, P&L: {pnl:+.2f})",
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
            stops_placed = 0
            for trade in trades:
                if trade.status == TradeStatus.OPEN and trade.entry_price is not None:
                    self._open_trades[trade.symbol] = trade
                    loaded += 1

                    # Re-place stop-loss at broker (may have been lost on restart)
                    if trade.stop_loss and trade.stop_loss > 0:
                        try:
                            await self._order_manager.submit_order(
                                trade_id=trade.id,
                                symbol=trade.symbol,
                                side="SELL",
                                quantity=trade.quantity,
                                order_type=OrderType.STOP,
                                stop_price=trade.stop_loss,
                            )
                            stops_placed += 1
                        except Exception:
                            logger.exception(
                                "engine.stop_loss_restore_failed",
                                symbol=trade.symbol,
                                stop_price=trade.stop_loss,
                            )

                    # Set take-profit if missing
                    if not trade.take_profit and trade.entry_price:
                        atr_val = self._get_atr(trade.symbol)
                        tp = calculate_take_profit(trade.entry_price, trade.symbol, atr_val)
                        trade.take_profit = tp

                elif trade.status == TradeStatus.PENDING:
                    # Stale pending — never filled, cancel it
                    trade.status = TradeStatus.CANCELLED
                    cancelled += 1
            if loaded or cancelled:
                logger.info(
                    "engine.loaded_open_trades",
                    count=loaded,
                    stale_cancelled=cancelled,
                    stops_restored=stops_placed,
                )
                if cancelled:
                    await send_alert(
                        "Stale Orders Cleaned",
                        f"Cancelled {cancelled} stale PENDING trade(s) on startup.",
                    )
            await self._db.flush()
        except Exception:
            logger.exception("engine.load_trades_error")

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
        }
