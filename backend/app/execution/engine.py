"""Main trading orchestrator - the heart of the system."""

import asyncio
from datetime import datetime, timezone
from enum import Enum

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import BrokerAdapter, OrderType
from app.core.event_bus import event_bus, SIGNAL_GENERATED, RISK_DAILY_STOP, SYSTEM_ERROR
from app.data.market_data import MarketDataService
from app.execution.order_manager import OrderManager
from app.execution.portfolio_tracker import PortfolioTracker
from app.models.order import OrderStatus
from app.models.signal import Signal as DBSignal, SignalAction as DBSignalAction
from app.models.trade import Trade, TradeSide, TradeStatus
from app.monitoring.alerts import send_alert
from app.monitoring.performance import PerformanceTracker
from app.risk.manager import RiskManager
from app.risk.position_sizer import calculate_trailing_stop
from app.strategy.base import SignalAction, Strategy, TradingSignal

logger = structlog.get_logger()


class EngineState(str, Enum):
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
    ):
        self._broker = broker
        self._strategies = strategies
        self._risk_manager = risk_manager
        self._market_data = market_data
        self._performance = performance
        self._db = db
        self._symbols = symbols
        self._trading_enabled = trading_enabled
        self._order_manager = OrderManager(broker, db)
        self._portfolio_tracker = PortfolioTracker(broker, db, performance)
        self._state = EngineState.STOPPED
        self._cycle_count = 0
        self._last_cycle_at: datetime | None = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        # Track open trades by symbol for close logic
        self._open_trades: dict[str, Trade] = {}

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
        """Run one complete trading cycle: data -> signals -> risk -> orders."""
        if self._state != EngineState.RUNNING:
            return

        try:
            # Check broker connectivity
            if not await self._ensure_connected():
                return

            # 1. Poll pending orders for status updates
            filled_orders = await self._order_manager.poll_pending_orders()
            for fill in filled_orders:
                await self._handle_order_fill(fill)

            # 2. Get market data
            snapshot = await self._market_data.get_snapshot(self._symbols)

            # 3. Check trailing stops on open positions
            await self._check_trailing_stops(snapshot.prices)

            # 4. Generate signals from all strategies
            all_signals: list[TradingSignal] = []
            for strategy in self._strategies:
                try:
                    signals = await strategy.generate_signals(snapshot)
                    all_signals.extend(signals)
                except Exception:
                    logger.exception("engine.strategy_error", strategy=strategy.name)

            if not all_signals:
                self._cycle_count += 1
                self._last_cycle_at = datetime.now(timezone.utc)
                return

            # 5. Get current portfolio for risk checks
            portfolio = await self._portfolio_tracker.get_current()

            # 6. Evaluate each signal through risk management
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

                # Handle SELL signals via position close
                if signal.action == SignalAction.SELL:
                    await self._handle_sell_signal(signal, price, db_signal.id)
                    continue

                # For BUY signals, run risk evaluation
                decision = await self._risk_manager.evaluate_signal(signal, portfolio, price)

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

            # 7. Take portfolio snapshot
            await self._portfolio_tracker.take_snapshot()
            await self._db.commit()

            self._cycle_count += 1
            self._last_cycle_at = datetime.now(timezone.utc)

        except Exception:
            logger.exception("engine.cycle_error")
            await event_bus.publish(SYSTEM_ERROR, {
                "component": "trading_engine",
                "error": "Cycle failed",
            })

    async def _ensure_connected(self) -> bool:
        """Check broker connection and attempt reconnect if needed."""
        if await self._broker.is_connected():
            self._reconnect_attempts = 0
            return True

        self._reconnect_attempts += 1

        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self._state = EngineState.ERROR
            await send_alert(
                "Broker Connection Lost",
                f"Failed to reconnect after {self._max_reconnect_attempts} attempts. "
                "Trading engine stopped.",
                critical=True,
            )
            return False
        logger.warning(
            "engine.reconnecting",
            attempt=self._reconnect_attempts,
            max=self._max_reconnect_attempts,
        )

        try:
            await self._broker.connect()
            self._reconnect_attempts = 0
            logger.info("engine.reconnected")
            return True
        except Exception:
            logger.exception("engine.reconnect_failed", attempt=self._reconnect_attempts)
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

        # Place sell order for the full position
        result = await self._order_manager.submit_order(
            trade_id=open_trade.id,
            symbol=signal.symbol,
            side="SELL",
            quantity=open_trade.quantity,
            order_type=OrderType.MARKET,
        )

        mapped_status = self._order_manager._map_status(result.status)
        if mapped_status == OrderStatus.FILLED and result.filled_price:
            await self._portfolio_tracker.record_trade_close(
                open_trade, result.filled_price
            )
            self._open_trades.pop(signal.symbol, None)
            logger.info(
                "engine.position_closed",
                symbol=signal.symbol,
                exit_price=result.filled_price,
                pnl=open_trade.realized_pnl,
            )

    async def _execute_buy(
        self, signal: TradingSignal, quantity: int, price: float, signal_id: int
    ) -> None:
        """Execute a BUY signal by placing orders."""
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

        # Place the order
        result = await self._order_manager.submit_order(
            trade_id=trade.id,
            symbol=signal.symbol,
            side="BUY",
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        mapped_status = self._order_manager._map_status(result.status)
        if mapped_status == OrderStatus.FILLED:
            trade.status = TradeStatus.OPEN
            trade.entry_price = result.filled_price
            self._open_trades[signal.symbol] = trade
            logger.info(
                "engine.trade_opened",
                symbol=signal.symbol,
                quantity=quantity,
                price=result.filled_price,
            )

            # Place stop-loss
            if result.filled_price:
                stop_price = round(result.filled_price * 0.97, 2)  # 3% stop-loss
                trade.stop_loss = stop_price
                await self._order_manager.submit_order(
                    trade_id=trade.id,
                    symbol=signal.symbol,
                    side="SELL",
                    quantity=quantity,
                    order_type=OrderType.STOP,
                    stop_price=stop_price,
                )
        elif mapped_status in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED):
            # Order accepted but not yet filled — track for fill polling
            trade.status = TradeStatus.PENDING
            self._open_trades[signal.symbol] = trade
            logger.info(
                "engine.trade_pending_fill",
                symbol=signal.symbol,
                quantity=quantity,
                broker_status=result.status,
            )
        else:
            trade.status = TradeStatus.CANCELLED

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
                # Place stop-loss for the newly filled buy
                if filled_price:
                    stop_price = round(filled_price * 0.97, 2)
                    trade.stop_loss = stop_price
                    await self._order_manager.submit_order(
                        trade_id=trade.id,
                        symbol=symbol,
                        side="SELL",
                        quantity=trade.quantity,
                        order_type=OrderType.STOP,
                        stop_price=stop_price,
                    )
        elif side == "SELL":
            if filled_price:
                await self._portfolio_tracker.record_trade_close(trade, filled_price)
            self._open_trades.pop(symbol, None)
            logger.info(
                "engine.position_closed_on_fill",
                symbol=symbol,
                exit_price=filled_price,
            )

        await self._db.flush()

    async def _check_trailing_stops(self, prices: dict[str, float]) -> None:
        """Update trailing stops for open positions based on current prices."""
        for symbol, trade in list(self._open_trades.items()):
            if trade.entry_price is None or trade.stop_loss is None:
                continue

            current_price = prices.get(symbol)
            if current_price is None:
                continue

            new_stop = calculate_trailing_stop(
                entry_price=trade.entry_price,
                current_price=current_price,
                atr=None,
                trail_pct=0.03,
            )

            if new_stop > trade.stop_loss:
                trade.stop_loss = new_stop
                logger.debug(
                    "engine.trailing_stop_updated",
                    symbol=symbol,
                    new_stop=new_stop,
                    current_price=current_price,
                )

    async def _load_open_trades(self) -> None:
        """Load open trades from database on startup for recovery."""
        try:
            result = await self._db.execute(
                select(Trade).where(Trade.status.in_([TradeStatus.OPEN, TradeStatus.PENDING]))
            )
            trades = result.scalars().all()
            for trade in trades:
                self._open_trades[trade.symbol] = trade
            if trades:
                logger.info("engine.loaded_open_trades", count=len(trades))
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
        logger.info("engine.daily_reset")

    def get_status(self) -> dict:
        """Return current engine status for the API."""
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
        }
