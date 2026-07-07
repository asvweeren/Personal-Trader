import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.broker.base import (
    AccountSummary,
    BrokerAdapter,
    OrderResult,
    Portfolio,
)
from app.data.market_data import MarketSnapshot
from app.execution.engine import EngineState, TradingEngine
from app.models.trade import Trade, TradeSide, TradeStatus
from app.monitoring.performance import PerformanceTracker
from app.risk.manager import RiskDecision, RiskManager
from app.strategy.base import SignalAction, Strategy, TradingSignal

# ── Helpers ──────────────────────────────────────────────────


def make_portfolio(total=5000.0, cash=3000.0, positions=None):
    return Portfolio(
        account_summary=AccountSummary(
            total_value=total, cash=cash, buying_power=cash,
            unrealized_pnl=0.0, realized_pnl=0.0,
        ),
        positions=positions or [],
    )


def make_snapshot(prices=None):
    return MarketSnapshot(
        timestamp=datetime.now(UTC),
        prices=prices or {"AAPL": 150.0},
        ohlcv={},
        features={},
    )


class MockStrategy(Strategy):
    def __init__(self, signals=None):
        self._signals = signals or []

    @property
    def name(self) -> str:
        return "mock_strategy"

    async def generate_signals(self, market_data):
        return self._signals


def make_engine(
    broker=None, strategies=None, risk_manager=None, market_data=None,
    trading_enabled=False,
):
    broker = broker or MagicMock(spec=BrokerAdapter)
    broker.connect = AsyncMock()
    broker.disconnect = AsyncMock()
    broker.is_connected = AsyncMock(return_value=True)
    broker.get_portfolio = AsyncMock(return_value=make_portfolio())
    broker.place_order = AsyncMock(return_value=OrderResult(
        order_id="ord-1", status="FILLED", filled_price=150.0, filled_quantity=10,
    ))
    broker.cancel_order = AsyncMock(return_value=True)
    broker.get_order_status = AsyncMock()

    risk_manager = risk_manager or MagicMock(spec=RiskManager)
    risk_manager.set_daily_start_value = MagicMock()
    risk_manager.daily_loss_triggered = False
    risk_manager.evaluate_signal = AsyncMock(
        return_value=RiskDecision(approved=True, signal=None, adjusted_quantity=10)
    )

    market_data = market_data or MagicMock()
    market_data.get_snapshot = AsyncMock(return_value=make_snapshot())

    db = MagicMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()
    # Mock the result of loading open trades
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    db.execute.return_value = mock_result

    performance = PerformanceTracker(initial_capital=5000.0)

    return TradingEngine(
        broker=broker,
        strategies=strategies or [],
        risk_manager=risk_manager,
        market_data=market_data,
        performance=performance,
        db=db,
        symbols=["AAPL"],
        trading_enabled=trading_enabled,
    )


# ── Start / Stop tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_engine_start():
    engine = make_engine()
    await engine.start()
    assert engine.state == EngineState.RUNNING


@pytest.mark.asyncio
async def test_engine_stop():
    engine = make_engine()
    await engine.start()
    await engine.stop()
    assert engine.state == EngineState.STOPPED


@pytest.mark.asyncio
async def test_engine_start_connects_broker():
    engine = make_engine()
    engine._broker.is_connected = AsyncMock(return_value=False)
    await engine.start()
    engine._broker.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_engine_start_error():
    engine = make_engine()
    engine._broker.is_connected = AsyncMock(side_effect=Exception("Connection failed"))

    with pytest.raises(Exception):
        await engine.start()
    assert engine.state == EngineState.ERROR


@pytest.mark.asyncio
async def test_engine_start_idempotent():
    """A second start() while already running must not re-run reconciliation.

    Both the app startup path and the broker watchdog can reach start() during
    the initialization race window; re-running stale-order reconciliation and
    open-trade loading against the same account is a real order-integrity risk.
    """
    engine = make_engine()
    await engine.start()
    assert engine.state == EngineState.RUNNING

    engine._reconcile_stale_orders = AsyncMock()
    engine._load_open_trades = AsyncMock()
    await engine.start()

    engine._reconcile_stale_orders.assert_not_awaited()
    engine._load_open_trades.assert_not_awaited()
    assert engine.state == EngineState.RUNNING


# ── Cycle tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cycle_not_running():
    engine = make_engine()
    # Engine is STOPPED, cycle should be a no-op
    await engine.run_cycle()
    engine._market_data.get_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_cycle_no_signals():
    engine = make_engine(strategies=[MockStrategy(signals=[])])
    await engine.start()
    await engine.run_cycle()
    assert engine._cycle_count == 1


@pytest.mark.asyncio
async def test_cycle_hold_signal_ignored():
    hold_signal = TradingSignal(
        symbol="AAPL", action=SignalAction.HOLD,
        confidence=0.5, strategy_name="mock",
    )
    engine = make_engine(
        strategies=[MockStrategy(signals=[hold_signal])],
    )
    await engine.start()
    await engine.run_cycle()
    assert engine._cycle_count == 1


@pytest.mark.asyncio
async def test_cycle_buy_signal_trading_disabled():
    buy_signal = TradingSignal(
        symbol="AAPL", action=SignalAction.BUY,
        confidence=0.8, strategy_name="mock",
    )
    engine = make_engine(
        strategies=[MockStrategy(signals=[buy_signal])],
        trading_enabled=False,
    )
    await engine.start()
    await engine.run_cycle()
    # Signal should be logged but not executed
    assert len(engine._open_trades) == 0


@pytest.mark.asyncio
@patch("app.risk.market_hours.is_market_open", return_value=True)
async def test_cycle_buy_signal_trading_enabled(_mock_open):
    from app.config import settings
    # Ensure R:R ratio passes the 2.5 minimum gate (risk=3% → need TP >= 7.5%)
    old_tp = settings.min_take_profit_pct
    old_opening = settings.opening_range_minutes
    settings.min_take_profit_pct = 8.0
    settings.opening_range_minutes = 0  # Disable opening range gate for test
    try:
        buy_signal = TradingSignal(
            symbol="AAPL", action=SignalAction.BUY,
            confidence=0.8, strategy_name="mock",
        )
        engine = make_engine(
            strategies=[MockStrategy(signals=[buy_signal])],
            trading_enabled=True,
        )
        await engine.start()
        await engine.run_cycle()
        assert len(engine._open_trades) == 1
        assert "AAPL" in engine._open_trades
    finally:
        settings.min_take_profit_pct = old_tp
        settings.opening_range_minutes = old_opening


@pytest.mark.asyncio
async def test_cycle_sell_signal_closes_position():
    sell_signal = TradingSignal(
        symbol="AAPL", action=SignalAction.SELL,
        confidence=0.8, strategy_name="mock",
    )
    engine = make_engine(
        strategies=[MockStrategy(signals=[sell_signal])],
        trading_enabled=True,
    )
    await engine.start()

    # Manually add an open trade
    mock_trade = MagicMock(spec=Trade)
    mock_trade.id = 1
    mock_trade.symbol = "AAPL"
    mock_trade.side = TradeSide.BUY
    mock_trade.quantity = 10
    mock_trade.entry_price = 145.0
    mock_trade.stop_loss = 140.0
    mock_trade.strategy_name = "mock"
    mock_trade.created_at = datetime.now(UTC) - timedelta(minutes=31)
    engine._open_trades["AAPL"] = mock_trade

    await engine.run_cycle()
    assert "AAPL" not in engine._open_trades


@pytest.mark.asyncio
async def test_cycle_sell_no_position():
    sell_signal = TradingSignal(
        symbol="AAPL", action=SignalAction.SELL,
        confidence=0.8, strategy_name="mock",
    )
    engine = make_engine(
        strategies=[MockStrategy(signals=[sell_signal])],
        trading_enabled=True,
    )
    await engine.start()
    # No open position, should just log and continue
    await engine.run_cycle()
    assert engine._cycle_count == 1


@pytest.mark.asyncio
async def test_cycle_signal_rejected():
    buy_signal = TradingSignal(
        symbol="AAPL", action=SignalAction.BUY,
        confidence=0.8, strategy_name="mock",
    )
    engine = make_engine(
        strategies=[MockStrategy(signals=[buy_signal])],
        trading_enabled=True,
    )
    engine._risk_manager.evaluate_signal = AsyncMock(
        return_value=RiskDecision(approved=False, signal=buy_signal, reason="Too risky")
    )
    await engine.start()
    await engine.run_cycle()
    assert len(engine._open_trades) == 0


# ── Reconnection tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_reconnect_on_disconnect():
    engine = make_engine()
    await engine.start()

    # Simulate disconnect then reconnect
    engine._broker.is_connected = AsyncMock(side_effect=[False, True])
    engine._broker.connect = AsyncMock()

    connected = await engine._ensure_connected()
    assert connected is True
    assert engine._reconnect_attempts == 0


@pytest.mark.asyncio
async def test_reconnect_never_gives_up():
    """Engine keeps trying to reconnect — it never enters ERROR state on its own."""
    engine = make_engine()
    await engine.start()

    engine._broker.is_connected = AsyncMock(return_value=False)
    engine._broker.connect = AsyncMock(side_effect=Exception("Connection failed"))

    # Multiple failed attempts should NOT put engine in ERROR
    for _ in range(10):
        await engine._ensure_connected()

    assert engine._reconnect_attempts == 10
    assert engine.state == EngineState.RUNNING  # Still running, waiting for reconnect

    # When connection is restored, attempts reset
    engine._broker.is_connected = AsyncMock(return_value=True)
    result = await engine._ensure_connected()
    assert result is True
    assert engine._reconnect_attempts == 0


# ── Trailing stop tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_trailing_stop_update():
    engine = make_engine()
    await engine.start()

    mock_trade = MagicMock(spec=Trade)
    mock_trade.symbol = "AAPL"
    mock_trade.entry_price = 100.0
    mock_trade.stop_loss = 97.0
    mock_trade.quantity = 10
    engine._open_trades["AAPL"] = mock_trade

    # Price went up, trailing stop should rise
    await engine._check_trailing_stops({"AAPL": 110.0})
    assert mock_trade.stop_loss > 97.0


@pytest.mark.asyncio
async def test_trailing_stop_no_decrease():
    engine = make_engine()
    await engine.start()

    mock_trade = MagicMock(spec=Trade)
    mock_trade.symbol = "AAPL"
    mock_trade.entry_price = 100.0
    mock_trade.stop_loss = 97.0
    mock_trade.quantity = 10
    engine._open_trades["AAPL"] = mock_trade

    # Price went down, stop should not decrease
    await engine._check_trailing_stops({"AAPL": 95.0})
    assert mock_trade.stop_loss == 97.0


# ── Continuous daily-loss halt ──────────────────────────────


@pytest.mark.asyncio
async def test_daily_loss_halt_triggers_and_blocks():
    """The 1-min safety check halts trading when the portfolio breaches the
    daily-loss limit even if no new signal arrives (force-close off)."""
    from app.config import settings

    rm = RiskManager(max_daily_loss_pct=5.0)
    rm.set_daily_start_value(5000.0)

    engine = make_engine(risk_manager=rm)
    # Broker now reports a 6% loss (4700 < 5000 * 0.95). Set after make_engine,
    # which otherwise resets get_portfolio to the 5000 default.
    engine._broker.get_portfolio = AsyncMock(return_value=make_portfolio(total=4700.0))

    with patch.object(settings, "daily_loss_force_close", False):
        triggered = await engine.check_daily_loss_halt()

    assert triggered is True
    assert rm.daily_loss_triggered is True


@pytest.mark.asyncio
async def test_daily_loss_halt_not_triggered_when_within_limit():
    rm = RiskManager(max_daily_loss_pct=5.0)
    rm.set_daily_start_value(5000.0)

    engine = make_engine(risk_manager=rm)
    engine._broker.get_portfolio = AsyncMock(return_value=make_portfolio(total=4900.0))  # -2%

    triggered = await engine.check_daily_loss_halt()

    assert triggered is False
    assert rm.daily_loss_triggered is False


# ── Trading toggle tests ────────────────────────────────────


def test_trading_toggle():
    engine = make_engine()
    assert engine.trading_enabled is False
    engine.trading_enabled = True
    assert engine.trading_enabled is True


# ── Status tests ─────────────────────────────────────────────


def test_get_status():
    engine = make_engine()
    status = engine.get_status()
    assert status["state"] == "STOPPED"
    assert status["trading_enabled"] is False
    assert status["symbols"] == ["AAPL"]
    assert status["cycle_count"] == 0


@pytest.mark.asyncio
async def test_get_status_running():
    engine = make_engine()
    await engine.start()
    status = engine.get_status()
    assert status["state"] == "RUNNING"


# ── Strategy error handling test ─────────────────────────────


@pytest.mark.asyncio
async def test_cycle_strategy_error_continues():
    class FailStrategy(Strategy):
        @property
        def name(self):
            return "fail"

        async def generate_signals(self, market_data):
            raise RuntimeError("Strategy crashed")

    good_signal = TradingSignal(
        symbol="AAPL", action=SignalAction.BUY,
        confidence=0.8, strategy_name="good",
    )

    engine = make_engine(
        strategies=[FailStrategy(), MockStrategy(signals=[good_signal])],
        trading_enabled=True,
    )
    await engine.start()
    # Should not raise - error from FailStrategy is caught
    await engine.run_cycle()
    assert engine._cycle_count == 1


# ── Naked-position guard (stop-loss placement failure) ───────


def _open_trade(symbol="AAPL", side=TradeSide.BUY, qty=10, entry=150.0, stop=145.0):
    t = MagicMock(spec=Trade)
    t.id = 1
    t.symbol = symbol
    t.side = side
    t.quantity = qty
    t.status = TradeStatus.OPEN
    t.entry_price = entry
    t.stop_loss = stop
    t.strategy_name = "mock"
    t.realized_pnl = 0.0
    t.commission = 0.0
    t.created_at = datetime.now(UTC) - timedelta(minutes=5)
    t.closed_at = None
    return t


@pytest.mark.asyncio
async def test_pending_stop_retry_flattens_after_max_attempts():
    """A position whose stop keeps failing must be flattened, never held naked forever."""
    engine = make_engine(trading_enabled=True)
    await engine.start()

    trade = _open_trade()
    engine._open_trades["AAPL"] = trade
    engine._pending_stop_retries.add("AAPL")

    engine._place_stop_verified = AsyncMock(return_value=False)
    engine._flatten_position = AsyncMock()
    engine._broker.cancel_open_orders_for_symbol = AsyncMock()

    # Attempts below the threshold do not flatten yet.
    for _ in range(engine._MAX_STOP_RETRY_ATTEMPTS - 1):
        await engine._retry_pending_stops()
    engine._flatten_position.assert_not_called()
    assert "AAPL" in engine._pending_stop_retries

    # The attempt that reaches the threshold flattens and clears the queue.
    await engine._retry_pending_stops()
    engine._flatten_position.assert_called_once()
    assert "AAPL" not in engine._pending_stop_retries
    assert "AAPL" not in engine._stop_retry_attempts


@pytest.mark.asyncio
async def test_pending_stop_retry_success_clears_queue():
    """A successful stop placement clears the retry queue and never flattens."""
    engine = make_engine(trading_enabled=True)
    await engine.start()

    trade = _open_trade()
    engine._open_trades["AAPL"] = trade
    engine._pending_stop_retries.add("AAPL")

    engine._place_stop_verified = AsyncMock(return_value=True)
    engine._flatten_position = AsyncMock()
    engine._broker.cancel_open_orders_for_symbol = AsyncMock()

    await engine._retry_pending_stops()

    engine._flatten_position.assert_not_called()
    assert "AAPL" not in engine._pending_stop_retries


@pytest.mark.asyncio
async def test_flatten_position_closes_long():
    """_flatten_position market-closes the position and removes it from tracking."""
    engine = make_engine(trading_enabled=True)
    await engine.start()

    trade = _open_trade()
    engine._open_trades["AAPL"] = trade
    engine._broker.cancel_open_orders_for_symbol = AsyncMock()

    await engine._flatten_position("AAPL", trade, "no_stop_protection")

    assert "AAPL" not in engine._open_trades


@pytest.mark.asyncio
async def test_buy_accepts_fill_after_cancel_race_no_duplicate():
    """If an order fills during the cancel/retry race, accept it — never duplicate."""
    from app.config import settings

    saved = {
        "t": settings.order_fill_timeout_seconds,
        "r": settings.order_max_retries,
        "tp": settings.min_take_profit_pct,
        "o": settings.opening_range_minutes,
    }
    settings.order_fill_timeout_seconds = 1
    settings.order_max_retries = 2
    settings.min_take_profit_pct = 8.0
    settings.opening_range_minutes = 0
    try:
        engine = make_engine(trading_enabled=True)
        await engine.start()

        # place_order returns SUBMITTED (IBKR does not fill synchronously).
        engine._broker.place_order = AsyncMock(
            return_value=OrderResult(order_id="o1", status="SUBMITTED")
        )
        # The poll window sees SUBMITTED; after the cancel the order is FILLED.
        engine._broker.get_order_status = AsyncMock(side_effect=[
            OrderResult(order_id="o1", status="SUBMITTED"),
            OrderResult(order_id="o1", status="FILLED",
                        filled_price=150.0, filled_quantity=10),
        ])
        engine._broker.cancel_order = AsyncMock(return_value=True)
        engine._place_stop_verified = AsyncMock(return_value=True)

        sig = TradingSignal(
            symbol="AAPL", action=SignalAction.BUY,
            confidence=0.9, strategy_name="mock",
        )
        await engine._execute_buy(sig, quantity=10, price=150.0, signal_id=1)

        # One position, and the order was placed exactly once (no duplicate retry).
        assert "AAPL" in engine._open_trades
        assert engine._broker.place_order.await_count == 1
    finally:
        settings.order_fill_timeout_seconds = saved["t"]
        settings.order_max_retries = saved["r"]
        settings.min_take_profit_pct = saved["tp"]
        settings.opening_range_minutes = saved["o"]


# ── Cycle/safety-loop session serialization (#5) ─────────────


@pytest.mark.asyncio
async def test_run_cycle_holds_and_releases_cycle_lock():
    """The cycle lock is held during the inner cycle and released afterwards."""
    engine = make_engine()
    await engine.start()

    held = {}

    async def fake_inner():
        held["locked"] = engine._cycle_lock.locked()

    engine._run_cycle_inner = fake_inner
    await engine.run_cycle()

    assert held["locked"] is True
    assert not engine._cycle_lock.locked()


# ── Reconciliation cadence (#3) ──────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_if_due_throttles():
    """Reconciliation runs at most once per interval, then again once it elapses."""
    engine = make_engine()
    await engine.start()
    engine._run_reconciliation = AsyncMock()

    await engine.reconcile_if_due(interval_minutes=5)
    assert engine._run_reconciliation.await_count == 1

    await engine.reconcile_if_due(interval_minutes=5)
    assert engine._run_reconciliation.await_count == 1  # throttled

    engine._last_reconcile_at = datetime.now(UTC) - timedelta(minutes=6)
    await engine.reconcile_if_due(interval_minutes=5)
    assert engine._run_reconciliation.await_count == 2  # interval elapsed


# ── EOD close reliability (#4) ───────────────────────────────


@pytest.mark.asyncio
@patch("app.execution.engine.minutes_until_close_for_symbol", return_value=1.0)
async def test_eod_close_failure_releases_guard(_mins):
    """An EOD close that does not fill must release its guard and keep retrying."""
    from app.config import settings

    old = settings.eod_close_minutes_before
    settings.eod_close_minutes_before = 30
    try:
        engine = make_engine(trading_enabled=True)
        await engine.start()

        trade = _open_trade()
        engine._open_trades["AAPL"] = trade
        # The close order never fills.
        engine._await_market_fill = AsyncMock(
            return_value=OrderResult(order_id="x", status="SUBMITTED")
        )

        await engine._check_eod_close({"AAPL": 150.0})

        # Guard released (so next tick retries) and position NOT dropped.
        assert "AAPL" not in engine._eod_sell_pending
        assert "AAPL" in engine._open_trades
    finally:
        settings.eod_close_minutes_before = old


# ── Engine init concurrency ──────────────────────────────────


@pytest.mark.asyncio
async def test_init_trading_engine_single_build_under_concurrency(monkeypatch):
    """Concurrent init callers must share one engine, never build two.

    App startup and the broker watchdog can both call init_trading_engine()
    around the same time. Without serialization each would construct its own
    TradingEngine and both would start()/reconcile against the same account.
    """
    import app.dependencies as deps

    monkeypatch.setattr(deps, "_trading_engine", None)
    build_count = 0
    sentinel = MagicMock(name="engine")

    async def fake_build(db):
        nonlocal build_count
        build_count += 1
        await asyncio.sleep(0.01)  # widen the race window between callers
        deps._trading_engine = sentinel
        return sentinel

    monkeypatch.setattr(deps, "_build_trading_engine", fake_build)

    results = await asyncio.gather(
        *[deps.init_trading_engine(MagicMock()) for _ in range(5)]
    )

    assert build_count == 1
    assert all(r is sentinel for r in results)
