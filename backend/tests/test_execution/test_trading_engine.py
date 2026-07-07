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
from app.models.trade import Trade, TradeSide
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
