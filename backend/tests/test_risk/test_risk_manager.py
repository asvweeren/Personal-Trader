from unittest.mock import patch

import pytest

from app.broker.base import AccountSummary, Portfolio, Position
from app.risk.manager import RiskManager
from app.strategy.base import SignalAction, TradingSignal


def make_portfolio(total_value=5000, cash=3000, positions=None):
    return Portfolio(
        account_summary=AccountSummary(
            total_value=total_value,
            cash=cash,
            buying_power=cash,
            unrealized_pnl=0,
            realized_pnl=0,
        ),
        positions=positions or [],
    )


def make_signal(symbol="AAPL", action=SignalAction.BUY, confidence=0.8):
    return TradingSignal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        strategy_name="test_strategy",
    )


# ── evaluate_signal tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_hold_signal_not_approved():
    rm = RiskManager()
    rm.set_daily_start_value(5000)
    signal = make_signal(action=SignalAction.HOLD)
    portfolio = make_portfolio()

    decision = await rm.evaluate_signal(signal, portfolio, 150.0)
    assert decision.approved is False
    assert "HOLD" in decision.reason


@pytest.mark.asyncio
async def test_sell_signal_with_position_approved():
    rm = RiskManager()
    rm.set_daily_start_value(5000)
    positions = [Position("AAPL", 10, 150, 150, 1500, 0)]
    portfolio = make_portfolio(positions=positions)
    signal = make_signal(action=SignalAction.SELL)

    with patch("app.risk.manager.is_market_open", return_value=True):
        decision = await rm.evaluate_signal(signal, portfolio, 150.0)
    assert decision.approved is True


@pytest.mark.asyncio
async def test_sell_signal_without_position_rejected():
    rm = RiskManager()
    rm.set_daily_start_value(5000)
    portfolio = make_portfolio()
    signal = make_signal(action=SignalAction.SELL)

    decision = await rm.evaluate_signal(signal, portfolio, 150.0)
    assert decision.approved is False
    assert "No position" in decision.reason


@pytest.mark.asyncio
async def test_buy_signal_approved_during_market_hours():
    rm = RiskManager()
    rm.set_daily_start_value(5000)
    portfolio = make_portfolio(total_value=5000, cash=3000)
    signal = make_signal(confidence=0.8)

    with patch("app.risk.hard_limits.is_market_open", return_value=True):
        decision = await rm.evaluate_signal(signal, portfolio, 100.0)

    assert decision.approved is True
    assert decision.adjusted_quantity is not None
    assert decision.adjusted_quantity > 0


@pytest.mark.asyncio
async def test_buy_signal_rejected_when_market_closed():
    rm = RiskManager()
    rm.set_daily_start_value(5000)
    portfolio = make_portfolio(total_value=5000, cash=3000)
    signal = make_signal(confidence=0.8)

    with patch("app.risk.hard_limits.is_market_open", return_value=False):
        decision = await rm.evaluate_signal(signal, portfolio, 100.0)

    assert decision.approved is False
    assert "closed" in decision.reason.lower()


@pytest.mark.asyncio
async def test_daily_loss_triggered_halts_all_trading():
    rm = RiskManager()
    rm.set_daily_start_value(5000)
    rm.daily_loss_triggered = True

    portfolio = make_portfolio()
    signal = make_signal()

    decision = await rm.evaluate_signal(signal, portfolio, 150.0)
    assert decision.approved is False
    assert "daily loss" in decision.reason.lower()


@pytest.mark.asyncio
async def test_daily_loss_triggers_on_violation():
    rm = RiskManager(max_daily_loss_pct=5.0)
    rm.set_daily_start_value(5000)

    # Portfolio lost > 5%
    portfolio = make_portfolio(total_value=4700, cash=2000)
    signal = make_signal(confidence=0.8)

    with patch("app.risk.hard_limits.is_market_open", return_value=True):
        decision = await rm.evaluate_signal(signal, portfolio, 100.0)

    assert decision.approved is False
    assert rm.daily_loss_triggered is True


# ── drawdown tracking tests ──────────────────────────────────


def test_drawdown_tracking():
    rm = RiskManager()
    rm.set_daily_start_value(5000)

    rm.update_peak_value(5000)
    assert rm._peak_value == 5000

    rm.update_peak_value(5500)
    assert rm._peak_value == 5500

    rm.update_peak_value(5000)
    # Drawdown = (5500 - 5000) / 5500 = ~9.09%
    assert rm._max_drawdown_pct > 9.0
    assert rm._max_drawdown_pct < 10.0


def test_drawdown_peak_never_decreases():
    rm = RiskManager()
    rm.update_peak_value(5000)
    rm.update_peak_value(4000)
    assert rm._peak_value == 5000


# ── check_portfolio_health tests ─────────────────────────────


@pytest.mark.asyncio
async def test_healthy_portfolio():
    rm = RiskManager()
    rm.set_daily_start_value(5000)

    portfolio = make_portfolio(total_value=5000, cash=2000)

    with patch("app.risk.manager.is_market_open", return_value=True):
        health = await rm.check_portfolio_health(portfolio)

    assert health.healthy is True
    assert health.checks["daily_loss_ok"] is True
    assert health.checks["cash_reserve_ok"] is True
    assert health.warnings == []


@pytest.mark.asyncio
async def test_unhealthy_portfolio_low_cash():
    rm = RiskManager(min_cash_reserve_pct=30.0)
    rm.set_daily_start_value(5000)

    portfolio = make_portfolio(total_value=5000, cash=1000)  # 20% cash

    with patch("app.risk.manager.is_market_open", return_value=True):
        health = await rm.check_portfolio_health(portfolio)

    assert health.checks["cash_reserve_ok"] is False
    assert health.healthy is False


@pytest.mark.asyncio
async def test_health_report_sector_exposure():
    rm = RiskManager()
    rm.set_daily_start_value(5000)

    positions = [
        Position("AAPL", 10, 150, 150, 1500, 0),
        Position("MSFT", 5, 380, 380, 1900, 0),
    ]
    portfolio = make_portfolio(total_value=5000, cash=1600, positions=positions)

    with patch("app.risk.manager.is_market_open", return_value=True):
        health = await rm.check_portfolio_health(portfolio)

    assert "technology" in health.sector_exposure
    # Both AAPL and MSFT are technology sector
    tech_pct = health.sector_exposure["technology"]
    assert tech_pct > 60  # (1500 + 1900) / 5000 = 68%


@pytest.mark.asyncio
async def test_health_report_largest_position():
    rm = RiskManager()
    rm.set_daily_start_value(5000)

    positions = [
        Position("AAPL", 10, 150, 150, 1500, 0),
        Position("JPM", 5, 160, 160, 800, 0),
    ]
    portfolio = make_portfolio(total_value=5000, cash=2700, positions=positions)

    with patch("app.risk.manager.is_market_open", return_value=True):
        health = await rm.check_portfolio_health(portfolio)

    assert health.largest_position_pct == 30.0  # 1500/5000


@pytest.mark.asyncio
async def test_health_report_warns_on_high_concentration():
    rm = RiskManager()
    rm.set_daily_start_value(5000)

    positions = [
        Position("AAPL", 10, 150, 150, 1500, 0),
        Position("MSFT", 10, 190, 190, 1900, 0),
    ]
    portfolio = make_portfolio(total_value=5000, cash=1600, positions=positions)

    with patch("app.risk.manager.is_market_open", return_value=True):
        health = await rm.check_portfolio_health(portfolio)

    # technology sector at 68% > 35% threshold
    sector_warnings = [w for w in health.warnings if "concentration" in w.lower()]
    assert len(sector_warnings) > 0


# ── get_limits / set_daily_start_value ────────────────────────


def test_get_limits():
    rm = RiskManager(max_daily_loss_pct=3.0, max_position_pct=15.0)
    limits = rm.get_limits()
    assert limits["max_daily_loss_pct"] == 3.0
    assert limits["max_position_pct"] == 15.0


def test_daily_reset_clears_flag():
    rm = RiskManager()
    rm.daily_loss_triggered = True
    rm.set_daily_start_value(5000)
    assert rm.daily_loss_triggered is False
    assert rm.daily_start_value == 5000
