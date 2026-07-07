from unittest.mock import patch

import pytest

from app.broker.base import AccountSummary, Portfolio, Position
from app.core.exceptions import (
    DailyLossLimitExceeded,
    InsufficientCashReserve,
    MarketClosedError,
    MaxPositionsExceeded,
    PositionSizeLimitExceeded,
)
from app.risk.hard_limits import (
    check_all_hard_limits,
    check_cash_reserve,
    check_daily_loss,
    check_market_hours,
    check_max_positions,
    check_position_size,
)


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


# ── Daily loss ────────────────────────────────────────────────


def test_daily_loss_ok():
    portfolio = make_portfolio(total_value=4800)
    check_daily_loss(portfolio, daily_start_value=5000, max_loss_pct=5.0)


def test_daily_loss_exceeded():
    portfolio = make_portfolio(total_value=4700)
    with pytest.raises(DailyLossLimitExceeded):
        check_daily_loss(portfolio, daily_start_value=5000, max_loss_pct=5.0)


# ── Position size ─────────────────────────────────────────────


def test_position_size_ok():
    portfolio = make_portfolio(total_value=5000)
    check_position_size(portfolio, order_value=500, max_position_pct=20.0)


def test_position_size_exceeded():
    portfolio = make_portfolio(total_value=5000)
    with pytest.raises(PositionSizeLimitExceeded):
        check_position_size(portfolio, order_value=1500, max_position_pct=20.0)


def test_position_size_absolute_notional_cap():
    """An order under the pct limit is still rejected if it exceeds the absolute cap."""
    from app.config import settings

    # Large portfolio so pct check passes, but a hard notional cap blocks it.
    portfolio = make_portfolio(total_value=250_000, cash=250_000)
    with patch.object(settings, "max_position_notional", 10_000):
        # 8% of portfolio (passes pct) but €20k > €10k cap → rejected
        with pytest.raises(PositionSizeLimitExceeded):
            check_position_size(portfolio, order_value=20_000, max_position_pct=20.0)
        # Under the cap → allowed
        check_position_size(portfolio, order_value=5_000, max_position_pct=20.0)


def test_position_size_notional_cap_disabled_by_default():
    """max_position_notional=0 disables the absolute cap (pct check only)."""
    from app.config import settings

    portfolio = make_portfolio(total_value=250_000, cash=250_000)
    with patch.object(settings, "max_position_notional", 0.0):
        check_position_size(portfolio, order_value=40_000, max_position_pct=20.0)


# ── Max positions ─────────────────────────────────────────────


def test_max_positions_ok():
    positions = [
        Position("AAPL", 10, 150, 150, 1500, 0) for _ in range(5)
    ]
    portfolio = make_portfolio(positions=positions)
    check_max_positions(portfolio, max_open_positions=10)


def test_max_positions_exceeded():
    positions = [
        Position(f"SYM{i}", 10, 150, 150, 1500, 0) for i in range(10)
    ]
    portfolio = make_portfolio(positions=positions)
    with pytest.raises(MaxPositionsExceeded):
        check_max_positions(portfolio, max_open_positions=10)


# ── Cash reserve ──────────────────────────────────────────────


def test_cash_reserve_ok():
    portfolio = make_portfolio(total_value=5000, cash=2000)
    check_cash_reserve(portfolio, order_value=200, min_cash_reserve_pct=30.0)


def test_cash_reserve_insufficient():
    portfolio = make_portfolio(total_value=5000, cash=1600)
    with pytest.raises(InsufficientCashReserve):
        check_cash_reserve(portfolio, order_value=200, min_cash_reserve_pct=30.0)


# ── Market hours ──────────────────────────────────────────────


def test_market_hours_open():
    with patch("app.risk.hard_limits.is_market_open", return_value=True):
        check_market_hours("AAPL")  # Should not raise


def test_market_hours_closed():
    with patch("app.risk.hard_limits.is_market_open", return_value=False):
        with pytest.raises(MarketClosedError):
            check_market_hours("AAPL")


# ── All limits combined ───────────────────────────────────────


def test_all_limits_pass():
    portfolio = make_portfolio(total_value=5000, cash=3000)
    with patch("app.risk.hard_limits.is_market_open", return_value=True):
        result = check_all_hard_limits(
            portfolio=portfolio,
            daily_start_value=5000,
            order_value=500,
            is_new_position=True,
            max_daily_loss_pct=5.0,
            max_position_pct=20.0,
            max_open_positions=10,
            min_cash_reserve_pct=30.0,
            symbol="AAPL",
        )
    assert result.passed
    assert len(result.violations) == 0


def test_all_limits_pass_without_hours_check():
    portfolio = make_portfolio(total_value=5000, cash=3000)
    result = check_all_hard_limits(
        portfolio=portfolio,
        daily_start_value=5000,
        order_value=500,
        is_new_position=True,
        max_daily_loss_pct=5.0,
        max_position_pct=20.0,
        max_open_positions=10,
        min_cash_reserve_pct=30.0,
        check_hours=False,
    )
    assert result.passed


def test_all_limits_market_closed_violation():
    portfolio = make_portfolio(total_value=5000, cash=3000)
    with patch("app.risk.hard_limits.is_market_open", return_value=False):
        result = check_all_hard_limits(
            portfolio=portfolio,
            daily_start_value=5000,
            order_value=500,
            is_new_position=True,
            max_daily_loss_pct=5.0,
            max_position_pct=20.0,
            max_open_positions=10,
            min_cash_reserve_pct=30.0,
            symbol="AAPL",
        )
    assert not result.passed
    assert any("closed" in v.lower() for v in result.violations)


def test_all_limits_multiple_violations():
    positions = [
        Position(f"SYM{i}", 10, 150, 150, 1500, 0) for i in range(10)
    ]
    portfolio = make_portfolio(total_value=4500, cash=500, positions=positions)
    with patch("app.risk.hard_limits.is_market_open", return_value=True):
        result = check_all_hard_limits(
            portfolio=portfolio,
            daily_start_value=5000,
            order_value=1500,
            is_new_position=True,
            max_daily_loss_pct=5.0,
            max_position_pct=20.0,
            max_open_positions=10,
            min_cash_reserve_pct=30.0,
            symbol="AAPL",
        )
    assert not result.passed
    assert len(result.violations) >= 2
