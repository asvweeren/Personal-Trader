"""
Hard safety limits that cannot be overridden by AI or configuration.
These are the absolute last line of defense.
"""

from dataclasses import dataclass, field
from datetime import datetime

from app.broker.base import Portfolio
from app.core.exceptions import (
    DailyLossLimitExceeded,
    InsufficientCashReserve,
    MarketClosedError,
    MaxPositionsExceeded,
    PositionSizeLimitExceeded,
)
from app.risk.market_hours import Exchange, get_exchange_for_symbol, is_market_open


@dataclass
class HardLimitCheck:
    passed: bool
    violations: list[str] = field(default_factory=list)


def check_daily_loss(
    portfolio: Portfolio, daily_start_value: float, max_loss_pct: float
) -> None:
    """Check if daily loss limit has been exceeded. Raises if violated."""
    if daily_start_value <= 0:
        return
    current_value = portfolio.account_summary.total_value
    loss_pct = ((daily_start_value - current_value) / daily_start_value) * 100
    if loss_pct >= max_loss_pct:
        raise DailyLossLimitExceeded(
            f"Daily loss of {loss_pct:.2f}% exceeds limit of {max_loss_pct}%"
        )


def check_position_size(
    portfolio: Portfolio, order_value: float, max_position_pct: float
) -> None:
    """Check if a new order would exceed max position size. Raises if violated."""
    total_value = portfolio.account_summary.total_value
    if total_value <= 0:
        raise PositionSizeLimitExceeded("Portfolio value is zero")
    position_pct = (order_value / total_value) * 100
    if position_pct > max_position_pct:
        raise PositionSizeLimitExceeded(
            f"Position size {position_pct:.2f}% exceeds limit of {max_position_pct}%"
        )


def check_max_positions(
    portfolio: Portfolio, max_open_positions: int
) -> None:
    """Check if maximum number of open positions would be exceeded. Raises if violated."""
    current_positions = len(portfolio.positions)
    if current_positions >= max_open_positions:
        raise MaxPositionsExceeded(
            f"Already at {current_positions} positions (max: {max_open_positions})"
        )


def check_cash_reserve(
    portfolio: Portfolio, order_value: float, min_cash_reserve_pct: float
) -> None:
    """Check if cash reserve would drop below minimum. Raises if violated."""
    total_value = portfolio.account_summary.total_value
    cash_after = portfolio.account_summary.cash - order_value
    if total_value <= 0:
        raise InsufficientCashReserve("Portfolio value is zero")
    reserve_pct = (cash_after / total_value) * 100
    if reserve_pct < min_cash_reserve_pct:
        raise InsufficientCashReserve(
            f"Cash reserve would be {reserve_pct:.2f}% (min: {min_cash_reserve_pct}%)"
        )


def check_market_hours(symbol: str, now: datetime | None = None) -> None:
    """Check if the relevant market is open for this symbol. Raises if closed."""
    exchange = get_exchange_for_symbol(symbol)
    if not is_market_open(exchange, now):
        raise MarketClosedError(
            f"Market {exchange.value} is closed for {symbol}"
        )


def check_all_hard_limits(
    portfolio: Portfolio,
    daily_start_value: float,
    order_value: float,
    is_new_position: bool,
    max_daily_loss_pct: float,
    max_position_pct: float,
    max_open_positions: int,
    min_cash_reserve_pct: float,
    symbol: str | None = None,
    check_hours: bool = True,
) -> HardLimitCheck:
    """Run all hard limit checks. Returns result instead of raising."""
    violations = []

    # Market hours check
    if check_hours and symbol:
        try:
            check_market_hours(symbol)
        except MarketClosedError as e:
            violations.append(str(e))

    try:
        check_daily_loss(portfolio, daily_start_value, max_daily_loss_pct)
    except DailyLossLimitExceeded as e:
        violations.append(str(e))

    try:
        check_position_size(portfolio, order_value, max_position_pct)
    except PositionSizeLimitExceeded as e:
        violations.append(str(e))

    if is_new_position:
        try:
            check_max_positions(portfolio, max_open_positions)
        except MaxPositionsExceeded as e:
            violations.append(str(e))

    try:
        check_cash_reserve(portfolio, order_value, min_cash_reserve_pct)
    except InsufficientCashReserve as e:
        violations.append(str(e))

    return HardLimitCheck(passed=len(violations) == 0, violations=violations)
