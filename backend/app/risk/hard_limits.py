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
    MaxLeverageExceeded,
    MaxPositionsExceeded,
    PositionSizeLimitExceeded,
    SectorConcentrationExceeded,
    ShortPositionError,
)
from app.risk.market_hours import get_exchange_for_symbol, is_market_open
from app.risk.position_sizer import get_sector

# Maximum ratio of total positions value to equity (net liquidation).
# 2.0 = max 2x leverage (e.g. $200k positions on $100k equity).
MAX_LEVERAGE_RATIO = 2.0


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
    """Check if cash reserve would drop below minimum. Raises if violated.

    On margin accounts, uses buying_power instead of cash (which can be
    negative due to margin loans).
    """
    total_value = portfolio.account_summary.total_value
    if total_value <= 0:
        raise InsufficientCashReserve("Portfolio value is zero")
    cash = portfolio.account_summary.cash
    # On margin accounts cash is negative; use buying_power instead
    if cash < 0:
        available = portfolio.account_summary.buying_power
        if available < order_value:
            raise InsufficientCashReserve(
                f"Insufficient buying power (${available:,.0f}) for ${order_value:,.0f} order"
            )
        return
    cash_after = cash - order_value
    reserve_pct = (cash_after / total_value) * 100
    if reserve_pct < min_cash_reserve_pct:
        raise InsufficientCashReserve(
            f"Cash reserve would be {reserve_pct:.2f}% (min: {min_cash_reserve_pct}%)"
        )


def check_leverage(portfolio: Portfolio, order_value: float) -> None:
    """Check if total leverage would exceed maximum. Raises if violated.

    Prevents the account from becoming overleveraged on margin accounts.
    Total positions value (including the new order) must not exceed
    MAX_LEVERAGE_RATIO × equity (net liquidation value).
    """
    equity = portfolio.account_summary.total_value
    if equity <= 0:
        raise MaxLeverageExceeded("Portfolio equity is zero")
    total_positions = sum(abs(p.market_value) for p in portfolio.positions)
    new_total = total_positions + order_value
    leverage = new_total / equity
    if leverage > MAX_LEVERAGE_RATIO:
        raise MaxLeverageExceeded(
            f"Leverage would be {leverage:.1f}x (max: {MAX_LEVERAGE_RATIO}x). "
            f"Positions: ${new_total:,.0f} on ${equity:,.0f} equity"
        )


def check_no_short_position(
    portfolio: Portfolio, symbol: str, sell_qty: int
) -> None:
    """Check if a SELL order would create a short position. Raises if so.

    Looks up the current position for *symbol* in the broker portfolio.
    If the sell quantity exceeds the held quantity, the order is rejected.
    """
    held_qty = 0
    for pos in portfolio.positions:
        if pos.symbol == symbol:
            held_qty = max(pos.quantity, 0)  # Treat existing shorts as 0
            break
    if sell_qty > held_qty:
        raise ShortPositionError(
            f"SELL {sell_qty} {symbol} would exceed position of {held_qty} "
            f"and create a short position"
        )


def check_sector_concentration(
    portfolio: Portfolio, symbol: str, order_value: float, max_sector_pct: float
) -> None:
    """Check if a new order would push sector concentration above the limit."""
    total_value = portfolio.account_summary.total_value
    if total_value <= 0:
        return

    target_sector = get_sector(symbol)
    if target_sector in ("unknown", "etf_broad"):
        return  # Don't limit unknown sectors or broad ETFs

    sector_value = sum(
        abs(p.market_value) for p in portfolio.positions
        if get_sector(p.symbol) == target_sector
    )
    new_sector_value = sector_value + order_value
    sector_pct = (new_sector_value / total_value) * 100

    if sector_pct > max_sector_pct:
        raise SectorConcentrationExceeded(
            f"Sector '{target_sector}' would be {sector_pct:.1f}% of portfolio "
            f"(max: {max_sector_pct}%)"
        )


def check_market_hours(symbol: str, now: datetime | None = None) -> None:
    """Check if the relevant market is open for this symbol. Raises if closed."""
    from app.config import settings
    exchange = get_exchange_for_symbol(symbol)
    if not is_market_open(exchange, now, include_extended=settings.extended_hours_enabled):
        raise MarketClosedError(
            f"Market {exchange.value} is closed for {symbol}"
        )


def check_economic_events(symbol: str) -> None:
    """Check if there's a high-impact economic event within 2 hours.

    Blocks trading to avoid volatility around major events.
    """
    try:
        from app.data.economic_calendar import get_economic_calendar
        calendar = get_economic_calendar()
        if calendar.has_high_impact_event(symbol, within_hours=2):
            raise MarketClosedError(
                f"High-impact economic event within 2 hours for {symbol}"
            )
    except MarketClosedError:
        raise
    except Exception:
        pass  # Calendar not available, don't block trading


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
    max_sector_pct: float = 35.0,
) -> HardLimitCheck:
    """Run all hard limit checks. Returns result instead of raising."""
    violations = []

    # Market hours check
    if check_hours and symbol:
        try:
            check_market_hours(symbol)
        except MarketClosedError as e:
            violations.append(str(e))

    # Economic events check
    if symbol:
        try:
            check_economic_events(symbol)
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

    try:
        check_leverage(portfolio, order_value)
    except MaxLeverageExceeded as e:
        violations.append(str(e))

    # Sector concentration check
    if symbol and is_new_position:
        try:
            check_sector_concentration(portfolio, symbol, order_value, max_sector_pct)
        except SectorConcentrationExceeded as e:
            violations.append(str(e))

    return HardLimitCheck(passed=len(violations) == 0, violations=violations)
