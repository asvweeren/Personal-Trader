"""Market hours enforcement - prevents trading outside exchange hours."""

from dataclasses import dataclass
from datetime import datetime, time, timezone, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger()


class Exchange(str, Enum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    EURONEXT = "EURONEXT"
    LSE = "LSE"
    XETRA = "XETRA"


@dataclass(frozen=True)
class TradingSession:
    exchange: Exchange
    timezone: str
    open_time: time
    close_time: time
    # Pre/post market (optional, not used for trading, only for data)
    pre_market_open: time | None = None
    post_market_close: time | None = None


# Exchange definitions
EXCHANGE_SESSIONS: dict[Exchange, TradingSession] = {
    Exchange.NYSE: TradingSession(
        exchange=Exchange.NYSE,
        timezone="America/New_York",
        open_time=time(9, 30),
        close_time=time(16, 0),
        pre_market_open=time(4, 0),
        post_market_close=time(20, 0),
    ),
    Exchange.NASDAQ: TradingSession(
        exchange=Exchange.NASDAQ,
        timezone="America/New_York",
        open_time=time(9, 30),
        close_time=time(16, 0),
        pre_market_open=time(4, 0),
        post_market_close=time(20, 0),
    ),
    Exchange.EURONEXT: TradingSession(
        exchange=Exchange.EURONEXT,
        timezone="Europe/Amsterdam",
        open_time=time(9, 0),
        close_time=time(17, 30),
    ),
    Exchange.LSE: TradingSession(
        exchange=Exchange.LSE,
        timezone="Europe/London",
        open_time=time(8, 0),
        close_time=time(16, 30),
    ),
    Exchange.XETRA: TradingSession(
        exchange=Exchange.XETRA,
        timezone="Europe/Berlin",
        open_time=time(9, 0),
        close_time=time(17, 30),
    ),
}

# US market holidays 2026 (NYSE/NASDAQ closed)
US_HOLIDAYS_2026 = {
    datetime(2026, 1, 1).date(),   # New Year's Day
    datetime(2026, 1, 19).date(),  # MLK Day
    datetime(2026, 2, 16).date(),  # Presidents' Day
    datetime(2026, 4, 3).date(),   # Good Friday
    datetime(2026, 5, 25).date(),  # Memorial Day
    datetime(2026, 7, 3).date(),   # Independence Day (observed)
    datetime(2026, 9, 7).date(),   # Labor Day
    datetime(2026, 11, 26).date(), # Thanksgiving
    datetime(2026, 12, 25).date(), # Christmas
}

# EU market holidays (Euronext - major ones)
EU_HOLIDAYS_2026 = {
    datetime(2026, 1, 1).date(),   # New Year's Day
    datetime(2026, 4, 3).date(),   # Good Friday
    datetime(2026, 4, 6).date(),   # Easter Monday
    datetime(2026, 5, 1).date(),   # Labour Day
    datetime(2026, 12, 25).date(), # Christmas
    datetime(2026, 12, 26).date(), # Boxing Day
}

EXCHANGE_HOLIDAYS: dict[Exchange, set] = {
    Exchange.NYSE: US_HOLIDAYS_2026,
    Exchange.NASDAQ: US_HOLIDAYS_2026,
    Exchange.EURONEXT: EU_HOLIDAYS_2026,
    Exchange.LSE: EU_HOLIDAYS_2026,
    Exchange.XETRA: EU_HOLIDAYS_2026,
}


def is_market_open(
    exchange: Exchange,
    now: datetime | None = None,
    buffer_minutes: int = 5,
) -> bool:
    """Check if the market is currently open for trading.

    Args:
        exchange: Which exchange to check.
        now: Current time (defaults to UTC now).
        buffer_minutes: Buffer before close to stop trading (avoid MOC issues).
    """
    session = EXCHANGE_SESSIONS[exchange]
    tz = ZoneInfo(session.timezone)

    if now is None:
        now = datetime.now(timezone.utc)

    local_now = now.astimezone(tz)

    # Check weekend
    if local_now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Check holidays
    holidays = EXCHANGE_HOLIDAYS.get(exchange, set())
    if local_now.date() in holidays:
        return False

    # Check trading hours (with buffer before close)
    local_time = local_now.time()
    close_with_buffer = datetime.combine(
        local_now.date(), session.close_time
    ) - timedelta(minutes=buffer_minutes)
    close_buffer_time = close_with_buffer.time()

    return session.open_time <= local_time < close_buffer_time


def get_exchange_for_symbol(symbol: str) -> Exchange:
    """Determine the primary exchange for a symbol.

    Uses simple heuristics. In production, this would use a symbol database.
    """
    # European symbols typically have a suffix or are well-known
    eu_suffixes = (".AS", ".PA", ".BR", ".L", ".DE")
    for suffix in eu_suffixes:
        if symbol.upper().endswith(suffix):
            if suffix == ".L":
                return Exchange.LSE
            if suffix == ".DE":
                return Exchange.XETRA
            return Exchange.EURONEXT

    # Default to NYSE for US stocks
    return Exchange.NYSE


# ── IBKR contract mapping ──────────────────────────────────────

SUFFIX_TO_IBKR: dict[str, dict[str, str]] = {
    ".AS": {"currency": "EUR", "primary_exchange": "AEB"},      # Euronext Amsterdam
    ".PA": {"currency": "EUR", "primary_exchange": "SBF"},      # Euronext Paris
    ".BR": {"currency": "EUR", "primary_exchange": "BVME"},     # Euronext Brussels
    ".DE": {"currency": "EUR", "primary_exchange": "IBIS"},     # Xetra (Frankfurt)
    ".L":  {"currency": "GBP", "primary_exchange": "LSE"},      # London Stock Exchange
}


def parse_symbol_for_ibkr(symbol: str) -> tuple[str, str, str | None]:
    """Parse a suffixed symbol into IBKR contract parameters.

    Args:
        symbol: Symbol with optional exchange suffix (e.g. "ASML.AS", "AAPL").

    Returns:
        Tuple of (bare_symbol, currency, primary_exchange).
        primary_exchange is None for US stocks (SMART routing handles it).
    """
    upper = symbol.upper()
    for suffix, info in SUFFIX_TO_IBKR.items():
        if upper.endswith(suffix):
            bare = symbol[: -len(suffix)]
            return bare, info["currency"], info["primary_exchange"]
    # US stock — no suffix
    return symbol, "USD", None


def minutes_until_close(
    exchange: Exchange,
    now: datetime | None = None,
) -> float | None:
    """Return minutes until the exchange closes, or None if market is not open today."""
    session = EXCHANGE_SESSIONS[exchange]
    tz = ZoneInfo(session.timezone)

    if now is None:
        now = datetime.now(timezone.utc)

    local_now = now.astimezone(tz)

    # Not a trading day
    if local_now.weekday() >= 5:
        return None
    holidays = EXCHANGE_HOLIDAYS.get(exchange, set())
    if local_now.date() in holidays:
        return None

    close_dt = local_now.replace(
        hour=session.close_time.hour,
        minute=session.close_time.minute,
        second=0,
        microsecond=0,
    )
    diff = (close_dt - local_now).total_seconds() / 60.0

    if diff <= 0:
        return None  # Already past close
    return diff


def minutes_until_close_for_symbol(
    symbol: str,
    now: datetime | None = None,
) -> float | None:
    """Return minutes until the relevant exchange closes for a symbol."""
    exchange = get_exchange_for_symbol(symbol)
    return minutes_until_close(exchange, now)


def is_any_market_open(
    symbols: list[str],
    now: datetime | None = None,
) -> bool:
    """Check if any relevant market is open for the given symbols."""
    exchanges = {get_exchange_for_symbol(s) for s in symbols}
    return any(is_market_open(ex, now) for ex in exchanges)


def next_market_open(exchange: Exchange, now: datetime | None = None) -> datetime:
    """Calculate when the market next opens."""
    session = EXCHANGE_SESSIONS[exchange]
    tz = ZoneInfo(session.timezone)

    if now is None:
        now = datetime.now(timezone.utc)

    local_now = now.astimezone(tz)
    candidate = local_now.replace(
        hour=session.open_time.hour,
        minute=session.open_time.minute,
        second=0, microsecond=0,
    )

    # If already past open today or market closed, try next day
    if local_now.time() >= session.open_time:
        candidate += timedelta(days=1)

    # Skip weekends and holidays
    holidays = EXCHANGE_HOLIDAYS.get(exchange, set())
    while candidate.weekday() >= 5 or candidate.date() in holidays:
        candidate += timedelta(days=1)

    return candidate.astimezone(timezone.utc)
