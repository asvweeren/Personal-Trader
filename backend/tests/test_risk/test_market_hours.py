from datetime import datetime, time, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.risk.market_hours import (
    Exchange,
    is_market_open,
    get_exchange_for_symbol,
    is_any_market_open,
    next_market_open,
)


def _make_dt(year, month, day, hour, minute, tz_name="America/New_York"):
    """Create a timezone-aware datetime."""
    tz = ZoneInfo(tz_name)
    return datetime(year, month, day, hour, minute, tzinfo=tz).astimezone(timezone.utc)


# ── is_market_open tests ─────────────────────────────────────


def test_nyse_open_during_hours():
    # Wednesday 10:30 AM ET → market should be open
    dt = _make_dt(2026, 3, 4, 10, 30)
    assert is_market_open(Exchange.NYSE, dt) is True


def test_nyse_closed_before_open():
    # Wednesday 8:00 AM ET → before 9:30 open
    dt = _make_dt(2026, 3, 4, 8, 0)
    assert is_market_open(Exchange.NYSE, dt) is False


def test_nyse_closed_after_close():
    # Wednesday 4:30 PM ET → after 4:00 close
    dt = _make_dt(2026, 3, 4, 16, 30)
    assert is_market_open(Exchange.NYSE, dt) is False


def test_nyse_closed_near_close_with_buffer():
    # Wednesday 3:57 PM ET → within 5 min buffer of close
    dt = _make_dt(2026, 3, 4, 15, 57)
    assert is_market_open(Exchange.NYSE, dt, buffer_minutes=5) is False


def test_nyse_closed_on_weekend():
    # Saturday 12:00 PM ET
    dt = _make_dt(2026, 3, 7, 12, 0)
    assert is_market_open(Exchange.NYSE, dt) is False


def test_nyse_closed_on_sunday():
    dt = _make_dt(2026, 3, 8, 12, 0)
    assert is_market_open(Exchange.NYSE, dt) is False


def test_nyse_closed_on_holiday():
    # July 3, 2026 is Independence Day (observed)
    dt = _make_dt(2026, 7, 3, 12, 0)
    assert is_market_open(Exchange.NYSE, dt) is False


def test_euronext_open_during_hours():
    # Wednesday 14:00 Amsterdam time → market should be open (9:00-17:30)
    dt = _make_dt(2026, 3, 4, 14, 0, "Europe/Amsterdam")
    assert is_market_open(Exchange.EURONEXT, dt) is True


def test_euronext_closed_on_labour_day():
    # May 1 is a holiday in EU
    dt = _make_dt(2026, 5, 1, 14, 0, "Europe/Amsterdam")
    assert is_market_open(Exchange.EURONEXT, dt) is False


# ── get_exchange_for_symbol tests ─────────────────────────────


def test_us_stock_defaults_to_nyse():
    assert get_exchange_for_symbol("AAPL") == Exchange.NYSE
    assert get_exchange_for_symbol("MSFT") == Exchange.NYSE


def test_amsterdam_suffix_returns_euronext():
    assert get_exchange_for_symbol("ASML.AS") == Exchange.EURONEXT


def test_london_suffix_returns_lse():
    assert get_exchange_for_symbol("HSBA.L") == Exchange.LSE


def test_german_suffix_returns_xetra():
    assert get_exchange_for_symbol("SAP.DE") == Exchange.XETRA


# ── is_any_market_open tests ─────────────────────────────────


def test_any_market_open_mixed_symbols():
    # During US market hours but not EU
    dt = _make_dt(2026, 3, 4, 14, 0)  # 2PM ET, EU markets closed
    result = is_any_market_open(["AAPL", "ASML.AS"], dt)
    assert result is True  # NYSE is open


def test_no_markets_open_on_weekend():
    dt = _make_dt(2026, 3, 7, 12, 0)  # Saturday
    assert is_any_market_open(["AAPL", "MSFT"], dt) is False


# ── next_market_open tests ────────────────────────────────────


def test_next_open_from_weekend():
    # Saturday → should return Monday
    dt = _make_dt(2026, 3, 7, 12, 0)
    next_open = next_market_open(Exchange.NYSE, dt)
    # Should be Monday March 9, 9:30 AM ET
    et = ZoneInfo("America/New_York")
    local = next_open.astimezone(et)
    assert local.weekday() == 0  # Monday
    assert local.hour == 9
    assert local.minute == 30
