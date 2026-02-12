"""Tests for economic event calendar."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from app.data.economic_calendar import (
    EconomicCalendar,
    EconomicEvent,
    FOMC_DATES,
)


def _make_event(
    hours_ahead: float = 1.0,
    impact: str = "high",
    symbol: str | None = None,
    title: str = "Test Event",
) -> EconomicEvent:
    ts = datetime.now(timezone.utc) + timedelta(hours=hours_ahead)
    return EconomicEvent(
        timestamp=ts,
        title=title,
        impact=impact,
        symbol=symbol,
        source="test",
    )


class TestEconomicEvent:
    def test_to_dict(self):
        event = _make_event(title="FOMC Decision", impact="high")
        d = event.to_dict()
        assert d["title"] == "FOMC Decision"
        assert d["impact"] == "high"
        assert "timestamp" in d

    def test_from_dict_roundtrip(self):
        event = _make_event(title="CPI Report")
        d = event.to_dict()
        restored = EconomicEvent.from_dict(d)
        assert restored.title == "CPI Report"
        assert restored.impact == event.impact

    def test_from_dict_with_missing_fields(self):
        event = EconomicEvent.from_dict({"timestamp": "2026-01-01T00:00:00+00:00"})
        assert event.title == ""
        assert event.impact == "low"


class TestEconomicCalendar:
    def test_has_high_impact_macro_event(self):
        cal = EconomicCalendar()
        cal._events = [_make_event(hours_ahead=1.0, impact="high", symbol=None)]
        assert cal.has_high_impact_event("AAPL", within_hours=2)

    def test_has_high_impact_symbol_event(self):
        cal = EconomicCalendar()
        cal._events = [_make_event(hours_ahead=1.0, impact="high", symbol="AAPL")]
        assert cal.has_high_impact_event("AAPL", within_hours=2)
        assert not cal.has_high_impact_event("MSFT", within_hours=2)

    def test_no_high_impact_low_event(self):
        cal = EconomicCalendar()
        cal._events = [_make_event(hours_ahead=1.0, impact="low", symbol=None)]
        assert not cal.has_high_impact_event("AAPL", within_hours=2)

    def test_event_too_far_in_future(self):
        cal = EconomicCalendar()
        cal._events = [_make_event(hours_ahead=5.0, impact="high", symbol=None)]
        assert not cal.has_high_impact_event("AAPL", within_hours=2)

    def test_event_in_past(self):
        cal = EconomicCalendar()
        cal._events = [_make_event(hours_ahead=-1.0, impact="high", symbol=None)]
        assert not cal.has_high_impact_event("AAPL", within_hours=2)

    def test_get_upcoming_events(self):
        cal = EconomicCalendar()
        cal._events = [
            _make_event(hours_ahead=1.0, title="Soon"),
            _make_event(hours_ahead=48.0, title="Later"),
        ]
        upcoming = cal.get_upcoming_events(hours=24)
        assert len(upcoming) == 1
        assert upcoming[0].title == "Soon"

    def test_get_upcoming_events_empty(self):
        cal = EconomicCalendar()
        cal._events = []
        assert cal.get_upcoming_events(hours=24) == []

    def test_fomc_dates_format(self):
        """Verify FOMC dates are valid date strings."""
        for date_str in FOMC_DATES:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            assert dt.year in (2025, 2026)

    def test_get_fomc_events(self):
        cal = EconomicCalendar()
        # FOMC events within a wide window
        events = cal._get_fomc_events(days_ahead=365 * 2)
        # Should find some FOMC events
        for event in events:
            assert event.impact == "high"
            assert event.source == "fomc_calendar"
            assert "FOMC" in event.title
