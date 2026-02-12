"""Economic event calendar: earnings, Fed meetings, macro events.

Data sources (all free, no extra API keys):
1. Yahoo Finance earnings via yfinance
2. FOMC meeting dates (hardcoded, updated annually)
3. Trading Economics RSS via feedparser
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import feedparser
import redis.asyncio as aioredis
import structlog

from app.config import settings

logger = structlog.get_logger()

# FOMC meeting dates for 2025-2026
FOMC_DATES = [
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

TRADING_ECONOMICS_RSS = "https://tradingeconomics.com/rss/calendar.aspx"


@dataclass
class EconomicEvent:
    timestamp: datetime
    title: str
    impact: str  # "high", "medium", "low"
    symbol: str | None = None  # None = macro event
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "impact": self.impact,
            "symbol": self.symbol,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EconomicEvent":
        ts = data.get("timestamp", "")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            timestamp=ts,
            title=data.get("title", ""),
            impact=data.get("impact", "low"),
            symbol=data.get("symbol"),
            source=data.get("source", ""),
        )


class EconomicCalendar:
    """Aggregates economic events from multiple free sources."""

    CACHE_KEY = "economic_calendar:events"
    CACHE_TTL = 6 * 3600  # 6 hours

    def __init__(self):
        self._events: list[EconomicEvent] = []
        self._redis: aioredis.Redis | None = None
        self._last_fetch: datetime | None = None

    async def _get_redis(self) -> aioredis.Redis | None:
        if self._redis is None:
            try:
                self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception:
                self._redis = None
        return self._redis

    async def fetch_events(self, days_ahead: int = 7) -> list[EconomicEvent]:
        """Fetch events from all sources, with Redis caching."""
        # Try cache first
        r = await self._get_redis()
        if r:
            try:
                cached = await r.get(self.CACHE_KEY)
                if cached:
                    data = json.loads(cached)
                    self._events = [EconomicEvent.from_dict(d) for d in data]
                    logger.debug("calendar.cache_hit", count=len(self._events))
                    return self._filter_upcoming(days_ahead)
            except Exception:
                pass

        # Fetch fresh events
        events: list[EconomicEvent] = []

        # 1. FOMC dates
        events.extend(self._get_fomc_events(days_ahead))

        # 2. Yahoo Finance earnings
        try:
            earnings = await self._fetch_earnings(days_ahead)
            events.extend(earnings)
        except Exception:
            logger.debug("calendar.earnings_fetch_error", exc_info=True)

        # 3. Trading Economics RSS
        try:
            macro = await self._fetch_trading_economics()
            events.extend(macro)
        except Exception:
            logger.debug("calendar.rss_fetch_error", exc_info=True)

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        self._events = events
        self._last_fetch = datetime.now(timezone.utc)

        # Cache in Redis
        if r and events:
            try:
                data = json.dumps([e.to_dict() for e in events], default=str)
                await r.setex(self.CACHE_KEY, self.CACHE_TTL, data)
            except Exception:
                pass

        logger.info("calendar.fetched", events=len(events))
        return self._filter_upcoming(days_ahead)

    def has_high_impact_event(self, symbol: str, within_hours: int = 2) -> bool:
        """Check if there's a high-impact event for this symbol within N hours."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=within_hours)

        for event in self._events:
            if event.impact != "high":
                continue
            if event.timestamp < now or event.timestamp > cutoff:
                continue
            # Macro events affect all symbols
            if event.symbol is None:
                return True
            # Symbol-specific event
            if event.symbol and event.symbol.upper() == symbol.upper():
                return True

        return False

    def get_upcoming_events(self, hours: int = 24) -> list[EconomicEvent]:
        """Get events in the next N hours."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)
        return [
            e for e in self._events
            if now <= e.timestamp <= cutoff
        ]

    def _filter_upcoming(self, days: int) -> list[EconomicEvent]:
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days)
        return [e for e in self._events if e.timestamp >= now and e.timestamp <= cutoff]

    def _get_fomc_events(self, days_ahead: int) -> list[EconomicEvent]:
        """Generate FOMC meeting events from hardcoded calendar."""
        events = []
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)

        for date_str in FOMC_DATES:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    hour=18, minute=0, tzinfo=timezone.utc  # 2 PM ET
                )
                if now <= dt <= cutoff:
                    events.append(EconomicEvent(
                        timestamp=dt,
                        title="FOMC Interest Rate Decision",
                        impact="high",
                        symbol=None,
                        source="fomc_calendar",
                    ))
            except ValueError:
                continue

        return events

    async def _fetch_earnings(self, days_ahead: int) -> list[EconomicEvent]:
        """Fetch upcoming earnings from yfinance."""
        import asyncio

        try:
            import yfinance as yf
        except ImportError:
            return []

        def _get_earnings():
            events = []
            try:
                # yfinance earnings calendar
                from datetime import date as date_type
                start = date_type.today()
                end = start + timedelta(days=days_ahead)

                # Get S&P 500 earnings calendar
                cal = yf.Ticker("SPY")
                try:
                    earnings_dates = cal.earnings_dates
                    if earnings_dates is not None and not earnings_dates.empty:
                        for ts, row in earnings_dates.iterrows():
                            try:
                                dt = ts.to_pydatetime()
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                                if dt.date() >= start and dt.date() <= end:
                                    events.append(EconomicEvent(
                                        timestamp=dt,
                                        title=f"Earnings: {getattr(row, 'name', 'Unknown')}",
                                        impact="high",
                                        symbol=None,
                                        source="yfinance",
                                    ))
                            except Exception:
                                continue
                except Exception:
                    pass
            except Exception:
                pass
            return events

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_earnings)

    async def _fetch_trading_economics(self) -> list[EconomicEvent]:
        """Fetch macro events from Trading Economics RSS."""
        import httpx

        events = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(TRADING_ECONOMICS_RSS)
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)

                for entry in feed.entries[:30]:
                    title = entry.get("title", "").strip()
                    if not title:
                        continue

                    published_at = datetime.now(timezone.utc)
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        try:
                            published_at = datetime(
                                *entry.published_parsed[:6], tzinfo=timezone.utc
                            )
                        except Exception:
                            pass

                    # Classify impact based on keywords
                    impact = "low"
                    high_keywords = [
                        "interest rate", "nonfarm", "payroll", "gdp", "cpi",
                        "inflation", "unemployment", "fed", "fomc", "ecb",
                    ]
                    medium_keywords = [
                        "retail sales", "pmi", "consumer", "housing",
                        "trade balance", "industrial",
                    ]
                    title_lower = title.lower()
                    if any(kw in title_lower for kw in high_keywords):
                        impact = "high"
                    elif any(kw in title_lower for kw in medium_keywords):
                        impact = "medium"

                    events.append(EconomicEvent(
                        timestamp=published_at,
                        title=title,
                        impact=impact,
                        symbol=None,
                        source="trading_economics",
                    ))
        except Exception:
            logger.debug("calendar.trading_economics_error", exc_info=True)

        return events


# Singleton
_calendar: EconomicCalendar | None = None


def get_economic_calendar() -> EconomicCalendar:
    global _calendar
    if _calendar is None:
        _calendar = EconomicCalendar()
    return _calendar
