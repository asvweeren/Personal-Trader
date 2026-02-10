import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote

import feedparser
import httpx
import redis.asyncio as redis
import structlog

from app.config import settings

logger = structlog.get_logger()


@dataclass
class NewsItem:
    title: str
    description: str
    source: str
    url: str
    published_at: datetime | None = None
    symbol: str | None = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "symbol": self.symbol,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NewsItem":
        pub = data.get("published_at")
        if pub and isinstance(pub, str):
            pub = datetime.fromisoformat(pub)
        return cls(
            title=data.get("title", ""),
            description=data.get("description", ""),
            source=data.get("source", ""),
            url=data.get("url", ""),
            published_at=pub,
            symbol=data.get("symbol"),
        )


class NewsFetcher:
    """Fetches financial news from multiple sources (RSS feeds + NewsAPI).

    Features:
    - Multiple RSS feeds for broad market coverage
    - Symbol-specific feeds (Yahoo Finance, Google News)
    - Optional NewsAPI.org integration when API key is set
    - Redis caching with 5-minute TTL to avoid hammering sources
    - Graceful error handling (returns empty list on failure)
    """

    # General market news RSS feeds
    RSS_FEEDS = [
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories"),
        (
            "CNBC Top News",
            "https://search.cnbc.com/rs/search/combinedcms/view.xml"
            "?partnerId=wrss01&id=100003114",
        ),
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
        ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
    ]

    CACHE_TTL = 300  # 5 minutes

    def __init__(self):
        self._api_key = settings.news_api_key
        self._http_client: httpx.AsyncClient | None = None
        self._redis: redis.Redis | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=15.0,
                headers={"User-Agent": "TraderBot/1.0"},
            )
        return self._http_client

    async def _get_redis(self) -> redis.Redis | None:
        """Get Redis connection, or None if unavailable."""
        if self._redis is None:
            try:
                self._redis = redis.from_url(settings.redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception:
                logger.debug("news.redis_unavailable")
                self._redis = None
        return self._redis

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
        if self._redis:
            await self._redis.close()

    # ── Cache helpers ─────────────────────────────────────────────

    async def _get_cached(self, cache_key: str) -> list[NewsItem] | None:
        """Return cached news items if available."""
        r = await self._get_redis()
        if not r:
            return None
        try:
            data = await r.get(cache_key)
            if data:
                items_data = json.loads(data)
                return [NewsItem.from_dict(d) for d in items_data]
        except Exception:
            logger.debug("news.cache_read_error", key=cache_key)
        return None

    async def _set_cached(self, cache_key: str, items: list[NewsItem]) -> None:
        """Store news items in Redis cache."""
        r = await self._get_redis()
        if not r:
            return
        try:
            data = json.dumps([item.to_dict() for item in items], default=str)
            await r.setex(cache_key, self.CACHE_TTL, data)
        except Exception:
            logger.debug("news.cache_write_error", key=cache_key)

    # ── Public API ────────────────────────────────────────────────

    async def fetch_news(self, symbol: str | None = None, limit: int = 20) -> list[NewsItem]:
        """Fetch news from all sources. Symbol-specific if provided.

        Results are cached in Redis with a 5-minute TTL.
        """
        cache_key = f"news:general:{symbol or 'market'}"
        cached = await self._get_cached(cache_key)
        if cached is not None:
            logger.debug("news.cache_hit", key=cache_key, count=len(cached))
            return cached[:limit]

        tasks = [self._fetch_all_rss()]

        # Add symbol-specific feeds when a symbol is given
        if symbol:
            tasks.append(self._fetch_symbol_rss(symbol))

        if self._api_key:
            tasks.append(self._fetch_newsapi(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[NewsItem] = []
        for result in results:
            if isinstance(result, list):
                items.extend(result)
            elif isinstance(result, Exception):
                logger.warning("news.source_error", error=str(result))

        # Deduplicate by title
        seen_titles: set[str] = set()
        unique_items: list[NewsItem] = []
        for item in items:
            title_key = item.title.strip().lower()
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_items.append(item)
        items = unique_items

        # Sort by publication date (newest first)
        items.sort(
            key=lambda x: x.published_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        # If symbol-specific, prioritize relevant news
        if symbol:
            symbol_upper = symbol.upper()
            relevant = [
                item
                for item in items
                if symbol_upper in (item.title or "").upper()
                or symbol_upper in (item.description or "").upper()
            ]
            general = [item for item in items if item not in relevant]
            items = relevant + general

        items = items[:limit]
        await self._set_cached(cache_key, items)
        return items

    async def fetch_symbol_news(self, symbol: str, limit: int = 10) -> list[NewsItem]:
        """Fetch news specifically about a symbol.

        Results are cached in Redis with a 5-minute TTL.
        """
        cache_key = f"news:symbol:{symbol}"
        cached = await self._get_cached(cache_key)
        if cached is not None:
            logger.debug("news.cache_hit", key=cache_key, count=len(cached))
            return cached[:limit]

        items: list[NewsItem] = []

        # Fetch from symbol-specific RSS feeds
        symbol_rss = await self._fetch_symbol_rss(symbol)
        items.extend(symbol_rss)

        if self._api_key:
            newsapi_items = await self._fetch_newsapi(symbol)
            items.extend(newsapi_items)

        # Also check general RSS for mentions
        rss_items = await self._fetch_all_rss()
        symbol_upper = symbol.upper()
        for item in rss_items:
            if (
                symbol_upper in (item.title or "").upper()
                or symbol_upper in (item.description or "").upper()
            ):
                item.symbol = symbol
                items.append(item)

        # Deduplicate
        seen_titles: set[str] = set()
        unique_items: list[NewsItem] = []
        for item in items:
            title_key = item.title.strip().lower()
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_items.append(item)
        items = unique_items

        items.sort(
            key=lambda x: x.published_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        items = items[:limit]

        # Tag all items with the symbol
        for item in items:
            item.symbol = symbol

        await self._set_cached(cache_key, items)
        return items

    # ── RSS feeds ─────────────────────────────────────────────────

    async def _fetch_all_rss(self) -> list[NewsItem]:
        """Fetch from all general RSS feeds concurrently."""
        tasks = [self._fetch_single_rss(name, url) for name, url in self.RSS_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[NewsItem] = []
        for result in results:
            if isinstance(result, list):
                items.extend(result)
        return items

    async def _fetch_symbol_rss(self, symbol: str) -> list[NewsItem]:
        """Fetch symbol-specific news from Yahoo Finance and Google News RSS."""
        tasks = [
            self._fetch_single_rss(
                f"Yahoo Finance ({symbol})",
                f"https://finance.yahoo.com/rss/headline?s={quote(symbol)}",
            ),
            self._fetch_single_rss(
                f"Google News ({symbol})",
                f"https://news.google.com/rss/search?q={quote(symbol)}+stock&hl=en-US&gl=US&ceid=US:en",
            ),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[NewsItem] = []
        for result in results:
            if isinstance(result, list):
                for item in result:
                    item.symbol = symbol
                items.extend(result)
        return items

    async def _fetch_single_rss(self, source_name: str, url: str) -> list[NewsItem]:
        """Fetch a single RSS feed."""
        try:
            client = await self._get_client()
            resp = await client.get(url)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)

            items: list[NewsItem] = []
            for entry in feed.entries[:15]:
                published_at = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        published_at = datetime(
                            *entry.published_parsed[:6], tzinfo=timezone.utc
                        )
                    except Exception:
                        published_at = datetime.now(timezone.utc)
                else:
                    published_at = datetime.now(timezone.utc)

                items.append(
                    NewsItem(
                        title=entry.get("title", "").strip(),
                        description=_clean_html(entry.get("summary", "").strip()),
                        source=source_name,
                        url=entry.get("link", ""),
                        published_at=published_at,
                    )
                )
            return items
        except Exception:
            logger.debug("news.rss_error", source=source_name)
            return []

    # ── NewsAPI ───────────────────────────────────────────────────

    async def _fetch_newsapi(self, symbol: str | None = None) -> list[NewsItem]:
        """Fetch from NewsAPI with symbol-specific query."""
        query = f"{symbol} stock" if symbol else "stock market trading"
        try:
            client = await self._get_client()
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "apiKey": self._api_key,
                    "sortBy": "publishedAt",
                    "pageSize": 15,
                    "language": "en",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                NewsItem(
                    title=(a.get("title") or "").strip(),
                    description=(a.get("description") or "").strip(),
                    source=a.get("source", {}).get("name", "NewsAPI"),
                    url=a.get("url", ""),
                    published_at=(
                        datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
                        if a.get("publishedAt")
                        else None
                    ),
                    symbol=symbol,
                )
                for a in data.get("articles", [])
                if a.get("title")  # Skip empty titles
            ]
        except Exception:
            logger.debug("news.newsapi_error", query=query)
            return []


def _clean_html(text: str) -> str:
    """Remove basic HTML tags from text."""
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()
