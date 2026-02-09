import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import httpx
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


class NewsFetcher:
    """Fetches financial news from multiple sources (RSS feeds + NewsAPI)."""

    RSS_FEEDS = [
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories"),
        ("CNBC Top News", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
        ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
    ]

    def __init__(self):
        self._api_key = settings.news_api_key
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=15.0)
        return self._http_client

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def fetch_news(self, symbol: str | None = None, limit: int = 20) -> list[NewsItem]:
        """Fetch news from all sources. Symbol-specific if provided."""
        tasks = [self._fetch_all_rss()]
        if self._api_key:
            tasks.append(self._fetch_newsapi(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        items = []
        for result in results:
            if isinstance(result, list):
                items.extend(result)
            elif isinstance(result, Exception):
                logger.warning("news.source_error", error=str(result))

        # Sort by publication date (newest first)
        items.sort(key=lambda x: x.published_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

        # If symbol-specific, filter for relevance
        if symbol:
            symbol_upper = symbol.upper()
            relevant = [
                item for item in items
                if symbol_upper in (item.title or "").upper()
                or symbol_upper in (item.description or "").upper()
            ]
            # Mix symbol-relevant news with general market news
            general = [item for item in items if item not in relevant]
            items = relevant + general

        return items[:limit]

    async def fetch_symbol_news(self, symbol: str, limit: int = 10) -> list[NewsItem]:
        """Fetch news specifically about a symbol."""
        items = []
        if self._api_key:
            items.extend(await self._fetch_newsapi(symbol))

        # Also check RSS for mentions
        rss_items = await self._fetch_all_rss()
        symbol_upper = symbol.upper()
        for item in rss_items:
            if symbol_upper in (item.title or "").upper() or symbol_upper in (item.description or "").upper():
                item.symbol = symbol
                items.append(item)

        items.sort(key=lambda x: x.published_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return items[:limit]

    async def _fetch_all_rss(self) -> list[NewsItem]:
        """Fetch from all RSS feeds concurrently."""
        tasks = [self._fetch_single_rss(name, url) for name, url in self.RSS_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items = []
        for result in results:
            if isinstance(result, list):
                items.extend(result)
        return items

    async def _fetch_single_rss(self, source_name: str, url: str) -> list[NewsItem]:
        """Fetch a single RSS feed."""
        try:
            client = await self._get_client()
            resp = await client.get(url)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)

            items = []
            for entry in feed.entries[:15]:
                published_at = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
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
    import re
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()
