import asyncio
import json
import time
from dataclasses import dataclass

import anthropic
import structlog

from app.config import settings
from app.data.news_fetcher import NewsItem

logger = structlog.get_logger()


@dataclass
class SentimentResult:
    symbol: str
    score: float  # -1.0 (very bearish) to 1.0 (very bullish)
    confidence: float
    reasoning: str
    news_count: int


class RateLimiter:
    """Simple token-bucket rate limiter for API calls."""

    def __init__(self, max_calls: int, period_seconds: float):
        self._max_calls = max_calls
        self._period = period_seconds
        self._calls: list[float] = []

    async def acquire(self) -> None:
        """Wait until a call is allowed."""
        now = time.monotonic()
        # Remove expired timestamps
        self._calls = [t for t in self._calls if now - t < self._period]
        if len(self._calls) >= self._max_calls:
            wait_time = self._period - (now - self._calls[0])
            if wait_time > 0:
                logger.debug("rate_limiter.waiting", wait_seconds=round(wait_time, 1))
                await asyncio.sleep(wait_time)
        self._calls.append(time.monotonic())


class SentimentAnalyzer:
    """Uses Claude LLM to analyze market sentiment from news.

    Features:
    - Rate limiting to control API costs
    - In-memory cache for recent results
    - Structured prompt for consistent JSON output
    """

    def __init__(
        self,
        max_calls_per_minute: int = 10,
        cache_ttl_seconds: int = 600,
    ):
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._rate_limiter = RateLimiter(max_calls=max_calls_per_minute, period_seconds=60)
        self._cache: dict[str, tuple[SentimentResult, float]] = {}
        self._cache_ttl = cache_ttl_seconds

    async def analyze(self, symbol: str, news_items: list[NewsItem]) -> SentimentResult:
        """Analyze sentiment for a symbol. Uses cache if available."""
        if not news_items:
            return SentimentResult(
                symbol=symbol,
                score=0.0,
                confidence=0.0,
                reasoning="No news available",
                news_count=0,
            )

        # Check cache
        cached = self._get_cached(symbol)
        if cached:
            logger.debug("sentiment.cache_hit", symbol=symbol)
            return cached

        # Rate limit before API call
        await self._rate_limiter.acquire()

        result = await self._call_api(symbol, news_items)

        # Store in cache
        self._cache[symbol] = (result, time.monotonic())

        return result

    async def analyze_batch(
        self, symbols_news: dict[str, list[NewsItem]]
    ) -> dict[str, SentimentResult]:
        """Analyze sentiment for multiple symbols."""
        results = {}
        for symbol, news in symbols_news.items():
            results[symbol] = await self.analyze(symbol, news)
        return results

    def _get_cached(self, symbol: str) -> SentimentResult | None:
        """Return cached result if still valid."""
        if symbol in self._cache:
            result, cached_at = self._cache[symbol]
            if time.monotonic() - cached_at < self._cache_ttl:
                return result
            del self._cache[symbol]
        return None

    def clear_cache(self) -> None:
        self._cache.clear()

    async def _call_api(self, symbol: str, news_items: list[NewsItem]) -> SentimentResult:
        """Make the actual Claude API call for sentiment analysis."""
        news_text = "\n\n".join(
            f"[{item.source}] {item.title}\n{item.description}"
            for item in news_items[:10]
        )

        prompt = (
            f"You are a financial sentiment analyst. Analyze these news articles about {symbol} stock.\n\n"
            "Score the overall sentiment and your confidence in the assessment.\n\n"
            "Rules:\n"
            "- score: -1.0 (extremely bearish) to 1.0 (extremely bullish), 0.0 = neutral\n"
            "- confidence: 0.0 (no confidence) to 1.0 (very confident)\n"
            "- reasoning: 1-2 sentences explaining your assessment\n"
            "- Consider both direct company impact and broader market implications\n"
            "- If news is generic/unrelated, score 0.0 with low confidence\n\n"
            f"Respond ONLY with valid JSON:\n"
            '{"score": 0.0, "confidence": 0.0, "reasoning": "..."}\n\n'
            f"News articles:\n{news_text}"
        )

        try:
            response = await self._client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text.strip()
            # Handle potential markdown code block wrapping
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(raw_text)

            # Clamp values to valid ranges
            score = max(-1.0, min(1.0, float(result.get("score", 0.0))))
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))

            logger.info(
                "sentiment.analyzed",
                symbol=symbol,
                score=score,
                confidence=confidence,
                news_count=len(news_items),
            )

            return SentimentResult(
                symbol=symbol,
                score=score,
                confidence=confidence,
                reasoning=result.get("reasoning", "No reasoning provided"),
                news_count=len(news_items),
            )
        except json.JSONDecodeError:
            logger.warning("sentiment.parse_error", symbol=symbol, raw=raw_text[:200])
            return SentimentResult(
                symbol=symbol,
                score=0.0,
                confidence=0.0,
                reasoning="Failed to parse LLM response",
                news_count=len(news_items),
            )
        except Exception:
            logger.exception("sentiment.api_error", symbol=symbol)
            return SentimentResult(
                symbol=symbol,
                score=0.0,
                confidence=0.0,
                reasoning="API call failed",
                news_count=len(news_items),
            )

    def to_dict(self, result: SentimentResult) -> dict:
        """Convert SentimentResult to dict for storage/serialization."""
        return {
            "symbol": result.symbol,
            "score": result.score,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "news_count": result.news_count,
        }
