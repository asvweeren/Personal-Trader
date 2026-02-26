import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import anthropic
import redis.asyncio as aioredis
import structlog

from app.config import settings
from app.data.news_fetcher import NewsItem

logger = structlog.get_logger()

# Use Haiku for cost efficiency (fast + cheap)
SENTIMENT_MODEL = "claude-haiku-4-5-20251001"

# Redis cache TTL for sentiment results
SENTIMENT_CACHE_TTL = 3600  # 60 minutes (matches refresh interval)

# Maximum headlines to send per API call
MAX_HEADLINES_PER_BATCH = 10


@dataclass
class SentimentResult:
    symbol: str
    score: float  # -1.0 (very bearish) to 1.0 (very bullish)
    confidence: float
    reasoning: str
    news_count: int
    headlines_analyzed: list[str] | None = None
    timestamp: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "score": self.score,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "news_count": self.news_count,
            "headlines_analyzed": self.headlines_analyzed or [],
            "timestamp": (self.timestamp or datetime.now(UTC)).isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SentimentResult":
        ts = data.get("timestamp")
        if ts and isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            symbol=data.get("symbol", ""),
            score=float(data.get("score", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
            news_count=int(data.get("news_count", 0)),
            headlines_analyzed=data.get("headlines_analyzed"),
            timestamp=ts,
        )


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
    - Uses Claude Haiku for cost efficiency
    - Rate limiting to control API costs
    - Redis cache with 15-minute TTL for sentiment results
    - In-memory fallback cache when Redis is unavailable
    - Graceful degradation when API key is not set (returns neutral)
    - Structured prompt for consistent JSON output
    """

    def __init__(
        self,
        max_calls_per_minute: int = 10,
        cache_ttl_seconds: int = SENTIMENT_CACHE_TTL,
    ):
        self._api_key = settings.anthropic_api_key
        self._client: anthropic.AsyncAnthropic | None = None
        if self._api_key:
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        else:
            logger.warning(
                "sentiment.no_api_key",
                msg="Anthropic API key not set; sentiment analysis will return neutral scores",
            )

        self._rate_limiter = RateLimiter(max_calls=max_calls_per_minute, period_seconds=60)

        # Track when client was disabled for periodic reset
        self._client_disabled_at: float | None = None
        self._client_reset_cooldown = 3600  # Retry API client after 1 hour

        # In-memory fallback cache
        self._memory_cache: dict[str, tuple[SentimentResult, float]] = {}
        self._cache_ttl = cache_ttl_seconds

        # Redis connection (lazy init)
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis | None:
        """Get Redis connection, or None if unavailable."""
        if self._redis is None:
            try:
                self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception:
                logger.debug("sentiment.redis_unavailable")
                self._redis = None
        return self._redis

    async def close(self) -> None:
        """Close connections."""
        if self._redis:
            await self._redis.close()

    # ── Cache layer (Redis with in-memory fallback) ───────────────

    async def _get_cached(self, symbol: str) -> SentimentResult | None:
        """Return cached sentiment result if available. Tries Redis first, then memory."""
        cache_key = f"sentiment:analysis:{symbol}"

        # Try Redis
        r = await self._get_redis()
        if r:
            try:
                data = await r.get(cache_key)
                if data:
                    logger.debug("sentiment.redis_cache_hit", symbol=symbol)
                    return SentimentResult.from_dict(json.loads(data))
            except Exception:
                logger.debug("sentiment.redis_cache_error", symbol=symbol)

        # Fallback to in-memory cache
        if symbol in self._memory_cache:
            result, cached_at = self._memory_cache[symbol]
            if time.monotonic() - cached_at < self._cache_ttl:
                logger.debug("sentiment.memory_cache_hit", symbol=symbol)
                return result
            del self._memory_cache[symbol]

        return None

    async def _set_cached(self, symbol: str, result: SentimentResult) -> None:
        """Store sentiment result in Redis and in-memory cache."""
        cache_key = f"sentiment:analysis:{symbol}"

        # Store in Redis
        r = await self._get_redis()
        if r:
            try:
                data = json.dumps(result.to_dict(), default=str)
                await r.setex(cache_key, self._cache_ttl, data)
            except Exception:
                logger.debug("sentiment.redis_cache_write_error", symbol=symbol)

        # Always store in memory as fallback
        self._memory_cache[symbol] = (result, time.monotonic())

    # ── Public API ────────────────────────────────────────────────

    async def analyze(self, symbol: str, news_items: list[NewsItem]) -> SentimentResult:
        """Analyze sentiment for a symbol based on news items.

        Returns cached result if available. Falls back to neutral sentiment
        if no API key is configured or if the API call fails.
        """
        if not news_items:
            return SentimentResult(
                symbol=symbol,
                score=0.0,
                confidence=0.0,
                reasoning="No news available",
                news_count=0,
                headlines_analyzed=[],
                timestamp=datetime.now(UTC),
            )

        # Check cache first
        cached = await self._get_cached(symbol)
        if cached:
            return cached

        # If client was disabled, try to reset after cooldown
        if not self._client and self._api_key and self._client_disabled_at:
            if time.monotonic() - self._client_disabled_at >= self._client_reset_cooldown:
                logger.info("sentiment.client_reset_attempt")
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
                self._client_disabled_at = None

        # If no API key or client still disabled, return neutral sentiment
        if not self._client:
            return self._neutral_result(
                symbol, news_items, reason="Anthropic API key not configured"
            )

        # Rate limit before API call
        await self._rate_limiter.acquire()

        result = await self._call_api(symbol, news_items)

        # Cache the result
        await self._set_cached(symbol, result)

        return result

    async def analyze_batch(
        self, symbols_news: dict[str, list[NewsItem]]
    ) -> dict[str, SentimentResult]:
        """Analyze sentiment for multiple symbols."""
        results: dict[str, SentimentResult] = {}
        for symbol, news in symbols_news.items():
            results[symbol] = await self.analyze(symbol, news)
        return results

    def clear_cache(self) -> None:
        """Clear in-memory cache. Redis cache expires via TTL."""
        self._memory_cache.clear()

    # ── API call ──────────────────────────────────────────────────

    async def _call_api(self, symbol: str, news_items: list[NewsItem]) -> SentimentResult:
        """Make the actual Claude API call for sentiment analysis.

        Batches headlines (max 10 at a time) and prompts Claude to return
        structured JSON with sentiment_score and reasoning.
        """
        # Limit to MAX_HEADLINES_PER_BATCH headlines
        batch = news_items[:MAX_HEADLINES_PER_BATCH]
        headlines = [item.title for item in batch if item.title.strip()]

        news_text = "\n\n".join(
            f"[{item.source}] {item.title}\n{item.description}"
            for item in batch
            if item.title.strip()
        )

        # Strip exchange suffix for clearer prompt (e.g. "ASML.AS" → "ASML")
        display_symbol = symbol.split(".")[0] if "." in symbol else symbol
        prompt = (
            f"You are a financial sentiment analyst. Analyze these news articles about "
            f"{display_symbol} stock.\n\n"
            "Score the overall sentiment and your confidence in the assessment.\n\n"
            "Rules:\n"
            "- score: -1.0 (extremely bearish) to 1.0 (extremely bullish), 0.0 = neutral\n"
            "- confidence: 0.0 (no confidence) to 1.0 (very confident)\n"
            "- reasoning: 1-2 sentences explaining your assessment\n"
            "- Consider both direct company impact and broader market implications\n"
            "- If news is generic/unrelated, score 0.0 with low confidence\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"score": 0.0, "confidence": 0.0, "reasoning": "..."}\n\n'
            f"News articles:\n{news_text}"
        )

        try:
            response = await self._client.messages.create(
                model=SENTIMENT_MODEL,
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
                news_count=len(batch),
            )

            return SentimentResult(
                symbol=symbol,
                score=score,
                confidence=confidence,
                reasoning=result.get("reasoning", "No reasoning provided"),
                news_count=len(batch),
                headlines_analyzed=headlines,
                timestamp=datetime.now(UTC),
            )
        except json.JSONDecodeError:
            logger.warning("sentiment.parse_error", symbol=symbol, raw=raw_text[:200])
            return self._neutral_result(
                symbol, batch, reason="Failed to parse LLM response"
            )
        except anthropic.AuthenticationError:
            logger.error("sentiment.auth_error", symbol=symbol)
            # Disable client temporarily — will retry after cooldown
            self._client = None
            self._client_disabled_at = time.monotonic()
            return self._neutral_result(
                symbol, batch, reason="Invalid Anthropic API key"
            )
        except anthropic.BadRequestError as e:
            error_msg = str(e)
            if "usage limits" in error_msg or "rate" in error_msg.lower():
                logger.warning("sentiment.api_budget_exhausted", detail=error_msg[:200])
                # Disable client temporarily — will retry after cooldown
                self._client = None
                self._client_disabled_at = time.monotonic()
            else:
                logger.error("sentiment.api_bad_request", symbol=symbol, error=error_msg[:200])
            return self._neutral_result(
                symbol, batch, reason="API budget/rate limit reached"
            )
        except Exception:
            logger.exception("sentiment.api_error", symbol=symbol)
            return self._neutral_result(
                symbol, batch, reason="API call failed"
            )

    # ── Helpers ───────────────────────────────────────────────────

    def _neutral_result(
        self, symbol: str, news_items: list[NewsItem], reason: str
    ) -> SentimentResult:
        """Return a neutral sentiment result for graceful degradation."""
        return SentimentResult(
            symbol=symbol,
            score=0.0,
            confidence=0.0,
            reasoning=reason,
            news_count=len(news_items),
            headlines_analyzed=[item.title for item in news_items if item.title.strip()],
            timestamp=datetime.now(UTC),
        )

    def to_dict(self, result: SentimentResult) -> dict:
        """Convert SentimentResult to dict for storage/serialization."""
        return result.to_dict()
