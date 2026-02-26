"""AI-driven position sizing modifier using Claude LLM."""

import json
import time
from dataclasses import dataclass, field

import anthropic
import redis.asyncio as aioredis
import structlog

from app.config import settings
from app.data.sentiment import RateLimiter

logger = structlog.get_logger()

AI_SIZING_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class AISizingResult:
    modifier: float  # 0.5–1.5 multiplier on Kelly-based size
    reasoning: str
    risk_factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "modifier": self.modifier,
            "reasoning": self.reasoning,
            "risk_factors": self.risk_factors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AISizingResult":
        return cls(
            modifier=float(data.get("modifier", 1.0)),
            reasoning=data.get("reasoning", ""),
            risk_factors=data.get("risk_factors", []),
        )


def _default_result() -> AISizingResult:
    """Neutral modifier (no effect) for graceful degradation."""
    return AISizingResult(modifier=1.0, reasoning="Default (no AI adjustment)", risk_factors=[])


class AISizingAdvisor:
    """Uses Claude LLM to advise on position sizing adjustments.

    Acts as an extra multiplier (0.5x–1.5x) on top of the existing
    Kelly criterion calculation. On any failure, returns 1.0 (no effect).

    Features:
    - Uses Claude Haiku for cost efficiency (~$0.0003/call)
    - Redis cache with configurable TTL (default 15 min)
    - In-memory fallback cache
    - Rate limiting (reuses RateLimiter from sentiment.py)
    - Graceful degradation: modifier=1.0 on any error
    """

    def __init__(
        self,
        max_calls_per_minute: int = 10,
        cache_ttl_seconds: int | None = None,
    ):
        self._api_key = settings.anthropic_api_key
        self._client: anthropic.AsyncAnthropic | None = None
        if self._api_key:
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        else:
            logger.warning(
                "ai_sizing.no_api_key",
                msg="Anthropic API key not set; AI sizing will return neutral modifier",
            )

        self._rate_limiter = RateLimiter(max_calls=max_calls_per_minute, period_seconds=60)
        self._cache_ttl = cache_ttl_seconds or settings.ai_sizing_cache_ttl

        # In-memory fallback cache
        self._memory_cache: dict[str, tuple[AISizingResult, float]] = {}

        # API call tracking
        self.call_count: int = 0
        self.estimated_cost_usd: float = 0.0

        # Redis connection (lazy init)
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis | None:
        """Get Redis connection, or None if unavailable."""
        if self._redis is None:
            try:
                self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception:
                logger.debug("ai_sizing.redis_unavailable")
                self._redis = None
        return self._redis

    async def close(self) -> None:
        """Close connections."""
        if self._redis:
            await self._redis.close()

    # ── Cache layer ───────────────────────────────────────────────

    async def _get_cached(self, symbol: str) -> AISizingResult | None:
        """Return cached result if available. Tries Redis first, then memory."""
        cache_key = f"ai_sizing:{symbol}"

        r = await self._get_redis()
        if r:
            try:
                data = await r.get(cache_key)
                if data:
                    logger.debug("ai_sizing.cache_hit", symbol=symbol, source="redis")
                    return AISizingResult.from_dict(json.loads(data))
            except Exception:
                logger.debug("ai_sizing.redis_cache_error", symbol=symbol)

        if symbol in self._memory_cache:
            result, cached_at = self._memory_cache[symbol]
            if time.monotonic() - cached_at < self._cache_ttl:
                logger.debug("ai_sizing.cache_hit", symbol=symbol, source="memory")
                return result
            del self._memory_cache[symbol]

        return None

    async def _set_cached(self, symbol: str, result: AISizingResult) -> None:
        """Store result in Redis and in-memory cache."""
        cache_key = f"ai_sizing:{symbol}"

        r = await self._get_redis()
        if r:
            try:
                data = json.dumps(result.to_dict())
                await r.setex(cache_key, self._cache_ttl, data)
            except Exception:
                logger.debug("ai_sizing.redis_cache_write_error", symbol=symbol)

        self._memory_cache[symbol] = (result, time.monotonic())

    # ── Public API ────────────────────────────────────────────────

    async def get_modifier(
        self,
        symbol: str,
        signal_confidence: float,
        strategy_name: str,
        portfolio_summary: dict,
        features: dict | None = None,
        sentiment: dict | None = None,
    ) -> AISizingResult:
        """Get AI-advised position sizing modifier for a symbol.

        Returns AISizingResult with modifier between 0.5 and 1.5.
        On any failure, returns modifier=1.0 (no effect).
        """
        if not self._client:
            return _default_result()

        # Check cache
        cached = await self._get_cached(symbol)
        if cached:
            return cached

        await self._rate_limiter.acquire()

        result = await self._call_api(
            symbol, signal_confidence, strategy_name,
            portfolio_summary, features, sentiment,
        )

        await self._set_cached(symbol, result)
        return result

    # ── API call ──────────────────────────────────────────────────

    async def _call_api(
        self,
        symbol: str,
        signal_confidence: float,
        strategy_name: str,
        portfolio_summary: dict,
        features: dict | None,
        sentiment: dict | None,
    ) -> AISizingResult:
        """Call Claude API for sizing advice."""
        display_symbol = symbol.split(".")[0] if "." in symbol else symbol

        context_parts = [
            f"Symbol: {display_symbol}",
            f"Signal confidence: {signal_confidence:.2f}",
            f"Strategy: {strategy_name}",
            f"Portfolio: {json.dumps(portfolio_summary)}",
        ]
        if features:
            # Include key technical indicators
            key_features = {k: v for k, v in features.items()
                           if any(ind in k.lower() for ind in
                                  ("rsi", "macd", "bb_", "atr", "obv", "vwap", "volume"))}
            if key_features:
                context_parts.append(f"Technicals: {json.dumps(key_features)}")
        if sentiment:
            context_parts.append(f"Sentiment: {json.dumps(sentiment)}")

        context = "\n".join(context_parts)

        prompt = (
            "You are a risk-aware position sizing advisor for a day trading system.\n"
            "Given the context below, recommend a position sizing modifier.\n\n"
            "Rules:\n"
            "- modifier: 0.5 (reduce size 50%) to 1.5 (increase size 50%), 1.0 = no change\n"
            "- Consider: earnings risk, macro environment, technical/sentiment alignment, "
            "portfolio concentration, unusual volume/volatility\n"
            "- Be conservative: prefer modifiers close to 1.0 unless there's a clear reason\n"
            "- reasoning: 1 sentence explaining your recommendation\n"
            "- risk_factors: list of 0-3 key risks (short strings)\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"modifier": 1.0, "reasoning": "...", "risk_factors": ["..."]}\n\n'
            f"Context:\n{context}"
        )

        try:
            response = await self._client.messages.create(
                model=AI_SIZING_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            parsed = json.loads(raw_text)

            modifier = max(0.5, min(1.5, float(parsed.get("modifier", 1.0))))
            reasoning = parsed.get("reasoning", "No reasoning provided")
            risk_factors = parsed.get("risk_factors", [])
            if not isinstance(risk_factors, list):
                risk_factors = []

            logger.info(
                "ai_sizing.result",
                symbol=symbol,
                modifier=modifier,
                reasoning=reasoning,
                risk_factors=risk_factors,
            )

            self.call_count += 1
            self.estimated_cost_usd += 0.0003

            return AISizingResult(
                modifier=modifier,
                reasoning=reasoning,
                risk_factors=risk_factors[:3],
            )

        except json.JSONDecodeError:
            logger.warning("ai_sizing.parse_error", symbol=symbol)
            return _default_result()
        except anthropic.AuthenticationError:
            logger.error("ai_sizing.auth_error")
            self._client = None
            return _default_result()
        except Exception:
            logger.exception("ai_sizing.api_error", symbol=symbol)
            return _default_result()
