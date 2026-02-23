import json
from datetime import datetime, timezone

import redis.asyncio as redis
import structlog

from app.config import settings

logger = structlog.get_logger()


class FeatureStore:
    """Caches computed features in Redis for fast access."""

    def __init__(self):
        self._redis: redis.Redis | None = None
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    async def connect(self) -> None:
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.close()

    async def store_features(self, symbol: str, features: dict, ttl: int = 300) -> None:
        if not self._redis:
            return
        key = f"features:{symbol}"
        data = {
            "features": features,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._redis.setex(key, ttl, json.dumps(data, default=str))

    async def get_features(self, symbol: str) -> dict | None:
        if not self._redis:
            return None
        key = f"features:{symbol}"
        data = await self._redis.get(key)
        if data:
            self._cache_hits += 1
            return json.loads(data).get("features")
        self._cache_misses += 1
        return None

    async def get_features_batch(self, symbols: list[str]) -> dict[str, dict | None]:
        """Batch fetch features for multiple symbols using Redis MGET."""
        if not self._redis or not symbols:
            return {s: None for s in symbols}
        keys = [f"features:{s}" for s in symbols]
        values = await self._redis.mget(keys)
        result: dict[str, dict | None] = {}
        for symbol, val in zip(symbols, values):
            if val:
                self._cache_hits += 1
                result[symbol] = json.loads(val).get("features")
            else:
                self._cache_misses += 1
                result[symbol] = None
        return result

    async def store_sentiment(self, symbol: str, sentiment: dict, ttl: int = 600) -> None:
        if not self._redis:
            return
        key = f"sentiment:{symbol}"
        await self._redis.setex(key, ttl, json.dumps(sentiment, default=str))

    async def get_sentiment(self, symbol: str) -> dict | None:
        if not self._redis:
            return None
        key = f"sentiment:{symbol}"
        data = await self._redis.get(key)
        if data:
            self._cache_hits += 1
            return json.loads(data)
        self._cache_misses += 1
        return None

    async def clear_symbol(self, symbol: str) -> None:
        """Remove all cached data for a symbol."""
        if not self._redis:
            return
        await self._redis.delete(f"features:{symbol}", f"sentiment:{symbol}")

    async def clear_all(self) -> None:
        """Remove all feature and sentiment cache entries."""
        if not self._redis:
            return
        keys = []
        async for key in self._redis.scan_iter("features:*"):
            keys.append(key)
        async for key in self._redis.scan_iter("sentiment:*"):
            keys.append(key)
        if keys:
            await self._redis.delete(*keys)

    def get_cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate_pct": round(hit_rate, 1),
        }
