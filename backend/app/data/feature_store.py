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
            return json.loads(data).get("features")
        return None

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
        return json.loads(data) if data else None
