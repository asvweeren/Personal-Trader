import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.data.sentiment import RateLimiter, SentimentAnalyzer, SentimentResult
from app.data.news_fetcher import NewsItem


# ── RateLimiter tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limiter_allows_within_limit():
    limiter = RateLimiter(max_calls=5, period_seconds=60)
    # Should not block for first 5 calls
    for _ in range(5):
        await limiter.acquire()
    assert len(limiter._calls) == 5


@pytest.mark.asyncio
async def test_rate_limiter_tracks_calls():
    limiter = RateLimiter(max_calls=3, period_seconds=60)
    await limiter.acquire()
    await limiter.acquire()
    assert len(limiter._calls) == 2


# ── SentimentAnalyzer tests ──────────────────────────────────


def _make_analyzer():
    """Create a SentimentAnalyzer bypassing __init__ with correct attributes."""
    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
    analyzer._client = MagicMock()
    analyzer._rate_limiter = RateLimiter(max_calls=10, period_seconds=60)
    analyzer._memory_cache = {}
    analyzer._cache_ttl = 600
    analyzer._redis = None
    analyzer._api_key = "test-key"
    return analyzer


@pytest.mark.asyncio
async def test_sentiment_no_news_returns_neutral():
    """No news should return a neutral score with zero confidence."""
    with patch("app.data.sentiment.anthropic"):
        analyzer = _make_analyzer()

    result = await analyzer.analyze("AAPL", [])
    assert result.score == 0.0
    assert result.confidence == 0.0
    assert result.news_count == 0


@pytest.mark.asyncio
async def test_sentiment_cache_hit():
    """Second call should return cached result from in-memory cache."""
    with patch("app.data.sentiment.anthropic"):
        analyzer = _make_analyzer()

    cached_result = SentimentResult(
        symbol="AAPL", score=0.5, confidence=0.8,
        reasoning="Bullish", news_count=5,
    )
    analyzer._memory_cache["AAPL"] = (cached_result, time.monotonic())

    result = await analyzer.analyze("AAPL", [
        NewsItem(title="Test", description="Test", source="Test", url="http://test.com"),
    ])
    assert result is cached_result


@pytest.mark.asyncio
async def test_sentiment_cache_expired():
    """Expired cache should trigger fresh API call."""
    with patch("app.data.sentiment.anthropic"):
        analyzer = _make_analyzer()

    # Cache with expired timestamp (1000 seconds ago)
    old_result = SentimentResult(
        symbol="AAPL", score=0.5, confidence=0.8,
        reasoning="Old", news_count=3,
    )
    analyzer._memory_cache["AAPL"] = (old_result, time.monotonic() - 1000)

    # Mock the API call
    new_result = SentimentResult(
        symbol="AAPL", score=-0.2, confidence=0.6,
        reasoning="Bearish", news_count=5,
    )
    analyzer._call_api = AsyncMock(return_value=new_result)

    news = [NewsItem(title="Test", description="Test", source="Test", url="http://test.com")]
    result = await analyzer.analyze("AAPL", news)
    assert result.score == -0.2
    assert result.reasoning == "Bearish"


def test_sentiment_to_dict():
    with patch("app.data.sentiment.anthropic"):
        analyzer = _make_analyzer()

    result = SentimentResult(
        symbol="AAPL", score=0.3, confidence=0.7,
        reasoning="Somewhat positive", news_count=5,
    )
    d = analyzer.to_dict(result)
    assert d["symbol"] == "AAPL"
    assert d["score"] == 0.3
    assert d["confidence"] == 0.7
    assert d["reasoning"] == "Somewhat positive"
    assert d["news_count"] == 5


def test_sentiment_clear_cache():
    with patch("app.data.sentiment.anthropic"):
        analyzer = _make_analyzer()
        analyzer._memory_cache["AAPL"] = ("result", time.monotonic())

    analyzer.clear_cache()
    assert len(analyzer._memory_cache) == 0
