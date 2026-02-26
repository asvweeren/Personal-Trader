from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.data.market_data import MarketSnapshot
from app.data.sentiment import SentimentResult
from app.strategy.base import SignalAction
from app.strategy.sentiment_strategy import SentimentStrategy


def make_snapshot(symbols: list[str] = None):
    symbols = symbols or ["AAPL"]
    return MarketSnapshot(
        timestamp=datetime.now(timezone.utc),
        prices={s: 150.0 for s in symbols},
        ohlcv={},
        features={},
    )


def make_sentiment(symbol="AAPL", score=0.5, confidence=0.8, news_count=5):
    return SentimentResult(
        symbol=symbol, score=score, confidence=confidence,
        reasoning="Test reasoning", news_count=news_count,
    )


def _setup_strategy_with_cached(strategy, cached_result):
    """Set up the strategy with a mocked cache that returns the given result."""
    strategy._analyzer = MagicMock()
    strategy._analyzer._get_cached = AsyncMock(return_value=cached_result)
    strategy._news_fetcher = MagicMock()


# ── Signal generation tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_bullish_sentiment_gives_buy():
    strategy = SentimentStrategy(buy_threshold=0.3, min_confidence=0.3)
    _setup_strategy_with_cached(strategy, make_sentiment(score=0.5, confidence=0.8, news_count=5))

    signals = await strategy.generate_signals(make_snapshot())
    assert len(signals) == 1
    assert signals[0].action == SignalAction.BUY


@pytest.mark.asyncio
async def test_bearish_sentiment_gives_sell():
    strategy = SentimentStrategy(sell_threshold=-0.3, min_confidence=0.3)
    _setup_strategy_with_cached(strategy, make_sentiment(score=-0.5, confidence=0.8, news_count=5))

    signals = await strategy.generate_signals(make_snapshot())
    assert len(signals) == 1
    assert signals[0].action == SignalAction.SELL


@pytest.mark.asyncio
async def test_neutral_sentiment_gives_hold():
    strategy = SentimentStrategy()
    _setup_strategy_with_cached(strategy, make_sentiment(score=0.0, confidence=0.5, news_count=3))

    signals = await strategy.generate_signals(make_snapshot())
    assert len(signals) == 1
    assert signals[0].action == SignalAction.HOLD


@pytest.mark.asyncio
async def test_low_confidence_gives_hold():
    """Even bullish score should HOLD if confidence is below threshold."""
    strategy = SentimentStrategy(min_confidence=0.7)
    _setup_strategy_with_cached(strategy, make_sentiment(score=0.8, confidence=0.3, news_count=1))

    signals = await strategy.generate_signals(make_snapshot())
    assert len(signals) == 1
    assert signals[0].action == SignalAction.HOLD


@pytest.mark.asyncio
async def test_no_cached_sentiment_skips_symbol():
    """If no cached sentiment is available, the symbol is skipped."""
    strategy = SentimentStrategy()
    _setup_strategy_with_cached(strategy, None)

    signals = await strategy.generate_signals(make_snapshot())
    assert len(signals) == 0


# ── Confidence calibration tests ─────────────────────────────


def test_calibration_many_articles():
    strategy = SentimentStrategy()
    result = make_sentiment(confidence=0.8, news_count=10)
    calibrated = strategy._calibrate_confidence(result)
    # Many articles → volume_factor = 1.0
    assert calibrated == pytest.approx(0.8, abs=0.01)


def test_calibration_few_articles():
    strategy = SentimentStrategy()
    result = make_sentiment(confidence=0.8, news_count=1)
    calibrated = strategy._calibrate_confidence(result)
    # Very few articles → volume_factor = 0.4
    assert calibrated < 0.5


def test_calibration_extreme_score():
    strategy = SentimentStrategy()
    result = make_sentiment(score=0.9, confidence=0.8, news_count=8)
    calibrated = strategy._calibrate_confidence(result)
    # Extreme score → extremity_factor = 0.85
    assert calibrated < 0.8


def test_calibration_moderate_articles():
    strategy = SentimentStrategy()
    result = make_sentiment(confidence=0.8, news_count=4)
    calibrated = strategy._calibrate_confidence(result)
    # Moderate articles → somewhere between 0.4 and 0.8
    assert 0.4 < calibrated < 0.8


# ── Stats tracking tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_symbols_analyzed_tracking():
    strategy = SentimentStrategy(min_confidence=0.3)
    strategy._analyzer = MagicMock()
    # Return cached result for both symbols
    strategy._analyzer._get_cached = AsyncMock(
        return_value=make_sentiment(news_count=5)
    )
    strategy._news_fetcher = MagicMock()

    await strategy.generate_signals(make_snapshot(["AAPL", "MSFT"]))
    stats = strategy.get_stats()
    assert stats["symbols_analyzed"] == 2


# ── Metadata tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_signal_metadata():
    strategy = SentimentStrategy(min_confidence=0.3)
    _setup_strategy_with_cached(strategy, make_sentiment(score=0.4, confidence=0.7, news_count=5))

    signals = await strategy.generate_signals(make_snapshot())
    meta = signals[0].metadata
    assert "sentiment_score" in meta
    assert "raw_confidence" in meta
    assert "calibrated_confidence" in meta
    assert "reasoning" in meta
    assert "news_count" in meta
