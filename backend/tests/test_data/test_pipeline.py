from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.data.pipeline import DataPipeline
from app.data.sentiment import SentimentResult
from app.broker.mock_adapter import MockBrokerAdapter


def make_ohlcv(n=100):
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": close + rng.normal(0, 0.5, n),
        "high": close + abs(rng.normal(0, 1, n)),
        "low": close - abs(rng.normal(0, 1, n)),
        "close": close,
        "volume": rng.integers(1000, 100000, n),
    })


@pytest.fixture
def pipeline():
    broker = MockBrokerAdapter(initial_cash=5000.0)
    p = DataPipeline(broker=broker)
    p._symbols = ["AAPL", "MSFT"]
    return p


@pytest.mark.asyncio
async def test_pipeline_get_status(pipeline):
    status = pipeline.get_status()
    assert status["symbols"] == ["AAPL", "MSFT"]
    assert status["features_computed"] == 0
    assert status["sentiment_computed"] == 0


@pytest.mark.asyncio
async def test_refresh_features_skips_insufficient_data(pipeline):
    """Should skip symbols with < 50 bars of data."""
    # Return a tiny DataFrame
    small_df = make_ohlcv(10)
    pipeline._market_data.get_historical_data = AsyncMock(return_value=small_df)

    # Mock the feature store to not require Redis
    pipeline._feature_store.store_features = AsyncMock()

    features = await pipeline.refresh_features()
    assert len(features) == 0


@pytest.mark.asyncio
async def test_refresh_features_computes_for_valid_data(pipeline):
    """Should compute features for symbols with enough data."""
    df = make_ohlcv(200)
    pipeline._market_data.get_historical_data = AsyncMock(return_value=df)
    pipeline._feature_store.store_features = AsyncMock()

    features = await pipeline.refresh_features()
    assert "AAPL" in features
    assert "MSFT" in features
    assert "sma_10" in features["AAPL"]
    assert "rsi_14" in features["AAPL"]


@pytest.mark.asyncio
async def test_refresh_features_handles_errors(pipeline):
    """Should continue processing other symbols when one fails."""
    call_count = 0

    async def mock_get_data(symbol, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if symbol == "AAPL":
            raise RuntimeError("Broker error")
        return make_ohlcv(200)

    pipeline._market_data.get_historical_data = mock_get_data
    pipeline._feature_store.store_features = AsyncMock()

    features = await pipeline.refresh_features()
    # AAPL should fail, MSFT should succeed
    assert "AAPL" not in features
    assert "MSFT" in features


@pytest.mark.asyncio
async def test_refresh_sentiment(pipeline):
    """Should call sentiment analyzer for each symbol."""
    pipeline._news_fetcher.fetch_symbol_news = AsyncMock(return_value=[])
    pipeline._sentiment.analyze = AsyncMock(
        return_value=SentimentResult(
            symbol="AAPL", score=0.5, confidence=0.8,
            reasoning="Positive outlook", news_count=5,
        )
    )
    pipeline._feature_store.store_sentiment = AsyncMock()

    sentiments = await pipeline.refresh_sentiment()
    assert len(sentiments) == 2
    assert pipeline._news_fetcher.fetch_symbol_news.call_count == 2


@pytest.mark.asyncio
async def test_get_enriched_snapshot_merges_features(pipeline):
    """Snapshot should include cached features and sentiment."""
    # Set up cached features
    pipeline._latest_features["AAPL"] = {"sma_10": 150.0, "rsi_14": 55.0}
    pipeline._latest_sentiment["AAPL"] = SentimentResult(
        symbol="AAPL", score=0.3, confidence=0.7,
        reasoning="Neutral", news_count=3,
    )

    # Mock the snapshot call
    from app.data.market_data import MarketSnapshot
    from datetime import datetime, timezone

    mock_snapshot = MarketSnapshot(
        timestamp=datetime.now(timezone.utc),
        prices={"AAPL": 150.0, "MSFT": 380.0},
        ohlcv={},
        features={},
    )
    pipeline._market_data.get_snapshot = AsyncMock(return_value=mock_snapshot)
    pipeline._feature_store.get_features = AsyncMock(return_value=None)
    pipeline._feature_store.get_sentiment = AsyncMock(return_value=None)

    snapshot = await pipeline.get_enriched_snapshot()
    assert "AAPL" in snapshot.features
    assert snapshot.features["AAPL"]["sma_10"] == 150.0
    assert snapshot.features["AAPL"]["sentiment_score"] == 0.3
