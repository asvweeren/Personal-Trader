"""Data pipeline orchestrator - coordinates data collection, feature computation, and caching."""

import asyncio

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import BrokerAdapter
from app.data.feature_store import FeatureStore
from app.data.indicators import compute_features
from app.data.market_data import MarketDataService, MarketSnapshot
from app.data.news_fetcher import NewsFetcher
from app.data.sentiment import SentimentAnalyzer, SentimentResult
from app.risk.market_hours import Exchange, is_market_open

logger = structlog.get_logger()

# Max symbols to analyze sentiment for (cost control)
MAX_SENTIMENT_SYMBOLS = 15


class DataPipeline:
    """Orchestrates the full data pipeline:
    1. Collect market data (prices, OHLCV)
    2. Compute technical indicators
    3. Fetch news and analyze sentiment
    4. Store computed features in Redis for fast access
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        db: AsyncSession | None = None,
        feature_store: FeatureStore | None = None,
    ):
        self._market_data = MarketDataService(broker, db=db)
        self._news_fetcher = NewsFetcher()
        self._sentiment = SentimentAnalyzer()
        self._feature_store = feature_store or FeatureStore()
        self._symbols: list[str] = []
        self._latest_features: dict[str, dict] = {}
        self._latest_sentiment: dict[str, SentimentResult] = {}

    @property
    def market_data(self) -> MarketDataService:
        return self._market_data

    @property
    def feature_store(self) -> FeatureStore:
        return self._feature_store

    async def start(self, symbols: list[str]) -> None:
        """Initialize the pipeline and start streaming."""
        self._symbols = symbols
        await self._feature_store.connect()
        await self._market_data.start_streaming(symbols)
        logger.info("pipeline.started", symbols=symbols)

    async def update_symbols(self, symbols: list[str]) -> None:
        """Update the active symbol list and restart streaming."""
        old = self._symbols
        # Stop streaming old symbols that are no longer in the list
        removed = [s for s in old if s not in symbols]
        if removed:
            await self._market_data.stop_streaming(removed)
        # Start streaming new symbols
        added = [s for s in symbols if s not in old]
        if added:
            await self._market_data.start_streaming(added)
        self._symbols = list(symbols)
        # Clear stale feature/sentiment caches for removed symbols
        for s in removed:
            self._latest_features.pop(s, None)
            self._latest_sentiment.pop(s, None)
        logger.info(
            "pipeline.symbols_updated",
            old=len(old), new=len(symbols),
            added=len(added), removed=len(removed),
        )

    async def stop(self) -> None:
        """Stop the pipeline and clean up."""
        await self._market_data.stop_streaming(self._symbols)
        await self._news_fetcher.close()
        await self._sentiment.close()
        await self._feature_store.disconnect()
        logger.info("pipeline.stopped")

    # ── Scheduled jobs ────────────────────────────────────────────

    async def refresh_features(self) -> dict[str, dict]:
        """Compute technical indicators for all symbols and cache them.
        Intended to be called periodically (e.g., every 5 minutes).
        Skips symbols whose exchange is currently closed.
        """
        from app.risk.market_hours import get_exchange_for_symbol, is_market_open
        open_symbols = [
            s for s in self._symbols
            if is_market_open(get_exchange_for_symbol(s))
        ]
        logger.info("pipeline.refresh_features", symbols=len(self._symbols), open=len(open_symbols))
        features = {}

        for symbol in open_symbols:
            try:
                df = await self._market_data.get_historical_data(
                    symbol, duration="1 Y", bar_size="1 day",
                )
                # Drop today's incomplete bar to match training data (completed bars only)
                df = self._market_data._drop_incomplete_daily_bar(df)
                if df.empty or len(df) < 50:
                    logger.debug(
                        "pipeline.skip_features",
                        symbol=symbol,
                        reason="insufficient_data",
                    )
                    continue

                features_df = compute_features(df)
                latest_row = features_df.iloc[-1]
                feature_dict = {
                    k: float(v) if not (isinstance(v, float) and v != v) else None  # NaN -> None
                    for k, v in latest_row.to_dict().items()
                    if k not in ("timestamp",)
                }

                features[symbol] = feature_dict
                self._latest_features[symbol] = feature_dict

                # Cache in Redis
                await self._feature_store.store_features(symbol, feature_dict, ttl=600)

            except Exception:
                logger.exception("pipeline.feature_error", symbol=symbol)

        logger.info("pipeline.features_computed", count=len(features))
        return features

    async def refresh_sentiment(self) -> dict[str, SentimentResult]:
        """Fetch news and compute sentiment for top symbols during market hours only.

        Cost optimizations:
        - Only runs when at least one market (EU/US) is open
        - Limits analysis to MAX_SENTIMENT_SYMBOLS (top screener picks)
        - Results cached in Redis with 30-min TTL
        """
        # Skip sentiment outside market hours to save API costs
        any_market_open = (
            is_market_open(Exchange.NYSE)
            or is_market_open(Exchange.EURONEXT)
            or is_market_open(Exchange.LSE)
            or is_market_open(Exchange.XETRA)
        )
        if not any_market_open:
            logger.debug("pipeline.sentiment_skipped", reason="markets_closed")
            return {}

        # Limit to top N symbols to control API costs
        symbols_to_analyze = self._symbols[:MAX_SENTIMENT_SYMBOLS]
        logger.info(
            "pipeline.refresh_sentiment",
            symbols=len(symbols_to_analyze),
            total=len(self._symbols),
        )
        sentiments = {}
        sem = asyncio.Semaphore(5)  # Limit concurrent API calls

        async def _analyze_one(symbol: str):
            async with sem:
                news = await self._news_fetcher.fetch_symbol_news(symbol, limit=10)
                result = await self._sentiment.analyze(symbol, news)
                self._latest_sentiment[symbol] = result
                await self._feature_store.store_sentiment(
                    symbol, self._sentiment.to_dict(result), ttl=1800
                )
                return symbol, result

        results = await asyncio.gather(
            *[_analyze_one(s) for s in symbols_to_analyze],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.exception("pipeline.sentiment_error", error=str(result))
                continue
            symbol, sentiment = result
            sentiments[symbol] = sentiment

        logger.info("pipeline.sentiment_computed", count=len(sentiments))
        return sentiments

    async def refresh_historical_data(self) -> None:
        """Download and store latest historical data for all symbols.
        Intended to be called once per day or on startup.
        """
        logger.info("pipeline.refresh_historical", symbols=len(self._symbols))
        sem = asyncio.Semaphore(5)  # Limit concurrent broker calls

        async def _download_one(symbol: str):
            async with sem:
                count = await self._market_data.download_and_store(
                    symbol, duration="1 Y", bar_size="1 day"
                )
                logger.info("pipeline.historical_stored", symbol=symbol, bars=count)

        results = await asyncio.gather(
            *[_download_one(s) for s in self._symbols],
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.exception(
                    "pipeline.historical_error",
                    symbol=self._symbols[i],
                    error=str(result),
                )

    # ── Snapshot with enriched features ───────────────────────────

    async def get_enriched_snapshot(self) -> MarketSnapshot:
        """Get a market snapshot enriched with pre-computed features."""
        snapshot = await self._market_data.get_snapshot(self._symbols)

        # Merge cached features into snapshot
        for symbol in self._symbols:
            features = self._latest_features.get(symbol)
            if not features:
                features = await self._feature_store.get_features(symbol)
            if features:
                snapshot.features[symbol] = features

            # Add sentiment as a feature
            sentiment = self._latest_sentiment.get(symbol)
            if not sentiment:
                cached = await self._feature_store.get_sentiment(symbol)
                if cached:
                    snapshot.features.setdefault(symbol, {})
                    snapshot.features[symbol]["sentiment_score"] = (
                        cached.get("score", 0.0)
                    )
                    snapshot.features[symbol]["sentiment_confidence"] = (
                        cached.get("confidence", 0.0)
                    )
            elif sentiment:
                snapshot.features.setdefault(symbol, {})
                snapshot.features[symbol]["sentiment_score"] = sentiment.score
                snapshot.features[symbol]["sentiment_confidence"] = sentiment.confidence

        return snapshot

    # ── Status ────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get current pipeline status."""
        return {
            "symbols": self._symbols,
            "streaming": self._market_data._streaming,
            "features_computed": len(self._latest_features),
            "sentiment_computed": len(self._latest_sentiment),
            "prices": {
                s: self._market_data._current_prices.get(s, 0.0)
                for s in self._symbols
            },
        }
