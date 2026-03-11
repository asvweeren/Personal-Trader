"""Trading strategy based on LLM sentiment analysis of news.

Uses Claude API to analyze financial news and generate trading signals.
Includes cost tracking and confidence calibration.
"""


import structlog

from app.data.market_data import MarketSnapshot
from app.data.news_fetcher import NewsFetcher
from app.data.sentiment import SentimentAnalyzer, SentimentResult
from app.strategy.base import SignalAction, Strategy, TradingSignal

logger = structlog.get_logger()


class SentimentStrategy(Strategy):
    """Trading strategy based on LLM sentiment analysis of news.

    Features:
    - Configurable buy/sell thresholds
    - Minimum confidence filter
    - Cost tracking for API usage
    - Confidence calibration based on news volume
    - Sentiment score + reasoning included in signal metadata

    Signal mapping:
    - score > buy_threshold  -> BUY  (confidence = abs(score))
    - score < sell_threshold -> SELL (confidence = abs(score))
    - else                   -> HOLD
    """

    @property
    def name(self) -> str:
        return "sentiment"

    def __init__(
        self,
        buy_threshold: float = 0.3,
        sell_threshold: float = -0.3,
        min_confidence: float = 0.35,
        min_news_count: int = 2,
    ):
        self._news_fetcher = NewsFetcher()
        self._analyzer = SentimentAnalyzer()
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold
        self._min_confidence = min_confidence
        self._min_news_count = min_news_count
        # Tracking
        self._total_symbols_analyzed = 0

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        """Generate trading signals from cached sentiment data.

        Uses pre-computed sentiment from the pipeline (cached in Redis).
        Does NOT make its own API calls — all API calls are centralized
        in pipeline.refresh_sentiment() to control costs.
        """
        signals: list[TradingSignal] = []
        for symbol in market_data.prices:
            try:
                # Use cached sentiment only — no direct API calls
                cached = await self._analyzer._get_cached(symbol)
                if not cached:
                    continue

                self._total_symbols_analyzed += 1

                # Calibrate confidence based on news volume
                calibrated_confidence = self._calibrate_confidence(cached)

                if calibrated_confidence < self._min_confidence:
                    action = SignalAction.HOLD
                elif cached.score >= self._buy_threshold:
                    action = SignalAction.BUY
                elif cached.score <= self._sell_threshold:
                    action = SignalAction.SELL
                else:
                    action = SignalAction.HOLD

                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        action=action,
                        confidence=calibrated_confidence,
                        strategy_name=self.name,
                        metadata={
                            "sentiment_score": cached.score,
                            "raw_confidence": cached.confidence,
                            "calibrated_confidence": calibrated_confidence,
                            "reasoning": cached.reasoning,
                            "news_count": cached.news_count,
                            "headlines_analyzed": cached.headlines_analyzed or [],
                            "analyzed_at": (
                                cached.timestamp.isoformat()
                                if cached.timestamp
                                else None
                            ),
                        },
                    )
                )
            except Exception:
                logger.exception("sentiment_strategy.error", symbol=symbol)

        return signals

    def _calibrate_confidence(self, sentiment: SentimentResult) -> float:
        """Adjust confidence based on news volume and score strength.

        More news articles -> higher confidence in the analysis.
        Extreme scores -> slightly lower confidence (might be noise).
        """
        base = sentiment.confidence

        # Volume boost: more news = more reliable (gentler curve)
        if sentiment.news_count >= 5:
            volume_factor = 1.0
        elif sentiment.news_count >= self._min_news_count:
            volume_factor = 0.8 + (sentiment.news_count / 5) * 0.2
        else:
            volume_factor = 0.6  # Few articles, moderate reliability

        # Extreme scores are strong conviction, not noise — no penalty
        extremity_factor = 1.0

        calibrated = base * volume_factor * extremity_factor
        return max(0.0, min(1.0, calibrated))

    async def close(self) -> None:
        """Clean up resources."""
        await self._news_fetcher.close()
        await self._analyzer.close()

    def get_stats(self) -> dict:
        """Return strategy usage statistics."""
        return {
            "symbols_analyzed": self._total_symbols_analyzed,
            "buy_threshold": self._buy_threshold,
            "sell_threshold": self._sell_threshold,
            "min_confidence": self._min_confidence,
        }
