"""Tests for multi-timeframe confirmation filter."""


from app.strategy.base import SignalAction, TradingSignal
from app.strategy.multi_timeframe import MultiTimeframeFilter


def _make_signal(
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.6,
    symbol: str = "AAPL",
) -> TradingSignal:
    return TradingSignal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        strategy_name="test",
    )


def _bullish_features() -> dict:
    return {
        "sma_10": 0.02,
        "rsi_14": 60.0,
        "macd_divergence": 0.5,
        "momentum_10d": 0.03,
    }


def _bearish_features() -> dict:
    return {
        "sma_10": -0.02,
        "rsi_14": 35.0,
        "macd_divergence": -0.5,
        "momentum_10d": -0.03,
    }


def _neutral_features() -> dict:
    return {
        "sma_10": 0.0,
        "rsi_14": 50.0,
        "macd_divergence": 0.0,
        "momentum_10d": 0.0,
    }


class TestMultiTimeframeFilter:
    def test_hold_signal_unchanged(self):
        filt = MultiTimeframeFilter()
        signal = _make_signal(action=SignalAction.HOLD, confidence=0.5)
        result = filt.confirm_signal(signal, _bullish_features(), _bullish_features())
        assert result.confidence == 0.5
        assert result.action == SignalAction.HOLD

    def test_buy_aligned_both_bullish(self):
        filt = MultiTimeframeFilter()
        signal = _make_signal(action=SignalAction.BUY, confidence=0.6)
        result = filt.confirm_signal(signal, _bullish_features(), _bullish_features())
        assert result.confidence > 0.6  # Boosted
        assert result.confidence <= 0.95  # Capped at MAX_CONFIDENCE
        assert result.metadata["mtf_alignment"] == "aligned"

    def test_buy_conflicting_both_bearish(self):
        filt = MultiTimeframeFilter()
        signal = _make_signal(action=SignalAction.BUY, confidence=0.6)
        result = filt.confirm_signal(signal, _bearish_features(), _bearish_features())
        assert result.confidence < 0.6  # Penalized
        assert result.metadata["mtf_alignment"] == "conflicting"

    def test_sell_aligned_both_bearish(self):
        filt = MultiTimeframeFilter()
        signal = _make_signal(action=SignalAction.SELL, confidence=0.6)
        result = filt.confirm_signal(signal, _bearish_features(), _bearish_features())
        assert result.confidence > 0.6
        assert result.metadata["mtf_alignment"] == "aligned"

    def test_sell_conflicting_both_bullish(self):
        filt = MultiTimeframeFilter()
        signal = _make_signal(action=SignalAction.SELL, confidence=0.6)
        result = filt.confirm_signal(signal, _bullish_features(), _bullish_features())
        assert result.confidence < 0.6
        assert result.metadata["mtf_alignment"] == "conflicting"

    def test_mixed_alignment(self):
        filt = MultiTimeframeFilter()
        signal = _make_signal(action=SignalAction.BUY, confidence=0.6)
        result = filt.confirm_signal(signal, _bullish_features(), _bearish_features())
        assert result.metadata["mtf_alignment"] == "mixed"
        # Mixed should give a slight penalty (0.9x)
        expected = 0.6 * 0.9
        assert abs(result.confidence - expected) < 0.01

    def test_confidence_capped_at_max(self):
        filt = MultiTimeframeFilter()
        signal = _make_signal(action=SignalAction.BUY, confidence=0.9)
        result = filt.confirm_signal(signal, _bullish_features(), _bullish_features())
        assert result.confidence <= 0.95

    def test_confidence_not_negative(self):
        filt = MultiTimeframeFilter()
        signal = _make_signal(action=SignalAction.BUY, confidence=0.01)
        result = filt.confirm_signal(signal, _bearish_features(), _bearish_features())
        assert result.confidence >= 0.0

    def test_empty_features(self):
        filt = MultiTimeframeFilter()
        signal = _make_signal(action=SignalAction.BUY, confidence=0.6)
        result = filt.confirm_signal(signal, {}, {})
        # With no features, trend is 0 → mixed
        assert result.confidence <= 0.6

    def test_metadata_preserved(self):
        filt = MultiTimeframeFilter()
        signal = TradingSignal(
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=0.6,
            strategy_name="test",
            metadata={"original_key": "value"},
        )
        result = filt.confirm_signal(signal, _bullish_features(), _bullish_features())
        assert result.metadata["original_key"] == "value"
        assert "mtf_alignment" in result.metadata

    def test_assess_trend_bullish(self):
        filt = MultiTimeframeFilter()
        score = filt._assess_trend(_bullish_features())
        assert score > 0

    def test_assess_trend_bearish(self):
        filt = MultiTimeframeFilter()
        score = filt._assess_trend(_bearish_features())
        assert score < 0

    def test_assess_trend_neutral(self):
        filt = MultiTimeframeFilter()
        score = filt._assess_trend(_neutral_features())
        assert score == 0.0
