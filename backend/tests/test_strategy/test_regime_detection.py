"""Tests for market regime detection."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from app.data.market_data import MarketSnapshot
from app.strategy.regime import RegimeDetector, MarketRegime, RegimeState


def _make_snapshot(trend: str = "up", volatility: str = "normal", n_symbols: int = 5) -> MarketSnapshot:
    """Create a synthetic market snapshot for testing regime detection."""
    ohlcv = {}
    prices = {}

    for i in range(n_symbols):
        symbol = f"SYM{i}"
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")

        if trend == "up":
            close = 100 + np.cumsum(np.random.normal(0.1, 0.5, n))
        elif trend == "down":
            close = 100 + np.cumsum(np.random.normal(-0.1, 0.5, n))
        else:
            close = 100 + np.cumsum(np.random.normal(0, 0.3, n))

        if volatility == "high":
            close = 100 + np.cumsum(np.random.normal(0, 2.0, n))
        elif volatility == "low":
            close = 100 + np.cumsum(np.random.normal(0, 0.1, n))

        close = np.maximum(close, 10)  # Ensure positive

        df = pd.DataFrame({
            "timestamp": dates,
            "open": close * (1 + np.random.normal(0, 0.005, n)),
            "high": close * (1 + abs(np.random.normal(0, 0.01, n))),
            "low": close * (1 - abs(np.random.normal(0, 0.01, n))),
            "close": close,
            "volume": np.random.randint(100000, 1000000, n),
        })
        ohlcv[symbol] = df
        prices[symbol] = float(close[-1])

    return MarketSnapshot(
        timestamp=datetime.now(timezone.utc),
        prices=prices,
        ohlcv=ohlcv,
        features={},
    )


class TestRegimeDetector:
    def test_detect_returns_regime_state(self):
        detector = RegimeDetector()
        snapshot = _make_snapshot()
        state = detector.detect(snapshot)
        assert isinstance(state, RegimeState)
        assert isinstance(state.regime, MarketRegime)
        assert 0 <= state.confidence <= 1
        assert state.adx >= 0

    def test_detect_stores_last_regime(self):
        detector = RegimeDetector()
        assert detector.current_regime is None
        snapshot = _make_snapshot()
        state = detector.detect(snapshot)
        assert detector.current_regime is state

    def test_regime_to_dict(self):
        state = RegimeState(
            regime=MarketRegime.TRENDING_UP,
            confidence=0.8,
            adx=30.0,
            volatility_ratio=1.0,
            mean_reversion_score=0.3,
            breadth=0.7,
        )
        d = state.to_dict()
        assert d["regime"] == "trending_up"
        assert d["confidence"] == 0.8

    def test_empty_snapshot(self):
        detector = RegimeDetector()
        snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            prices={},
            ohlcv={},
            features={},
        )
        state = detector.detect(snapshot)
        assert state.regime == MarketRegime.RANGING

    def test_classify_high_volatility(self):
        detector = RegimeDetector()
        regime, conf = detector._classify(adx=20, vol_ratio=2.0, breadth=0.5, mean_reversion=0.0)
        assert regime == MarketRegime.HIGH_VOLATILITY

    def test_classify_low_volatility(self):
        detector = RegimeDetector()
        regime, conf = detector._classify(adx=20, vol_ratio=0.5, breadth=0.5, mean_reversion=0.0)
        assert regime == MarketRegime.LOW_VOLATILITY

    def test_classify_trending_up(self):
        detector = RegimeDetector()
        regime, conf = detector._classify(adx=35, vol_ratio=1.0, breadth=0.8, mean_reversion=0.3)
        assert regime == MarketRegime.TRENDING_UP

    def test_classify_trending_down(self):
        detector = RegimeDetector()
        regime, conf = detector._classify(adx=35, vol_ratio=1.0, breadth=0.2, mean_reversion=0.3)
        assert regime == MarketRegime.TRENDING_DOWN

    def test_classify_ranging(self):
        detector = RegimeDetector()
        regime, conf = detector._classify(adx=15, vol_ratio=1.0, breadth=0.5, mean_reversion=0.0)
        assert regime == MarketRegime.RANGING
