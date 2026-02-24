"""Tests for alternative data features in indicators."""

import pandas as pd
import numpy as np

from app.data.indicators import compute_features


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.normal(0.01, 0.5, n))
    close = np.maximum(close, 10)
    return pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.002, n)),
        "high": close * (1 + abs(np.random.normal(0, 0.005, n))),
        "low": close * (1 - abs(np.random.normal(0, 0.005, n))),
        "close": close,
        "volume": np.random.randint(100000, 500000, n).astype(float),
    })


class TestAlternativeDataFeatures:
    def test_compute_features_includes_adx(self):
        df = _make_ohlcv()
        features = compute_features(df)
        assert "adx_14" in features.columns

    def test_compute_features_includes_put_call_proxy(self):
        df = _make_ohlcv()
        features = compute_features(df)
        assert "put_call_proxy" in features.columns
        # Values should be -1, 0, or 1
        unique_vals = set(features["put_call_proxy"].dropna().unique())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_compute_features_includes_volatility_regime(self):
        df = _make_ohlcv()
        features = compute_features(df)
        assert "volatility_regime" in features.columns

    def test_adx_values_reasonable(self):
        df = _make_ohlcv(n=200)
        features = compute_features(df)
        adx_vals = features["adx_14"].dropna()
        assert len(adx_vals) > 0
        # ADX should be between 0 and 100
        assert adx_vals.min() >= 0
        assert adx_vals.max() <= 100

    def test_volatility_regime_ratio(self):
        df = _make_ohlcv(n=200)
        features = compute_features(df)
        vol_regime = features["volatility_regime"].dropna()
        # Should be a ratio > 0
        assert all(vol_regime[vol_regime.notna()] > 0)

    def test_sma_ratio_features(self):
        df = _make_ohlcv()
        features = compute_features(df)
        assert "sma_50" in features.columns
        assert "sma_10" in features.columns

    def test_momentum_features(self):
        df = _make_ohlcv()
        features = compute_features(df)
        assert "momentum_10d" in features.columns
        assert "momentum_20d" in features.columns

    def test_volume_features(self):
        df = _make_ohlcv()
        features = compute_features(df)
        assert "volume_ratio" in features.columns
        assert "volume_change_5d" in features.columns
