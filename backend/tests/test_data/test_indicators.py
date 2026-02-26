import numpy as np
import pandas as pd

from app.data.indicators import atr, bollinger_bands, compute_features, macd, rsi


def make_ohlcv(n=100):
    """Create a sample OHLCV DataFrame."""
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


def test_rsi():
    df = make_ohlcv()
    result = rsi(df["close"], 14)
    assert len(result) == len(df)
    # RSI values should be between 0 and 100 (after warmup)
    valid = result.dropna()
    assert valid.min() >= 0
    assert valid.max() <= 100


def test_macd():
    df = make_ohlcv()
    result = macd(df["close"])
    assert "macd" in result
    assert "signal" in result
    assert "histogram" in result
    assert len(result["macd"]) == len(df)


def test_bollinger_bands():
    df = make_ohlcv()
    result = bollinger_bands(df["close"])
    assert "upper" in result
    assert "middle" in result
    assert "lower" in result
    # Upper should be > middle > lower
    valid_idx = result["middle"].dropna().index
    assert (result["upper"][valid_idx] >= result["middle"][valid_idx]).all()
    assert (result["middle"][valid_idx] >= result["lower"][valid_idx]).all()


def test_atr():
    df = make_ohlcv()
    result = atr(df["high"], df["low"], df["close"])
    valid = result.dropna()
    assert len(valid) > 0
    assert (valid > 0).all()


def test_compute_features():
    df = make_ohlcv(200)
    features = compute_features(df)
    assert "sma_10" in features.columns
    assert "rsi_14" in features.columns
    assert "macd" in features.columns
    assert "bb_upper" in features.columns
    assert "atr_14" in features.columns
    assert "obv" in features.columns
    assert "return_1d" in features.columns
    # Interaction features
    assert "bb_squeeze" in features.columns
    assert "rsi_sma_cross" in features.columns
    assert "vwap_distance" in features.columns
    assert "volume_price_trend" in features.columns
    assert "macd_cross" in features.columns
    assert "bb_position" in features.columns
    assert "trend_strength" in features.columns
    # Regime features
    assert "regime_trending" in features.columns
    assert "regime_breadth_proxy" in features.columns
    assert "regime_mean_reversion" in features.columns
    assert len(features) == len(df)
