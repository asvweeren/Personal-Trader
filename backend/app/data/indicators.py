import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> dict[str, pd.Series]:
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    return {
        "upper": middle + std_dev * std,
        "middle": middle,
        "lower": middle - std_dev * std,
    }


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(com=period - 1, min_periods=period).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators for a DataFrame with OHLCV columns."""
    features = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # Moving averages
    features["sma_10"] = sma(close, 10)
    features["sma_20"] = sma(close, 20)
    features["sma_50"] = sma(close, 50)
    features["ema_10"] = ema(close, 10)
    features["ema_20"] = ema(close, 20)

    # RSI
    features["rsi_14"] = rsi(close, 14)

    # MACD
    macd_data = macd(close)
    features["macd"] = macd_data["macd"]
    features["macd_signal"] = macd_data["signal"]
    features["macd_histogram"] = macd_data["histogram"]

    # Bollinger Bands
    bb = bollinger_bands(close)
    features["bb_upper"] = bb["upper"]
    features["bb_middle"] = bb["middle"]
    features["bb_lower"] = bb["lower"]
    features["bb_width"] = (bb["upper"] - bb["lower"]) / bb["middle"]

    # ATR
    features["atr_14"] = atr(high, low, close, 14)

    # Volume
    features["obv"] = obv(close, vol)
    features["vwap"] = vwap(high, low, close, vol)
    features["volume_sma_20"] = sma(vol, 20)

    # Price features
    features["return_1d"] = close.pct_change(1)
    features["return_5d"] = close.pct_change(5)
    features["volatility_20d"] = close.pct_change().rolling(20).std()

    # Price position relative to moving averages
    features["price_vs_sma50"] = close / features["sma_50"] - 1
    features["price_vs_sma10"] = close / features["sma_10"] - 1

    # Momentum
    features["momentum_10d"] = close / close.shift(10) - 1
    features["momentum_20d"] = close / close.shift(20) - 1

    # Volume dynamics
    features["volume_change_5d"] = vol / vol.shift(5) - 1
    features["volume_ratio"] = vol / vol.rolling(20).mean()

    # Volatility ratio (short vs long)
    vol_10 = close.pct_change().rolling(10).std()
    features["vol_ratio_10_20"] = vol_10 / features["volatility_20d"]

    # RSI momentum
    features["rsi_change_5d"] = features["rsi_14"] - features["rsi_14"].shift(5)

    # MACD divergence
    features["macd_divergence"] = features["macd"] - features["macd_signal"]

    return features
