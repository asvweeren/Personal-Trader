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


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX) for trend strength measurement."""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Smoothed averages
    atr_smooth = true_range.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_smooth
    minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_smooth

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.ewm(com=period - 1, min_periods=period).mean()

    return adx_val


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

    # ADX (trend strength)
    features["adx_14"] = adx(high, low, close, 14)

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

    # --- Alternative Data Proxies ---

    # Put/call proxy: bearish signal when RSI < 30 with high volume
    rsi_val = features["rsi_14"]
    vol_ratio = features["volume_ratio"]
    features["put_call_proxy"] = np.where(
        (rsi_val < 30) & (vol_ratio > 1.5), -1.0,
        np.where((rsi_val > 70) & (vol_ratio > 1.5), 1.0, 0.0),
    )

    # Volatility regime: 20-day realized vol vs 60-day
    vol_60d = close.pct_change().rolling(60).std()
    features["volatility_regime"] = features["volatility_20d"] / vol_60d

    # --- Interaction Features ---

    # BB squeeze: BB width in lowest 20% of rolling 20-bar range
    bb_width = features["bb_width"]
    bb_width_min = bb_width.rolling(20).min()
    bb_width_max = bb_width.rolling(20).max()
    bb_width_pctile = (bb_width - bb_width_min) / (bb_width_max - bb_width_min + 1e-10)
    features["bb_squeeze"] = (bb_width_pctile < 0.2).astype(float)

    # RSI-SMA cross: RSI minus its own 14-period SMA
    features["rsi_sma_cross"] = rsi_val - sma(rsi_val, 14)

    # VWAP distance: price distance from VWAP as %
    features["vwap_distance"] = (close - features["vwap"]) / features["vwap"]

    # Volume-price trend: OBV normalized by price
    features["volume_price_trend"] = features["obv"] / close

    # ATR ratio: ATR/close (normalized volatility)
    features["atr_ratio"] = features["atr_14"] / close

    # MACD cross: 1=bullish cross, -1=bearish, 0=none
    macd_diff = features["macd"] - features["macd_signal"]
    macd_diff_prev = macd_diff.shift(1)
    features["macd_cross"] = np.where(
        (macd_diff > 0) & (macd_diff_prev <= 0), 1.0,
        np.where((macd_diff < 0) & (macd_diff_prev >= 0), -1.0, 0.0),
    )

    # BB position: position within Bollinger Bands (0=lower, 1=upper)
    bb_range = bb["upper"] - bb["lower"]
    features["bb_position"] = np.where(bb_range > 0, (close - bb["lower"]) / bb_range, 0.5)

    # Trend strength: ADX × sign(price_vs_sma50)
    features["trend_strength"] = features["adx_14"] * np.sign(features["price_vs_sma50"])

    # --- Regime Features ---

    features["regime_trending"] = (features["adx_14"] > 25.0).astype(float)
    features["regime_breadth_proxy"] = (features["price_vs_sma50"] > 0).astype(float)
    features["regime_mean_reversion"] = (features["rsi_14"] - 50).abs() / 50

    # --- Additional Quantitative Features ---

    # Volatility skew: downside vol / upside vol (>1 = more downside risk)
    returns = close.pct_change()
    downside_vol = returns.where(returns < 0, 0.0).rolling(20).std()
    upside_vol = returns.where(returns > 0, 0.0).rolling(20).std()
    features["volatility_skew"] = downside_vol / (upside_vol + 1e-10)

    # Vol-of-vol: volatility of volatility (regime change detector)
    features["vol_of_vol"] = features["volatility_20d"].rolling(30).std()

    # OBV momentum: rate of change of OBV
    obv_series = features["obv"]
    features["obv_momentum"] = obv_series / (obv_series.shift(5) + 1e-10)
    features["obv_divergence"] = obv_series - obv_series.rolling(20).mean()

    return features
