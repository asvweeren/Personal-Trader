"""Intraday momentum strategy for short-term trading.

Buys stocks showing strong intraday momentum with volume confirmation.
Designed for day trading with tight stops and quick exits.

Entry rules (ALL must be true):
  1. Price above VWAP (buying strength)
  2. RSI between 50-70 (momentum but not overbought)
  3. Volume > 1.5x 20-day average (institutional interest)
  4. MACD histogram positive and rising (momentum accelerating)
  5. Price above EMA10 (short-term trend up)

Exit rules:
  - Take profit at 1.5-3% (ATR-based)
  - Stop loss at 1-2% (ATR-based)
  - EOD close (never hold overnight)
"""

import structlog

from app.data.indicators import compute_features
from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction, Strategy, TradingSignal

logger = structlog.get_logger()


class MomentumStrategy(Strategy):
    """Intraday momentum strategy using price action + volume confirmation."""

    @property
    def name(self) -> str:
        return "momentum"

    def __init__(self, confidence_threshold: float = 0.60):
        self._confidence_threshold = confidence_threshold

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        signals = []

        for symbol, df in market_data.ohlcv.items():
            if df.empty or len(df) < 50:
                continue

            try:
                # Use pre-computed features if available
                features_df = market_data.computed_features_df.get(symbol)
                if features_df is None:
                    features_df = compute_features(df)

                if features_df.empty or len(features_df) < 2:
                    continue

                latest = features_df.iloc[-1]
                prev = features_df.iloc[-2]

                price = float(df["close"].iloc[-1])
                if price <= 0:
                    continue

                # Extract indicators
                rsi = float(latest.get("rsi_14", 50))
                macd_hist = float(latest.get("macd_histogram", 0))
                macd_hist_prev = float(prev.get("macd_histogram", 0))
                vwap = float(latest.get("vwap", 0))
                ema10 = float(latest.get("ema_10", 0))
                sma_20 = float(latest.get("sma_20", 0))
                adx = float(latest.get("adx_14", 0))
                bb_width = float(latest.get("bb_width", 0))
                vol_ratio = float(latest.get("vol_ratio_10_20", 1.0))
                momentum_1d = float(latest.get("return_1d", 0))

                # --- BUY SIGNALS: Momentum breakout ---
                score = 0.0
                reasons = []

                # 1. Price above VWAP (buying pressure)
                if vwap > 0 and price > vwap:
                    score += 0.20
                    reasons.append("above_vwap")

                # 2. RSI in momentum zone (50-70)
                if 50 <= rsi <= 70:
                    score += 0.20
                    reasons.append(f"rsi_momentum={rsi:.0f}")
                elif 45 <= rsi < 50:
                    score += 0.10  # Partial credit

                # 3. Volume surge (above average)
                if vol_ratio > 1.3:
                    score += 0.20
                    reasons.append(f"volume_surge={vol_ratio:.1f}x")
                elif vol_ratio > 1.0:
                    score += 0.10

                # 4. MACD momentum accelerating
                if macd_hist > 0 and macd_hist > macd_hist_prev:
                    score += 0.20
                    reasons.append("macd_accelerating")
                elif macd_hist > 0:
                    score += 0.10

                # 5. Price above EMA10 (short-term uptrend)
                if ema10 > 0 and price > ema10:
                    score += 0.10
                    reasons.append("above_ema10")

                # 6. ADX shows trending market (bonus)
                if adx > 25:
                    score += 0.10
                    reasons.append(f"trending_adx={adx:.0f}")

                # 7. Positive intraday momentum (bonus)
                if momentum_1d > 0.005:
                    score += 0.10
                    reasons.append(f"intraday_up={momentum_1d:.2%}")

                # Cap at 1.0
                confidence = min(score, 1.0)

                # --- SELL SIGNALS: Momentum reversal ---
                sell_score = 0.0
                if vwap > 0 and price < vwap:
                    sell_score += 0.25
                if rsi > 75:
                    sell_score += 0.25
                if macd_hist < 0 and macd_hist < macd_hist_prev:
                    sell_score += 0.25
                if ema10 > 0 and price < ema10:
                    sell_score += 0.25

                # Determine action
                if confidence >= self._confidence_threshold and confidence > sell_score:
                    action = SignalAction.BUY
                    final_confidence = confidence
                elif sell_score >= 0.60:
                    action = SignalAction.SELL
                    final_confidence = min(sell_score, 1.0)
                else:
                    action = SignalAction.HOLD
                    final_confidence = max(confidence, sell_score)

                feature_snapshot = {
                    "rsi_14": rsi,
                    "macd_histogram": macd_hist,
                    "vwap": vwap,
                    "ema_10": ema10,
                    "adx_14": adx,
                    "vol_ratio": vol_ratio,
                    "momentum_1d": momentum_1d,
                    "bb_width": bb_width,
                    "atr_14": float(latest.get("atr_14", 0)),
                }

                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        action=action,
                        confidence=final_confidence,
                        strategy_name=self.name,
                        features_snapshot=feature_snapshot,
                        metadata={
                            "reasons": reasons,
                            "buy_score": round(confidence, 3),
                            "sell_score": round(sell_score, 3),
                        },
                    )
                )
            except Exception:
                logger.exception("momentum_strategy.signal_error", symbol=symbol)

        return signals
