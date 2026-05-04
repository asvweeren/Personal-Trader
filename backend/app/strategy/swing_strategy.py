"""Swing trading strategy using daily indicators.

Holds positions for days/weeks, following established trends.
Uses daily SMA, RSI, MACD, ADX — not intraday noise.

BUY entry:
  1. Price above SMA50 (established uptrend)
  2. SMA20 > SMA50 (golden cross zone)
  3. RSI 40-60 (room to run, not overbought)
  4. MACD histogram positive and rising
  5. ADX > 20 (trend exists)
  6. Volume above 20-day average
  7. 10-day momentum positive

SELL/SHORT entry (inverse):
  1. Price below SMA50
  2. SMA20 < SMA50 (death cross zone)
  3. RSI 40-60 (room to fall, not oversold)
  4. MACD histogram negative and falling
  5. ADX > 20
  6. Volume above average on decline
  7. 10-day momentum negative

Exit: ATR-based stops (3x daily ATR), take-profit (6x daily ATR), or max_hold_days.
"""

import structlog

from app.data.indicators import compute_features
from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction, Strategy, TradingSignal

logger = structlog.get_logger()


class SwingStrategy(Strategy):
    """Daily trend-following swing strategy."""

    @property
    def name(self) -> str:
        return "swing"

    def __init__(self, confidence_threshold: float = 0.60):
        self._confidence_threshold = confidence_threshold

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        signals = []

        for symbol, df in market_data.ohlcv.items():
            if df.empty or len(df) < 60:
                continue

            try:
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

                # Daily indicators (all normalized ratios from compute_features)
                sma20 = float(latest.get("sma_20", 0))       # close/sma20 - 1
                sma50 = float(latest.get("sma_50", 0))       # close/sma50 - 1
                rsi = float(latest.get("rsi_14", 50))
                macd_hist = float(latest.get("macd_histogram", 0))
                macd_hist_prev = float(prev.get("macd_histogram", 0))
                adx = float(latest.get("adx_14", 0))
                vol_ratio = float(latest.get("volume_ratio", latest.get("volume_sma_20", 1.0)))
                momentum_10d = float(latest.get("momentum_10d", 0))
                momentum_20d = float(latest.get("momentum_20d", 0))
                atr = float(latest.get("atr_14", 0))
                bb_width = float(latest.get("bb_width", 0))
                regime_bull = float(latest.get("regime_bull_bear", 0))

                # --- BUY SCORING: Daily uptrend confirmation ---
                buy_score = 0.0
                buy_reasons = []

                # 1. Price above SMA50 (established uptrend)
                if sma50 > 0:
                    buy_score += 0.20
                    buy_reasons.append("above_sma50")

                # 2. SMA20 above SMA50 (golden cross zone) — sma20 > sma50 means
                # price/sma20 - 1 being closer to 0 than price/sma50 - 1 isn't
                # directly comparable. Use regime_bull_bear (SMA50 vs SMA200) instead.
                if regime_bull == 1.0:
                    buy_score += 0.15
                    buy_reasons.append("bull_regime")
                elif sma20 > sma50 and sma20 > 0:
                    buy_score += 0.10
                    buy_reasons.append("sma20_above_sma50")

                # 3. RSI in sweet spot (40-60 for entries — room to run)
                if 40 <= rsi <= 60:
                    buy_score += 0.20
                    buy_reasons.append(f"rsi_sweet={rsi:.0f}")
                elif 35 <= rsi < 40:
                    buy_score += 0.10  # Near oversold bounce

                # 4. MACD positive and accelerating
                if macd_hist > 0 and macd_hist > macd_hist_prev:
                    buy_score += 0.20
                    buy_reasons.append("macd_accel")
                elif macd_hist > 0:
                    buy_score += 0.10

                # 5. ADX > 20 (trend exists, not directionless chop)
                if adx > 20:
                    buy_score += 0.10
                    buy_reasons.append(f"adx={adx:.0f}")

                # 6. Volume above average (conviction)
                if vol_ratio > 1.2:
                    buy_score += 0.10
                    buy_reasons.append(f"vol={vol_ratio:.1f}x")

                # 7. Positive multi-day momentum
                if momentum_10d > 0.02:
                    buy_score += 0.10
                    buy_reasons.append(f"mom10d={momentum_10d:.1%}")
                elif momentum_10d > 0:
                    buy_score += 0.05

                buy_score = min(buy_score, 1.0)

                # --- SELL/SHORT SCORING: Daily downtrend confirmation ---
                sell_score = 0.0
                sell_reasons = []

                # 1. Price below SMA50 (downtrend)
                if sma50 < 0:
                    sell_score += 0.20
                    sell_reasons.append("below_sma50")

                # 2. Bear regime (SMA50 below SMA200)
                if regime_bull == -1.0:
                    sell_score += 0.15
                    sell_reasons.append("bear_regime")
                elif sma20 < sma50 and sma20 < 0:
                    sell_score += 0.10
                    sell_reasons.append("sma20_below_sma50")

                # 3. RSI in short sweet spot (40-60 from above — room to fall)
                if 40 <= rsi <= 60:
                    sell_score += 0.20
                    sell_reasons.append(f"rsi_short_zone={rsi:.0f}")
                elif 60 < rsi <= 65:
                    sell_score += 0.10  # Near overbought reversal

                # 4. MACD negative and decelerating
                if macd_hist < 0 and macd_hist < macd_hist_prev:
                    sell_score += 0.20
                    sell_reasons.append("macd_decline")
                elif macd_hist < 0:
                    sell_score += 0.10

                # 5. ADX > 20 (trending, not chop)
                if adx > 20:
                    sell_score += 0.10
                    sell_reasons.append(f"adx={adx:.0f}")

                # 6. Volume on decline
                if vol_ratio > 1.2 and sma50 < 0:
                    sell_score += 0.10
                    sell_reasons.append(f"vol_decline={vol_ratio:.1f}x")

                # 7. Negative multi-day momentum
                if momentum_10d < -0.02:
                    sell_score += 0.10
                    sell_reasons.append(f"mom10d={momentum_10d:.1%}")
                elif momentum_10d < 0:
                    sell_score += 0.05

                sell_score = min(sell_score, 1.0)

                # Determine action
                if buy_score >= self._confidence_threshold and buy_score > sell_score:
                    action = SignalAction.BUY
                    final_confidence = buy_score
                    reasons = buy_reasons
                elif sell_score >= self._confidence_threshold and sell_score > buy_score:
                    action = SignalAction.SELL
                    final_confidence = sell_score
                    reasons = sell_reasons
                else:
                    continue  # Skip HOLD signals entirely for swing

                signals.append(TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=final_confidence,
                    strategy_name=self.name,
                    features_snapshot={
                        "atr_14": atr,
                        "rsi_14": rsi,
                        "adx_14": adx,
                        "macd_histogram": macd_hist,
                        "sma_20": sma20,
                        "sma_50": sma50,
                        "ema_10": float(latest.get("ema_10", 0)),
                        "vwap": float(latest.get("vwap", 0)),
                        "bb_width": bb_width,
                        "volume_ratio": vol_ratio,
                        "return_1d": float(latest.get("return_1d", 0)),
                        "momentum_10d": momentum_10d,
                        "momentum_20d": momentum_20d,
                    },
                    metadata={
                        "reasons": reasons,
                        "buy_score": round(buy_score, 3),
                        "sell_score": round(sell_score, 3),
                        "regime": "bull" if regime_bull == 1.0 else "bear" if regime_bull == -1.0 else "neutral",
                    },
                ))
            except Exception:
                logger.exception("swing_strategy.signal_error", symbol=symbol)

        return signals
