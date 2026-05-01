"""Agentic trading strategy powered by Claude AI.

Instead of fixed rules, an LLM agent analyzes market data, technical indicators,
positions, and risk metrics to make trading decisions with full reasoning.

Uses Claude Haiku for cost efficiency (~$0.03-0.05 per cycle).
Hard risk limits in the engine remain as non-overridable safety layer.
"""

import json
import time
from datetime import UTC, datetime

import anthropic
import structlog

from app.config import settings
from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction, Strategy, TradingSignal

logger = structlog.get_logger()

AGENT_MODEL = "claude-haiku-4-5-20251001"
MAX_SYMBOLS_PER_CALL = 10


class AgenticStrategy(Strategy):
    """LLM-powered trading agent that reasons about market conditions."""

    @property
    def name(self) -> str:
        return "agentic"

    def __init__(self, confidence_threshold: float = 0.65):
        self._confidence_threshold = confidence_threshold
        self._client: anthropic.AsyncAnthropic | None = None
        self._calls: list[float] = []
        self._max_calls_per_minute = 5
        if settings.anthropic_api_key:
            self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def _rate_limit(self) -> None:
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < 60]
        if len(self._calls) >= self._max_calls_per_minute:
            wait = 60 - (now - self._calls[0])
            if wait > 0:
                import asyncio
                await asyncio.sleep(wait)
        self._calls.append(time.monotonic())

    def _get_features(self, market_data: MarketSnapshot, symbol: str) -> dict:
        """Extract features from computed_features_df (primary) or fallback to features dict."""
        feat = {}
        # Primary source: computed features DataFrame (always populated by engine)
        df = market_data.computed_features_df.get(symbol)
        if df is not None and not df.empty:
            try:
                row = df.iloc[-1]
                feat = {
                    k: float(v) for k, v in row.items()
                    if isinstance(v, (int, float)) and v == v  # skip NaN
                }
            except Exception:
                pass
        # Overlay intraday features if available
        intraday = market_data.intraday_features.get(symbol, {})
        if intraday:
            feat.update(intraday)
        return feat

    def _build_market_context(self, market_data: MarketSnapshot, symbols: list[str]) -> str:
        """Build a concise market context string for the LLM."""
        lines = []
        now = datetime.now(UTC)
        lines.append(f"Timestamp: {now.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"Strategy: Day trading (all positions close EOD)")
        lines.append("")

        for symbol in symbols:
            price = market_data.prices.get(symbol, 0)
            if price <= 0:
                continue

            feat = self._get_features(market_data, symbol)
            if not feat:
                continue

            rsi = feat.get("rsi_14", 0)
            macd_hist = feat.get("macd_histogram", 0)
            vwap = feat.get("vwap", 0)
            adx = feat.get("adx_14", 0)
            atr = feat.get("atr_14", 0)
            vol_ratio = feat.get("vol_ratio_10_20", 1.0)
            ema10 = feat.get("ema_10", 0)
            momentum = feat.get("return_1d", 0)
            bb_width = feat.get("bb_width", 0)
            sentiment = feat.get("sentiment_score", 0)

            # ema_10/vwap are normalized: positive = price above
            above_vwap = "above" if vwap < 0 else "below" if vwap > 0 else "at"
            above_ema = "above" if ema10 > 0 else "below" if ema10 < 0 else "at"

            line = (
                f"{symbol}: ${price:.2f} | RSI={rsi:.0f} | MACD_hist={macd_hist:.4f} | "
                f"VWAP={above_vwap} | ADX={adx:.0f} | ATR={atr:.3f} | "
                f"Vol={vol_ratio:.1f}x | EMA10={above_ema} | "
                f"Momentum={momentum:.2%} | BB_width={bb_width:.3f}"
            )
            if sentiment:
                line += f" | Sentiment={sentiment:.2f}"
            lines.append(line)

        return "\n".join(lines)

    def _build_prompt(self, market_context: str) -> str:
        return f"""You are an expert day trader analyzing US stocks. Identify the best BUY and SELL (short) opportunities.

RULES:
- Day trading only: all positions close at end of day
- BUY: strong bullish momentum (RSI 50-70, MACD accelerating, above VWAP/EMA, ADX>25, volume surge)
- SELL (short): strong bearish momentum (RSI<40, MACD declining, below VWAP/EMA, ADX>25, negative momentum)
- Maximum 3 recommendations total (BUY + SELL combined)
- Confidence: 0.80 = good, 0.85 = strong, 0.90+ = excellent. Minimum 0.75.
- In bearish markets prefer SELL signals over BUY
- Avoid: low volume (<1.0x), weak ADX (<20), unclear direction

MARKET DATA:
{market_context}

Respond with ONLY a valid JSON array:
{{"symbol": "TICKER", "action": "BUY" or "SELL", "confidence": 0.75-0.95, "reasoning": "one sentence"}}

If no opportunities: []
JSON:"""

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        if not self._client:
            return []

        symbols = [s for s in market_data.prices if market_data.prices[s] > 0]
        if not symbols:
            return []

        # Limit symbols to avoid token overflow
        symbols = symbols[:MAX_SYMBOLS_PER_CALL]

        try:
            await self._rate_limit()

            market_context = self._build_market_context(market_data, symbols)
            prompt = self._build_prompt(market_context)

            response = await self._client.messages.create(
                model=AGENT_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.content[0].text.strip()
            # Remove markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            # Extract JSON array even if Claude adds reasoning after it
            bracket_start = raw.find("[")
            bracket_end = raw.rfind("]")
            if bracket_start >= 0 and bracket_end > bracket_start:
                raw = raw[bracket_start:bracket_end + 1]

            decisions = json.loads(raw)

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost_est = (input_tokens * 0.80 + output_tokens * 4.0) / 1_000_000
            logger.info(
                "agentic.response",
                decisions=len(decisions),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=round(cost_est, 4),
            )

            signals = []
            for d in decisions:
                action_str = d.get("action", "HOLD").upper()
                if action_str not in ("BUY", "SELL"):
                    continue

                confidence = float(d.get("confidence", 0))
                if confidence < self._confidence_threshold:
                    continue

                symbol = d.get("symbol", "").upper()
                if symbol not in market_data.prices:
                    continue

                action = SignalAction.BUY if action_str == "BUY" else SignalAction.SELL
                reasoning = d.get("reasoning", "")

                feat = self._get_features(market_data, symbol)

                signals.append(TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    strategy_name=self.name,
                    features_snapshot={
                        "atr_14": feat.get("atr_14", 0),
                        "rsi_14": feat.get("rsi_14", 0),
                        "adx_14": feat.get("adx_14", 0),
                        "macd_histogram": feat.get("macd_histogram", 0),
                        "sma_50": feat.get("sma_50", 0),
                        "sma_20": feat.get("sma_20", 0),
                        "ema_10": feat.get("ema_10", 0),
                        "vwap": feat.get("vwap", 0),
                        "bb_width": feat.get("bb_width", 0),
                        "volume_ratio": feat.get("volume_ratio", feat.get("vol_ratio_10_20", 1.0)),
                        "return_1d": feat.get("return_1d", 0),
                    },
                    metadata={
                        "reasoning": reasoning,
                        "model": AGENT_MODEL,
                    },
                ))

            return signals

        except json.JSONDecodeError:
            logger.warning("agentic.json_parse_error", raw=raw[:200])
            return []
        except anthropic.AuthenticationError:
            logger.error("agentic.auth_error")
            self._client = None
            return []
        except Exception:
            logger.exception("agentic.error")
            return []
