"""Market regime detection using technical indicators."""

from dataclasses import dataclass
from enum import StrEnum

import structlog

from app.data.indicators import compute_features
from app.data.market_data import MarketSnapshot

logger = structlog.get_logger()


class MarketRegime(StrEnum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeState:
    regime: MarketRegime
    confidence: float
    adx: float
    volatility_ratio: float
    mean_reversion_score: float
    breadth: float  # % symbols above SMA50

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 3),
            "adx": round(self.adx, 2),
            "volatility_ratio": round(self.volatility_ratio, 3),
            "mean_reversion_score": round(self.mean_reversion_score, 3),
            "breadth": round(self.breadth, 3),
        }


class RegimeDetector:
    """Detect current market regime from a snapshot of market data.

    Uses ADX, volatility ratio, and market breadth to classify regime.
    """

    # ADX thresholds
    ADX_TRENDING = 25.0
    ADX_STRONG_TREND = 40.0

    # Volatility ratio thresholds (short-term vol / long-term vol)
    VOL_RATIO_HIGH = 1.5
    VOL_RATIO_LOW = 0.7

    def __init__(self):
        self._last_regime: RegimeState | None = None

    @property
    def current_regime(self) -> RegimeState | None:
        return self._last_regime

    def detect(self, snapshot: MarketSnapshot) -> RegimeState:
        """Detect market regime from multi-symbol snapshot."""
        adx_values = []
        vol_ratios = []
        above_sma50_count = 0
        total_symbols = 0

        for symbol, df in snapshot.ohlcv.items():
            if df.empty or len(df) < 60:
                continue

            try:
                features = compute_features(df)
                latest = features.iloc[-1]

                # ADX (use adx_14 if available, else estimate from ATR)
                adx_val = latest.get("adx_14")
                if adx_val is None or adx_val != adx_val:  # NaN check
                    # Estimate trend strength from price vs SMA ratio
                    pvs = latest.get("sma_50", latest.get("price_vs_sma50", 0.0))
                    adx_val = min(abs(pvs) * 200, 60.0) if pvs == pvs else 20.0
                adx_values.append(float(adx_val))

                # Volatility ratio
                vr = latest.get("vol_ratio_10_20")
                if vr is not None and vr == vr:
                    vol_ratios.append(float(vr))

                # Breadth: is price above SMA50?
                pvs50 = latest.get("sma_50", latest.get("price_vs_sma50", 0.0))
                if pvs50 is not None and pvs50 == pvs50 and pvs50 > 0:
                    above_sma50_count += 1
                total_symbols += 1

            except Exception:
                logger.debug("regime.symbol_error", symbol=symbol, exc_info=True)
                continue

        # Calculate aggregate metrics
        avg_adx = sum(adx_values) / len(adx_values) if adx_values else 20.0
        avg_vol_ratio = sum(vol_ratios) / len(vol_ratios) if vol_ratios else 1.0
        breadth = above_sma50_count / total_symbols if total_symbols > 0 else 0.5

        # Mean reversion score: how much breadth deviates from 0.5
        mean_reversion_score = abs(breadth - 0.5) * 2  # 0 = balanced, 1 = extreme

        # Classify regime
        regime, confidence = self._classify(
            avg_adx, avg_vol_ratio, breadth, mean_reversion_score
        )

        state = RegimeState(
            regime=regime,
            confidence=confidence,
            adx=avg_adx,
            volatility_ratio=avg_vol_ratio,
            mean_reversion_score=mean_reversion_score,
            breadth=breadth,
        )
        self._last_regime = state

        logger.info(
            "regime.detected",
            regime=regime.value,
            confidence=round(confidence, 3),
            adx=round(avg_adx, 1),
            vol_ratio=round(avg_vol_ratio, 2),
            breadth=round(breadth, 2),
        )

        return state

    def _classify(
        self,
        adx: float,
        vol_ratio: float,
        breadth: float,
        mean_reversion: float,
    ) -> tuple[MarketRegime, float]:
        """Classify regime based on indicator values."""
        # High volatility overrides everything
        if vol_ratio > self.VOL_RATIO_HIGH:
            confidence = min(0.5 + (vol_ratio - self.VOL_RATIO_HIGH) * 0.3, 0.95)
            return MarketRegime.HIGH_VOLATILITY, confidence

        # Low volatility
        if vol_ratio < self.VOL_RATIO_LOW:
            confidence = min(0.5 + (self.VOL_RATIO_LOW - vol_ratio) * 0.5, 0.90)
            return MarketRegime.LOW_VOLATILITY, confidence

        # Trending
        if adx > self.ADX_TRENDING:
            trend_confidence = min(0.5 + (adx - self.ADX_TRENDING) / 30, 0.95)
            if breadth > 0.6:
                return MarketRegime.TRENDING_UP, trend_confidence
            elif breadth < 0.4:
                return MarketRegime.TRENDING_DOWN, trend_confidence

        # Ranging (low ADX, balanced breadth)
        if adx < self.ADX_TRENDING:
            range_confidence = min(0.5 + (self.ADX_TRENDING - adx) / 25, 0.85)
            return MarketRegime.RANGING, range_confidence

        # Default: ranging with lower confidence
        return MarketRegime.RANGING, 0.5
