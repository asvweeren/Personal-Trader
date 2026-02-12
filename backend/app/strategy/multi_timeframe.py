"""Multi-timeframe confirmation filter for trading signals."""

import structlog

from app.strategy.base import TradingSignal, SignalAction

logger = structlog.get_logger()


class MultiTimeframeFilter:
    """Confirms signals using alignment between hourly and daily timeframes.

    If both timeframes agree on direction, confidence is boosted.
    If they conflict, confidence is reduced.
    """

    BOOST_FACTOR = 1.15
    PENALTY_FACTOR = 0.7
    MAX_CONFIDENCE = 0.95

    def confirm_signal(
        self,
        signal: TradingSignal,
        hourly_features: dict,
        daily_features: dict,
    ) -> TradingSignal:
        """Adjust signal confidence based on multi-timeframe alignment.

        Args:
            signal: The trading signal to confirm
            hourly_features: Latest features from hourly data
            daily_features: Latest features from daily data

        Returns:
            Signal with adjusted confidence
        """
        if signal.action == SignalAction.HOLD:
            return signal

        hourly_trend = self._assess_trend(hourly_features)
        daily_trend = self._assess_trend(daily_features)

        # Determine alignment
        if signal.action == SignalAction.BUY:
            signal_direction = 1
        else:
            signal_direction = -1

        hourly_aligned = (hourly_trend > 0 and signal_direction > 0) or (
            hourly_trend < 0 and signal_direction < 0
        )
        daily_aligned = (daily_trend > 0 and signal_direction > 0) or (
            daily_trend < 0 and signal_direction < 0
        )

        # Calculate adjustment
        if hourly_aligned and daily_aligned:
            # Both timeframes agree — boost confidence
            new_confidence = min(
                signal.confidence * self.BOOST_FACTOR,
                self.MAX_CONFIDENCE,
            )
            alignment = "aligned"
        elif not hourly_aligned and not daily_aligned:
            # Both timeframes conflict — reduce confidence
            new_confidence = signal.confidence * self.PENALTY_FACTOR
            alignment = "conflicting"
        else:
            # Mixed — slight penalty
            new_confidence = signal.confidence * 0.9
            alignment = "mixed"

        logger.debug(
            "mtf.confirmation",
            symbol=signal.symbol,
            action=signal.action.value,
            hourly_trend=round(hourly_trend, 3),
            daily_trend=round(daily_trend, 3),
            alignment=alignment,
            old_confidence=round(signal.confidence, 3),
            new_confidence=round(new_confidence, 3),
        )

        # Return a new signal with updated confidence and metadata
        metadata = dict(signal.metadata or {})
        metadata["mtf_alignment"] = alignment
        metadata["mtf_hourly_trend"] = round(hourly_trend, 3)
        metadata["mtf_daily_trend"] = round(daily_trend, 3)

        return TradingSignal(
            symbol=signal.symbol,
            action=signal.action,
            confidence=max(0.0, min(new_confidence, 1.0)),
            strategy_name=signal.strategy_name,
            features_snapshot=signal.features_snapshot,
            metadata=metadata,
        )

    def _assess_trend(self, features: dict) -> float:
        """Assess trend direction from features. Returns -1 to +1.

        Uses: price_vs_sma10, rsi_14, macd_divergence.
        """
        score = 0.0
        count = 0

        # Price vs SMA10 (positive = bullish)
        pvs = features.get("price_vs_sma10")
        if pvs is not None and pvs == pvs:  # NaN check
            score += 1.0 if pvs > 0.005 else (-1.0 if pvs < -0.005 else 0.0)
            count += 1

        # RSI (above 50 = bullish, below 50 = bearish)
        rsi = features.get("rsi_14")
        if rsi is not None and rsi == rsi:
            if rsi > 55:
                score += 1.0
            elif rsi < 45:
                score += -1.0
            count += 1

        # MACD divergence (positive = bullish)
        macd_div = features.get("macd_divergence")
        if macd_div is not None and macd_div == macd_div:
            score += 1.0 if macd_div > 0 else (-1.0 if macd_div < 0 else 0.0)
            count += 1

        # Momentum
        mom = features.get("momentum_10d")
        if mom is not None and mom == mom:
            score += 1.0 if mom > 0.01 else (-1.0 if mom < -0.01 else 0.0)
            count += 1

        if count == 0:
            return 0.0
        return score / count
