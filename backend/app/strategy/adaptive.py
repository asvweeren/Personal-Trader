"""Self-improving adaptive strategy layer.

Manages online learning from trade outcomes to continuously improve:
1. Per-symbol confidence thresholds
2. Per-regime parameter sets
3. Feature importance tracking
4. Confidence calibration from realized P&L
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

ADAPTIVE_STATE_PATH = Path("ml/models/adaptive_state.json")


@dataclass
class TradeOutcome:
    """Recorded outcome of a trade for learning."""
    symbol: str
    regime: str
    confidence: float
    pnl: float
    strategy_name: str
    timestamp: str
    features_used: list[str] = field(default_factory=list)


@dataclass
class SymbolProfile:
    """Adaptive per-symbol trading profile."""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    confidence_adjustment: float = 0.0  # Added to base threshold
    avg_winning_confidence: float = 0.0
    avg_losing_confidence: float = 0.0
    _winning_confidences: list[float] = field(default_factory=list)
    _losing_confidences: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0

    def record(self, pnl: float, confidence: float) -> None:
        self.trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
            self._winning_confidences.append(confidence)
            # Keep last 50
            self._winning_confidences = self._winning_confidences[-50:]
            self.avg_winning_confidence = (
                sum(self._winning_confidences) / len(self._winning_confidences)
            )
        else:
            self.losses += 1
            self._losing_confidences.append(confidence)
            self._losing_confidences = self._losing_confidences[-50:]
            self.avg_losing_confidence = (
                sum(self._losing_confidences) / len(self._losing_confidences)
            )
        self._update_adjustment()

    def _update_adjustment(self) -> None:
        """Compute confidence threshold adjustment based on outcome history.

        If win rate is low, raise the bar. If high, lower it (trade more).
        Only adjusts after sufficient data (10+ trades).
        """
        if self.trades < 10:
            self.confidence_adjustment = 0.0
            return

        # Target: win rate >= 40% to be profitable with 1.5:1 R:R
        wr = self.win_rate
        if wr >= 0.50:
            # Performing well — lower threshold slightly to capture more trades
            self.confidence_adjustment = -0.03
        elif wr >= 0.40:
            # Acceptable — no adjustment
            self.confidence_adjustment = 0.0
        elif wr >= 0.30:
            # Underperforming — raise threshold
            self.confidence_adjustment = 0.05
        else:
            # Poor — significantly raise threshold
            self.confidence_adjustment = 0.10

    def to_dict(self) -> dict:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate, 4),
            "confidence_adjustment": round(self.confidence_adjustment, 4),
            "avg_winning_confidence": round(self.avg_winning_confidence, 4),
            "avg_losing_confidence": round(self.avg_losing_confidence, 4),
        }


@dataclass
class RegimeProfile:
    """Adaptive parameter set per market regime."""
    regime_name: str = ""
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    confidence_adjustment: float = 0.0
    optimal_hold_minutes: float = 0.0
    _hold_times_winning: list[float] = field(default_factory=list)
    _pnls: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0

    def record(self, pnl: float, hold_minutes: float = 0.0) -> None:
        self.trades += 1
        self.total_pnl += pnl
        self._pnls.append(pnl)
        self._pnls = self._pnls[-100:]
        if pnl > 0:
            self.wins += 1
            if hold_minutes > 0:
                self._hold_times_winning.append(hold_minutes)
                self._hold_times_winning = self._hold_times_winning[-50:]
                self.optimal_hold_minutes = (
                    sum(self._hold_times_winning) / len(self._hold_times_winning)
                )
        self._update_adjustment()

    def _update_adjustment(self) -> None:
        if self.trades < 15:
            self.confidence_adjustment = 0.0
            return
        wr = self.win_rate
        if wr >= 0.50:
            self.confidence_adjustment = -0.03
        elif wr >= 0.40:
            self.confidence_adjustment = 0.0
        elif wr >= 0.30:
            self.confidence_adjustment = 0.05
        else:
            self.confidence_adjustment = 0.10

    def to_dict(self) -> dict:
        return {
            "regime": self.regime_name,
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "confidence_adjustment": round(self.confidence_adjustment, 4),
            "optimal_hold_minutes": round(self.optimal_hold_minutes, 1),
        }


class AdaptiveManager:
    """Central manager for all self-improving features.

    Learns from trade outcomes to adjust:
    - Per-symbol confidence thresholds
    - Per-regime parameters
    - Feature importance tracking
    - Overall confidence calibration
    """

    def __init__(self) -> None:
        self._symbol_profiles: dict[str, SymbolProfile] = {}
        self._regime_profiles: dict[str, RegimeProfile] = {}
        self._feature_hit_rates: dict[str, list[bool]] = defaultdict(list)
        self._outcomes: list[TradeOutcome] = []
        self._max_outcomes = 500
        self._load_state()

    def record_trade_outcome(
        self,
        symbol: str,
        pnl: float,
        confidence: float,
        strategy_name: str,
        regime: str = "unknown",
        hold_minutes: float = 0.0,
        top_features: list[str] | None = None,
    ) -> None:
        """Record a completed trade for adaptive learning."""
        outcome = TradeOutcome(
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            pnl=pnl,
            strategy_name=strategy_name,
            timestamp=datetime.now(UTC).isoformat(),
            features_used=top_features or [],
        )
        self._outcomes.append(outcome)
        if len(self._outcomes) > self._max_outcomes:
            self._outcomes = self._outcomes[-self._max_outcomes:]

        # Update symbol profile
        if symbol not in self._symbol_profiles:
            self._symbol_profiles[symbol] = SymbolProfile()
        self._symbol_profiles[symbol].record(pnl, confidence)

        # Update regime profile
        if regime not in self._regime_profiles:
            self._regime_profiles[regime] = RegimeProfile(regime_name=regime)
        self._regime_profiles[regime].record(pnl, hold_minutes)

        # Track feature effectiveness
        was_correct = pnl > 0
        for feat in (top_features or []):
            self._feature_hit_rates[feat].append(was_correct)
            # Keep last 100 per feature
            if len(self._feature_hit_rates[feat]) > 100:
                self._feature_hit_rates[feat] = self._feature_hit_rates[feat][-100:]

        logger.info(
            "adaptive.outcome_recorded",
            symbol=symbol,
            regime=regime,
            pnl=round(pnl, 2),
            confidence=round(confidence, 3),
        )

    def get_symbol_threshold_adjustment(self, symbol: str) -> float:
        """Get the confidence threshold adjustment for a specific symbol.

        Returns a value to ADD to the base threshold (can be negative).
        """
        profile = self._symbol_profiles.get(symbol)
        if profile is None or profile.trades < 10:
            return 0.0
        return profile.confidence_adjustment

    def get_regime_threshold_adjustment(self, regime: str) -> float:
        """Get the confidence threshold adjustment for a market regime."""
        profile = self._regime_profiles.get(regime)
        if profile is None or profile.trades < 15:
            return 0.0
        return profile.confidence_adjustment

    def get_adjusted_threshold(
        self, base_threshold: float, symbol: str, regime: str = "unknown"
    ) -> float:
        """Get the fully adjusted confidence threshold for a symbol + regime.

        Combines base threshold with per-symbol and per-regime adjustments.
        Clamped to [0.30, 0.80] to prevent extreme values.
        """
        sym_adj = self.get_symbol_threshold_adjustment(symbol)
        regime_adj = self.get_regime_threshold_adjustment(regime)
        adjusted = base_threshold + sym_adj + regime_adj
        return max(0.30, min(0.80, adjusted))

    def get_declining_features(self, min_samples: int = 30) -> list[str]:
        """Identify features whose predictive power is declining.

        Returns features with < 40% hit rate over recent trades.
        """
        declining = []
        for feat, outcomes in self._feature_hit_rates.items():
            if len(outcomes) < min_samples:
                continue
            recent = outcomes[-min_samples:]
            hit_rate = sum(recent) / len(recent)
            if hit_rate < 0.40:
                declining.append(feat)
        return declining

    def get_feature_effectiveness(self) -> dict[str, dict]:
        """Get effectiveness stats for all tracked features."""
        result = {}
        for feat, outcomes in self._feature_hit_rates.items():
            if len(outcomes) < 10:
                continue
            recent = outcomes[-50:]
            result[feat] = {
                "total_samples": len(outcomes),
                "recent_hit_rate": round(sum(recent) / len(recent), 4),
                "overall_hit_rate": round(sum(outcomes) / len(outcomes), 4),
            }
        return dict(sorted(result.items(), key=lambda x: x[1]["recent_hit_rate"], reverse=True))

    def get_regime_optimal_hold(self, regime: str) -> float | None:
        """Get the optimal hold time for a regime based on winning trades."""
        profile = self._regime_profiles.get(regime)
        if profile and profile.optimal_hold_minutes > 0:
            return profile.optimal_hold_minutes
        return None

    def should_skip_symbol(self, symbol: str) -> bool:
        """Check if a symbol should be skipped based on adaptive learning.

        More nuanced than the static blacklist — uses win rate and P&L trend.
        """
        profile = self._symbol_profiles.get(symbol)
        if profile is None:
            return False
        # Skip if >= 15 trades with < 25% win rate
        if profile.trades >= 15 and profile.win_rate < 0.25:
            return True
        # Skip if >= 10 trades and losing significant money
        if profile.trades >= 10 and profile.total_pnl < -1000:
            return True
        return False

    def save_state(self) -> None:
        """Persist adaptive state to disk for survival across restarts."""
        try:
            ADAPTIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "saved_at": datetime.now(UTC).isoformat(),
                "symbol_profiles": {
                    sym: p.to_dict() for sym, p in self._symbol_profiles.items()
                },
                "regime_profiles": {
                    r: p.to_dict() for r, p in self._regime_profiles.items()
                },
                "feature_effectiveness": self.get_feature_effectiveness(),
                "total_outcomes": len(self._outcomes),
            }
            with open(ADAPTIVE_STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)
            logger.info("adaptive.state_saved", outcomes=len(self._outcomes))
        except Exception:
            logger.exception("adaptive.save_error")

    def _load_state(self) -> None:
        """Load adaptive state from disk."""
        if not ADAPTIVE_STATE_PATH.exists():
            return
        try:
            with open(ADAPTIVE_STATE_PATH) as f:
                state = json.load(f)

            for sym, data in state.get("symbol_profiles", {}).items():
                profile = SymbolProfile(
                    trades=data.get("trades", 0),
                    wins=data.get("wins", 0),
                    losses=data.get("losses", 0),
                    total_pnl=data.get("total_pnl", 0.0),
                    confidence_adjustment=data.get("confidence_adjustment", 0.0),
                    avg_winning_confidence=data.get("avg_winning_confidence", 0.0),
                    avg_losing_confidence=data.get("avg_losing_confidence", 0.0),
                )
                self._symbol_profiles[sym] = profile

            for regime, data in state.get("regime_profiles", {}).items():
                profile = RegimeProfile(
                    regime_name=data.get("regime", regime),
                    trades=data.get("trades", 0),
                    wins=data.get("wins", 0),
                    total_pnl=data.get("total_pnl", 0.0),
                    confidence_adjustment=data.get("confidence_adjustment", 0.0),
                    optimal_hold_minutes=data.get("optimal_hold_minutes", 0.0),
                )
                self._regime_profiles[regime] = profile

            logger.info(
                "adaptive.state_loaded",
                symbols=len(self._symbol_profiles),
                regimes=len(self._regime_profiles),
            )
        except Exception:
            logger.warning("adaptive.load_error", exc_info=True)

    def get_summary(self) -> dict:
        """Get a summary of all adaptive state for dashboard/API."""
        return {
            "total_outcomes": len(self._outcomes),
            "symbol_profiles": {
                sym: p.to_dict() for sym, p in self._symbol_profiles.items()
            },
            "regime_profiles": {
                r: p.to_dict() for r, p in self._regime_profiles.items()
            },
            "declining_features": self.get_declining_features(),
            "feature_effectiveness": self.get_feature_effectiveness(),
        }
