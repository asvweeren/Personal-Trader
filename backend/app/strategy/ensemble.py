"""Ensemble strategy with dynamic weight adjustment and conflict resolution."""

from collections import defaultdict

import structlog

from app.data.market_data import MarketSnapshot
from app.strategy.base import SignalAction, Strategy, TradingSignal

logger = structlog.get_logger()


class EnsembleStrategy(Strategy):
    """Combines signals from multiple strategies using weighted voting.

    Features:
    - Configurable weights per strategy
    - Dynamic weight adjustment based on recent accuracy
    - Minimum agreement threshold
    - Conflict resolution when strategies disagree
    """

    @property
    def name(self) -> str:
        return "ensemble"

    def __init__(
        self,
        strategies: list[Strategy],
        weights: dict[str, float] | None = None,
        agreement_threshold: float = 0.3,
        min_strategies: int = 1,
    ):
        self._strategies = strategies
        self._weights = weights or {s.name: 1.0 for s in strategies}
        self._agreement_threshold = agreement_threshold
        self._min_strategies = min_strategies
        # Track accuracy for dynamic weighting
        self._signal_history: dict[str, list[dict]] = defaultdict(list)
        self._max_history = 100

    async def generate_signals(self, market_data: MarketSnapshot) -> list[TradingSignal]:
        # Collect signals from all sub-strategies
        all_signals: dict[str, list[tuple[TradingSignal, float]]] = {}

        for strategy in self._strategies:
            weight = self._weights.get(strategy.name, 1.0)
            try:
                signals = await strategy.generate_signals(market_data)
                for signal in signals:
                    if signal.symbol not in all_signals:
                        all_signals[signal.symbol] = []
                    all_signals[signal.symbol].append((signal, weight))
            except Exception:
                logger.exception("ensemble.strategy_error", strategy=strategy.name)

        combined = []
        for symbol, signal_weights in all_signals.items():
            ensemble_signal = self._combine_signals(symbol, signal_weights)
            if ensemble_signal:
                combined.append(ensemble_signal)

        return combined

    def _combine_signals(
        self,
        symbol: str,
        signal_weights: list[tuple[TradingSignal, float]],
    ) -> TradingSignal | None:
        """Combine signals for a single symbol using weighted voting."""
        if len(signal_weights) < self._min_strategies:
            return None

        # Weighted voting
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = 0.0
        strategy_votes: dict[str, str] = {}

        for signal, weight in signal_weights:
            w = weight * signal.confidence
            if signal.action == SignalAction.BUY:
                buy_score += w
            elif signal.action == SignalAction.SELL:
                sell_score += w
            else:
                hold_score += w
            total_weight += w
            strategy_votes[signal.strategy_name] = signal.action.value

        if total_weight == 0:
            return None

        # Normalize scores
        buy_pct = buy_score / total_weight
        sell_pct = sell_score / total_weight
        hold_pct = hold_score / total_weight

        # Determine action
        net_score = buy_pct - sell_pct

        # Check agreement: at least agreement_threshold of weighted votes must agree
        if net_score > self._agreement_threshold:
            action = SignalAction.BUY
            confidence = buy_pct
        elif net_score < -self._agreement_threshold:
            action = SignalAction.SELL
            confidence = sell_pct
        else:
            # No clear consensus → HOLD
            action = SignalAction.HOLD
            confidence = hold_pct

        # Conflict detection: if strategies strongly disagree, reduce confidence
        has_buy = any(s.action == SignalAction.BUY for s, _ in signal_weights)
        has_sell = any(s.action == SignalAction.SELL for s, _ in signal_weights)
        conflict = has_buy and has_sell

        if conflict:
            confidence *= 0.7  # Reduce confidence when strategies disagree
            logger.debug(
                "ensemble.conflict",
                symbol=symbol,
                votes=strategy_votes,
                net_score=round(net_score, 3),
            )

        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=max(0.0, min(1.0, confidence)),
            strategy_name=self.name,
            metadata={
                "ensemble_score": round(net_score, 4),
                "buy_pct": round(buy_pct, 3),
                "sell_pct": round(sell_pct, 3),
                "hold_pct": round(hold_pct, 3),
                "strategy_votes": strategy_votes,
                "conflict": conflict,
                "num_strategies": len(signal_weights),
            },
        )

    def record_outcome(self, symbol: str, strategy_name: str, was_correct: bool) -> None:
        """Record whether a strategy's signal was correct for dynamic weighting."""
        self._signal_history[strategy_name].append({
            "symbol": symbol,
            "correct": was_correct,
        })
        # Trim history
        if len(self._signal_history[strategy_name]) > self._max_history:
            self._signal_history[strategy_name] = (
                self._signal_history[strategy_name][-self._max_history:]
            )

    def update_weights_from_history(self, performance_tracker=None) -> dict[str, float]:
        """Recalculate weights based on recent accuracy of each strategy.

        If a performance_tracker is provided, uses live attribution data.
        Falls back to signal history otherwise.
        """
        new_weights = {}

        # Try live attribution data first
        if performance_tracker is not None:
            breakdown = performance_tracker.get_strategy_breakdown()
            has_live_data = False
            for strategy in self._strategies:
                stats = breakdown.get(strategy.name)
                if stats and stats.get("trades", 0) >= 10:
                    has_live_data = True
                    win_rate = stats.get("win_rate", 50.0) / 100.0
                    pf = stats.get("profit_factor", 1.0)
                    if isinstance(pf, str):
                        pf = 2.0  # infinity
                    # Weight from win rate and profit factor
                    weight = max(0.1, win_rate * min(pf, 3.0) / 3.0)
                    new_weights[strategy.name] = weight
                else:
                    new_weights[strategy.name] = self._weights.get(strategy.name, 1.0)

            if has_live_data:
                self._weights = new_weights
                logger.info("ensemble.weights_updated_from_attribution", weights=new_weights)
                return new_weights

        # Fall back to signal history
        for strategy in self._strategies:
            history = self._signal_history.get(strategy.name, [])
            if len(history) < 10:
                new_weights[strategy.name] = self._weights.get(strategy.name, 1.0)
                continue

            recent = history[-50:]
            accuracy = sum(1 for h in recent if h["correct"]) / len(recent)
            new_weights[strategy.name] = max(0.1, accuracy ** 2)

        self._weights = new_weights
        logger.info("ensemble.weights_updated", weights=new_weights)
        return new_weights

    def get_weights(self) -> dict[str, float]:
        return dict(self._weights)
