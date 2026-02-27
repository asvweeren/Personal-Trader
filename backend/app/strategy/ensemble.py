"""Ensemble strategy with dynamic weight adjustment and conflict resolution."""

import asyncio
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
        agreement_threshold: float = 0.5,
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
        # Collect signals from all sub-strategies in parallel
        all_signals: dict[str, list[tuple[TradingSignal, float]]] = {}

        async def _run_strategy(strategy: Strategy):
            return strategy.name, await strategy.generate_signals(market_data)

        results = await asyncio.gather(
            *[_run_strategy(s) for s in self._strategies],
            return_exceptions=True,
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.exception(
                    "ensemble.strategy_error",
                    strategy=self._strategies[i].name,
                    error=str(result),
                )
                continue
            strategy_name, signals = result
            weight = self._weights.get(strategy_name, 1.0)
            for signal in signals:
                if signal.symbol not in all_signals:
                    all_signals[signal.symbol] = []
                all_signals[signal.symbol].append((signal, weight))

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

        # Conflict detection: only block when strategies *strongly* disagree
        has_strong_buy = any(
            s.action == SignalAction.BUY and s.confidence > 0.7
            for s, _ in signal_weights
        )
        has_strong_sell = any(
            s.action == SignalAction.SELL and s.confidence > 0.7
            for s, _ in signal_weights
        )
        conflict = has_strong_buy and has_strong_sell

        if conflict:
            # Strong disagreement — force HOLD
            action = SignalAction.HOLD
            confidence = 0.0
            logger.debug(
                "ensemble.conflict_forced_hold",
                symbol=symbol,
                votes=strategy_votes,
                net_score=round(net_score, 3),
            )
        elif any(s.action == SignalAction.BUY for s, _ in signal_weights) and \
                any(s.action == SignalAction.SELL for s, _ in signal_weights):
            # Mild disagreement — reduce confidence by 30%
            confidence *= 0.7
            logger.debug(
                "ensemble.mild_conflict_reduced",
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

    def check_diversity(self) -> dict:
        """Check if strategies are sufficiently diverse (not always agreeing).

        Returns diversity stats including agreement rate and warning flag.
        """
        if len(self._signal_history) < 2:
            return {"diverse": True, "agreement_rate": 0.0, "samples": 0}

        # Collect recent outcomes per symbol across strategies
        strategy_names = list(self._signal_history.keys())
        if len(strategy_names) < 2:
            return {"diverse": True, "agreement_rate": 0.0, "samples": 0}

        s1_history = self._signal_history.get(strategy_names[0], [])
        s2_history = self._signal_history.get(strategy_names[1], [])
        if len(s1_history) < 10 or len(s2_history) < 10:
            return {"diverse": True, "agreement_rate": 0.0, "samples": 0}

        # Compare last 50 outcomes
        s1_recent = [h["correct"] for h in s1_history[-50:]]
        s2_recent = [h["correct"] for h in s2_history[-50:]]
        min_len = min(len(s1_recent), len(s2_recent))
        agreements = sum(
            1 for i in range(min_len) if s1_recent[i] == s2_recent[i]
        )
        agreement_rate = agreements / min_len if min_len > 0 else 0.0

        is_diverse = agreement_rate < 0.9
        if not is_diverse:
            logger.warning(
                "ensemble.low_diversity",
                agreement_rate=round(agreement_rate, 3),
                strategies=strategy_names[:2],
            )

        return {
            "diverse": is_diverse,
            "agreement_rate": round(agreement_rate, 3),
            "samples": min_len,
        }
