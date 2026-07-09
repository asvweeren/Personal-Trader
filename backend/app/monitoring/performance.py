from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

logger = structlog.get_logger()


@dataclass
class StrategyMetrics:
    """Per-strategy performance metrics."""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    avg_confidence: float = 0.0
    total_confidence: float = 0.0
    trade_pnls: list[float] = field(default_factory=list)
    hold_durations: list[float] = field(default_factory=list)  # minutes

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(p for p in self.trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in self.trade_pnls if p < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def avg_hold_time(self) -> float:
        return sum(self.hold_durations) / len(self.hold_durations) if self.hold_durations else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.pnl / self.trades if self.trades > 0 else 0.0

    def to_dict(self) -> dict:
        pf = self.profit_factor
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "pnl": round(self.pnl, 2),
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(pf, 2) if pf != float("inf") else "∞",
            "avg_confidence": round(self.avg_confidence, 3),
            "avg_hold_time_minutes": round(self.avg_hold_time, 1),
        }


@dataclass
class PerformanceTracker:
    """Tracks trading performance metrics in memory, periodically persisted to DB."""

    initial_capital: float = 5000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    # Realized P&L accumulated since the last daily reset. Unlike
    # realized_pnl (all-time, restored from trade history on startup),
    # this only counts trades closed today.
    daily_realized_pnl: float = 0.0
    # Unrealized P&L at the moment of the last daily reset, so intraday
    # unrealized *change* can be computed.
    daily_start_unrealized: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_commission: float = 0.0
    max_drawdown: float = 0.0
    peak_value: float = 0.0
    consecutive_losses: int = 0
    trade_pnls: list[float] = field(default_factory=list)
    daily_start_value: float = 0.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(UTC))
    strategy_metrics: dict[str, StrategyMetrics] = field(default_factory=dict)
    symbol_metrics: dict[str, StrategyMetrics] = field(default_factory=dict)
    api_calls_today: int = 0
    api_cost_today_usd: float = 0.0

    def __post_init__(self):
        if self.peak_value == 0.0:
            self.peak_value = self.initial_capital
        if self.daily_start_value == 0.0:
            self.daily_start_value = self.initial_capital

    @property
    def total_value(self) -> float:
        return self.initial_capital + self.realized_pnl + self.unrealized_pnl

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return ((self.total_value - self.initial_capital) / self.initial_capital) * 100

    @property
    def daily_return_pct(self) -> float:
        if self.daily_start_value == 0:
            return 0.0
        return ((self.total_value - self.daily_start_value) / self.daily_start_value) * 100

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(p for p in self.trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in self.trade_pnls if p < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def record_trade(
        self,
        pnl: float,
        commission: float = 0.0,
        strategy_name: str | None = None,
        confidence: float = 0.0,
        hold_duration_minutes: float = 0.0,
        symbol: str | None = None,
    ) -> None:
        self.trade_pnls.append(pnl)
        self.realized_pnl += pnl
        self.daily_realized_pnl += pnl
        self.total_commission += commission
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.losing_trades += 1
            self.consecutive_losses += 1

        # Per-strategy attribution
        if strategy_name:
            if strategy_name not in self.strategy_metrics:
                self.strategy_metrics[strategy_name] = StrategyMetrics()
            sm = self.strategy_metrics[strategy_name]
            sm.trades += 1
            sm.pnl += pnl
            sm.trade_pnls.append(pnl)
            sm.total_confidence += confidence
            sm.avg_confidence = sm.total_confidence / sm.trades
            if hold_duration_minutes > 0:
                sm.hold_durations.append(hold_duration_minutes)
            if pnl > 0:
                sm.wins += 1
            elif pnl < 0:
                sm.losses += 1

        # Per-symbol attribution
        if symbol:
            if symbol not in self.symbol_metrics:
                self.symbol_metrics[symbol] = StrategyMetrics()
            sym = self.symbol_metrics[symbol]
            sym.trades += 1
            sym.pnl += pnl
            sym.trade_pnls.append(pnl)
            if pnl > 0:
                sym.wins += 1
            elif pnl < 0:
                sym.losses += 1

        self._update_drawdown()

    def update_unrealized(self, unrealized_pnl: float) -> None:
        self.unrealized_pnl = unrealized_pnl
        self._update_drawdown()

    def record_api_call(self, cost_usd: float = 0.0003) -> None:
        """Record an API call and its estimated cost."""
        self.api_calls_today += 1
        self.api_cost_today_usd += cost_usd

    def reset_daily(self) -> None:
        self.daily_start_value = self.total_value
        self.daily_pnl = 0.0
        self.daily_realized_pnl = 0.0
        self.daily_start_unrealized = self.unrealized_pnl
        self.api_calls_today = 0
        self.api_cost_today_usd = 0.0
        self.last_reset = datetime.now(UTC)
        logger.info("performance.daily_reset", start_value=self.daily_start_value)

    def should_alert_consecutive_losses(self, threshold: int) -> bool:
        """Return True if consecutive losses just hit the threshold (exact match to alert once)."""
        return self.consecutive_losses == threshold

    def restore_from_trades(self, closed_trades: list[dict]) -> None:
        """Restore performance metrics from historical closed trades.

        Each dict should have: realized_pnl, commission, strategy_name,
        entry_price, exit_price, created_at, closed_at.
        """
        for t in closed_trades:
            pnl = t.get("realized_pnl") or 0.0
            commission = t.get("commission") or 0.0
            strategy = t.get("strategy_name") or "unknown"

            self.trade_pnls.append(pnl)
            self.realized_pnl += pnl
            self.total_commission += commission
            self.total_trades += 1

            if pnl > 0:
                self.winning_trades += 1
            elif pnl < 0:
                self.losing_trades += 1

            # Per-strategy
            if strategy not in self.strategy_metrics:
                self.strategy_metrics[strategy] = StrategyMetrics()
            sm = self.strategy_metrics[strategy]
            sm.trades += 1
            sm.pnl += pnl
            sm.trade_pnls.append(pnl)
            if pnl > 0:
                sm.wins += 1
            elif pnl < 0:
                sm.losses += 1

            # Hold duration
            created = t.get("created_at")
            closed = t.get("closed_at")
            if created and closed:
                try:
                    minutes = (closed - created).total_seconds() / 60
                    if minutes > 0:
                        sm.hold_durations.append(minutes)
                except Exception:
                    pass

        self._update_drawdown()
        if closed_trades:
            logger.info(
                "performance.restored",
                trades=len(closed_trades),
                realized_pnl=round(self.realized_pnl, 2),
                win_rate=round(self.win_rate, 2),
            )

    def _update_drawdown(self) -> None:
        current = self.total_value
        if current > self.peak_value:
            self.peak_value = current
        drawdown = (self.peak_value - current) / self.peak_value * 100 if self.peak_value > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def get_strategy_breakdown(self) -> dict[str, dict]:
        """Return per-strategy performance breakdown."""
        return {
            name: metrics.to_dict()
            for name, metrics in self.strategy_metrics.items()
        }

    def get_symbol_breakdown(self) -> dict[str, dict]:
        """Return per-symbol performance breakdown."""
        return {
            name: metrics.to_dict()
            for name, metrics in self.symbol_metrics.items()
        }

    def get_underperforming_symbols(self, min_trades: int = 3, max_loss: float = -300.0) -> set[str]:
        """Return symbols that are consistently losing money.

        A symbol is flagged if it has >= min_trades and total P&L < max_loss.
        Used for dynamic blacklisting without data snooping.
        """
        bad_symbols = set()
        for symbol, metrics in self.symbol_metrics.items():
            if metrics.trades >= min_trades and metrics.pnl < max_loss:
                bad_symbols.add(symbol)
                logger.info(
                    "performance.underperforming_symbol",
                    symbol=symbol,
                    trades=metrics.trades,
                    pnl=round(metrics.pnl, 2),
                    win_rate=round(metrics.win_rate, 2),
                )
        return bad_symbols

    def to_dict(self) -> dict:
        return {
            "total_value": round(self.total_value, 2),
            "initial_capital": self.initial_capital,
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "daily_return_pct": round(self.daily_return_pct, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": (
                round(self.profit_factor, 2)
                if self.profit_factor != float("inf")
                else "\u221e"
            ),
            "max_drawdown": round(self.max_drawdown, 2),
            "total_commission": round(self.total_commission, 2),
            "strategy_breakdown": self.get_strategy_breakdown(),
            "api_calls_today": self.api_calls_today,
            "api_cost_today_usd": round(self.api_cost_today_usd, 6),
        }
