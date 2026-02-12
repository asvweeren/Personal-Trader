from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger()


@dataclass
class PerformanceTracker:
    """Tracks trading performance metrics in memory, periodically persisted to DB."""

    initial_capital: float = 5000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_commission: float = 0.0
    max_drawdown: float = 0.0
    peak_value: float = 5000.0
    consecutive_losses: int = 0
    trade_pnls: list[float] = field(default_factory=list)
    daily_start_value: float = 5000.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

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

    def record_trade(self, pnl: float, commission: float = 0.0) -> None:
        self.trade_pnls.append(pnl)
        self.realized_pnl += pnl
        self.total_commission += commission
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.losing_trades += 1
            self.consecutive_losses += 1

        self._update_drawdown()

    def update_unrealized(self, unrealized_pnl: float) -> None:
        self.unrealized_pnl = unrealized_pnl
        self._update_drawdown()

    def reset_daily(self) -> None:
        self.daily_start_value = self.total_value
        self.daily_pnl = 0.0
        self.last_reset = datetime.now(timezone.utc)
        logger.info("performance.daily_reset", start_value=self.daily_start_value)

    def should_alert_consecutive_losses(self, threshold: int) -> bool:
        """Return True if consecutive losses just hit the threshold (exact match to alert once)."""
        return self.consecutive_losses == threshold

    def _update_drawdown(self) -> None:
        current = self.total_value
        if current > self.peak_value:
            self.peak_value = current
        drawdown = (self.peak_value - current) / self.peak_value * 100 if self.peak_value > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

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
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor != float("inf") else "∞",
            "max_drawdown": round(self.max_drawdown, 2),
            "total_commission": round(self.total_commission, 2),
        }
