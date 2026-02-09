"""Risk management engine combining hard limits with AI-driven decisions."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import Portfolio
from app.models.risk_event import RiskEvent, RiskEventType, RiskEventSeverity
from app.risk.hard_limits import check_all_hard_limits, HardLimitCheck
from app.risk.market_hours import is_market_open, get_exchange_for_symbol, next_market_open
from app.risk.position_sizer import (
    calculate_position_size,
    calculate_sector_factor,
    get_sector,
)
from app.strategy.base import SignalAction, TradingSignal

logger = structlog.get_logger()


@dataclass
class RiskDecision:
    approved: bool
    signal: TradingSignal
    adjusted_quantity: int | None = None
    reason: str = ""


@dataclass
class HealthReport:
    healthy: bool
    checks: dict[str, bool]
    warnings: list[str]
    daily_loss_pct: float = 0.0
    cash_reserve_pct: float = 0.0
    position_count: int = 0
    max_drawdown_pct: float = 0.0
    sector_exposure: dict[str, float] = field(default_factory=dict)
    largest_position_pct: float = 0.0
    market_open: bool = False


class RiskManager:
    """Risk management engine combining hard limits with AI-driven decisions."""

    def __init__(
        self,
        max_daily_loss_pct: float = 5.0,
        max_position_pct: float = 20.0,
        max_open_positions: int = 10,
        min_cash_reserve_pct: float = 30.0,
        db: AsyncSession | None = None,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_pct = max_position_pct
        self.max_open_positions = max_open_positions
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self._db = db
        self.daily_start_value: float = 0.0
        self.daily_loss_triggered: bool = False
        # Drawdown tracking
        self._peak_value: float = 0.0
        self._max_drawdown_pct: float = 0.0

    def set_daily_start_value(self, value: float) -> None:
        self.daily_start_value = value
        self.daily_loss_triggered = False
        if value > self._peak_value:
            self._peak_value = value
        logger.info("risk.daily_reset", start_value=value)

    def update_peak_value(self, current_value: float) -> None:
        """Update peak portfolio value for drawdown tracking."""
        if current_value > self._peak_value:
            self._peak_value = current_value
        if self._peak_value > 0:
            drawdown = ((self._peak_value - current_value) / self._peak_value) * 100
            if drawdown > self._max_drawdown_pct:
                self._max_drawdown_pct = drawdown

    async def evaluate_signal(
        self, signal: TradingSignal, portfolio: Portfolio, estimated_price: float
    ) -> RiskDecision:
        """Evaluate a trading signal against all risk rules."""
        # Update drawdown tracking
        self.update_peak_value(portfolio.account_summary.total_value)

        if self.daily_loss_triggered:
            await self._log_risk_event(
                RiskEventType.SIGNAL_REJECTED,
                RiskEventSeverity.WARNING,
                signal.symbol,
                "Signal rejected: daily loss limit triggered",
                "Trading halted",
                portfolio,
            )
            return RiskDecision(
                approved=False, signal=signal,
                reason="Daily loss limit triggered - trading halted",
            )

        if signal.action == SignalAction.HOLD:
            return RiskDecision(approved=False, signal=signal, reason="HOLD signal, no action")

        # For SELL signals on existing positions, fewer checks needed
        if signal.action == SignalAction.SELL:
            has_position = any(p.symbol == signal.symbol for p in portfolio.positions)
            if not has_position:
                return RiskDecision(
                    approved=False, signal=signal, reason="No position to sell",
                )
            return RiskDecision(approved=True, signal=signal)

        # For BUY signals, run full risk checks
        quantity = calculate_position_size(
            portfolio=portfolio,
            price=estimated_price,
            max_position_pct=self.max_position_pct,
            confidence=signal.confidence,
            symbol=signal.symbol,
        )
        order_value = quantity * estimated_price

        if quantity <= 0:
            await self._log_risk_event(
                RiskEventType.SIGNAL_REJECTED,
                RiskEventSeverity.INFO,
                signal.symbol,
                f"Position size too small (confidence={signal.confidence:.2f})",
                "Signal rejected",
                portfolio,
            )
            return RiskDecision(
                approved=False, signal=signal, reason="Position size too small",
            )

        is_new_position = not any(p.symbol == signal.symbol for p in portfolio.positions)

        limit_check: HardLimitCheck = check_all_hard_limits(
            portfolio=portfolio,
            daily_start_value=self.daily_start_value,
            order_value=order_value,
            is_new_position=is_new_position,
            max_daily_loss_pct=self.max_daily_loss_pct,
            max_position_pct=self.max_position_pct,
            max_open_positions=self.max_open_positions,
            min_cash_reserve_pct=self.min_cash_reserve_pct,
            symbol=signal.symbol,
            check_hours=True,
        )

        if not limit_check.passed:
            for v in limit_check.violations:
                if "Daily loss" in v:
                    self.daily_loss_triggered = True
                    await self._log_risk_event(
                        RiskEventType.DAILY_LOSS_TRIGGERED,
                        RiskEventSeverity.CRITICAL,
                        signal.symbol,
                        v,
                        "All trading halted",
                        portfolio,
                    )
                elif "Market" in v and "closed" in v:
                    await self._log_risk_event(
                        RiskEventType.MARKET_CLOSED,
                        RiskEventSeverity.INFO,
                        signal.symbol,
                        v,
                        "Signal rejected",
                        portfolio,
                    )
                elif "position" in v.lower():
                    event_type = (
                        RiskEventType.MAX_POSITIONS_EXCEEDED
                        if "positions" in v.lower()
                        else RiskEventType.POSITION_SIZE_EXCEEDED
                    )
                    await self._log_risk_event(
                        event_type,
                        RiskEventSeverity.WARNING,
                        signal.symbol,
                        v,
                        "Signal rejected",
                        portfolio,
                    )
                elif "Cash reserve" in v:
                    await self._log_risk_event(
                        RiskEventType.CASH_RESERVE_LOW,
                        RiskEventSeverity.WARNING,
                        signal.symbol,
                        v,
                        "Signal rejected",
                        portfolio,
                    )

            return RiskDecision(
                approved=False,
                signal=signal,
                reason="; ".join(limit_check.violations),
            )

        logger.info(
            "risk.signal_approved",
            symbol=signal.symbol,
            action=signal.action.value,
            quantity=quantity,
        )

        return RiskDecision(approved=True, signal=signal, adjusted_quantity=quantity)

    async def check_portfolio_health(self, portfolio: Portfolio) -> HealthReport:
        """Check overall portfolio health with comprehensive metrics."""
        total_value = portfolio.account_summary.total_value
        cash = portfolio.account_summary.cash

        # Update drawdown tracking
        self.update_peak_value(total_value)

        # Daily loss
        daily_loss_pct = 0.0
        if self.daily_start_value > 0:
            daily_loss_pct = (
                (self.daily_start_value - total_value) / self.daily_start_value
            ) * 100

        cash_reserve_pct = (cash / total_value * 100) if total_value > 0 else 0.0
        position_count = len(portfolio.positions)

        # Sector exposure
        sector_exposure: dict[str, float] = {}
        for pos in portfolio.positions:
            sector = get_sector(pos.symbol)
            pct = (pos.market_value / total_value * 100) if total_value > 0 else 0.0
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + pct

        # Largest position
        largest_position_pct = 0.0
        if portfolio.positions and total_value > 0:
            largest = max(portfolio.positions, key=lambda p: p.market_value)
            largest_position_pct = (largest.market_value / total_value) * 100

        # Market status
        symbols = [p.symbol for p in portfolio.positions]
        market_open = False
        if symbols:
            exchanges = {get_exchange_for_symbol(s) for s in symbols}
            market_open = any(is_market_open(ex) for ex in exchanges)

        # Warnings
        warnings = []
        if daily_loss_pct > self.max_daily_loss_pct * 0.7:
            warnings.append(f"Daily loss at {daily_loss_pct:.1f}% - approaching limit")
        if cash_reserve_pct < self.min_cash_reserve_pct * 1.2:
            warnings.append(f"Cash reserve at {cash_reserve_pct:.1f}% - approaching minimum")
        if position_count >= self.max_open_positions - 1:
            warnings.append(f"Position count at {position_count} - near limit")
        if self._max_drawdown_pct > 10:
            warnings.append(f"Max drawdown at {self._max_drawdown_pct:.1f}%")
        for sector, pct in sector_exposure.items():
            if pct > 35:
                warnings.append(f"Sector '{sector}' concentration at {pct:.1f}%")
        if largest_position_pct > self.max_position_pct * 0.8:
            warnings.append(f"Largest position at {largest_position_pct:.1f}% of portfolio")

        # Log drawdown warning if significant
        if self._max_drawdown_pct > 15:
            await self._log_risk_event(
                RiskEventType.DRAWDOWN_WARNING,
                RiskEventSeverity.WARNING,
                None,
                f"Max drawdown reached {self._max_drawdown_pct:.1f}%",
                "Monitoring",
                portfolio,
            )

        checks = {
            "daily_loss_ok": daily_loss_pct < self.max_daily_loss_pct,
            "cash_reserve_ok": cash_reserve_pct >= self.min_cash_reserve_pct,
            "positions_ok": position_count < self.max_open_positions,
            "drawdown_ok": self._max_drawdown_pct < 20,
            "concentration_ok": all(pct < 40 for pct in sector_exposure.values()),
            "broker_connected": True,
        }

        return HealthReport(
            healthy=all(checks.values()),
            checks=checks,
            warnings=warnings,
            daily_loss_pct=round(daily_loss_pct, 2),
            cash_reserve_pct=round(cash_reserve_pct, 2),
            position_count=position_count,
            max_drawdown_pct=round(self._max_drawdown_pct, 2),
            sector_exposure={k: round(v, 2) for k, v in sector_exposure.items()},
            largest_position_pct=round(largest_position_pct, 2),
            market_open=market_open,
        )

    def get_limits(self) -> dict:
        return {
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_position_pct": self.max_position_pct,
            "max_open_positions": self.max_open_positions,
            "min_cash_reserve_pct": self.min_cash_reserve_pct,
        }

    async def _log_risk_event(
        self,
        event_type: RiskEventType,
        severity: RiskEventSeverity,
        symbol: str | None,
        description: str,
        action_taken: str,
        portfolio: Portfolio | None = None,
    ) -> None:
        """Persist a risk event to the database."""
        if severity == RiskEventSeverity.CRITICAL:
            logger.critical(
                "risk.event",
                event_type=event_type.value,
                severity=severity.value,
                symbol=symbol,
                description=description,
            )
        else:
            logger.warning(
                "risk.event",
                event_type=event_type.value,
                severity=severity.value,
                symbol=symbol,
                description=description,
            )

        if not self._db:
            return

        try:
            daily_loss_pct = 0.0
            portfolio_value = 0.0
            if portfolio:
                portfolio_value = portfolio.account_summary.total_value
                if self.daily_start_value > 0:
                    daily_loss_pct = (
                        (self.daily_start_value - portfolio_value) / self.daily_start_value
                    ) * 100

            event = RiskEvent(
                event_type=event_type.value,
                severity=severity.value,
                symbol=symbol,
                description=description,
                action_taken=action_taken,
                portfolio_value=portfolio_value,
                daily_loss_pct=round(daily_loss_pct, 2),
            )
            self._db.add(event)
            await self._db.flush()
        except Exception:
            logger.debug("risk.event_persist_error", event_type=event_type.value)
