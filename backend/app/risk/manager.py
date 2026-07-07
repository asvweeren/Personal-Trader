"""Risk management engine combining hard limits with AI-driven decisions."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import Portfolio
from app.config import settings
from app.models.risk_event import RiskEvent, RiskEventSeverity, RiskEventType
from app.risk.ai_sizing import AISizingAdvisor
from app.risk.hard_limits import HardLimitCheck, check_all_hard_limits
from app.risk.market_hours import get_exchange_for_symbol, is_market_open
from app.risk.position_sizer import (
    calculate_position_size,
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
    ai_modifier: float = 1.0


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
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0


class RiskManager:
    """Risk management engine combining hard limits with AI-driven decisions."""

    def __init__(
        self,
        max_daily_loss_pct: float = 5.0,
        max_position_pct: float = 20.0,
        max_open_positions: int = 10,
        min_cash_reserve_pct: float = 30.0,
        db: AsyncSession | None = None,
        session_factory=None,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_pct = max_position_pct
        self.max_open_positions = max_open_positions
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self._db = db
        # Independent session factory so risk events persist even when the
        # manager is a long-lived singleton with no request-scoped session.
        self._session_factory = session_factory
        self.daily_start_value: float = 0.0
        self.daily_loss_triggered: bool = False
        # Hourly loss tracking — pause trading if losses accelerate
        self._hourly_loss_paused_until: datetime | None = None
        self._hourly_snapshots: list[tuple[datetime, float]] = []  # (time, value)
        # Drawdown tracking
        self._peak_value: float = 0.0
        self._max_drawdown_pct: float = 0.0
        # AI sizing advisor
        self._ai_advisor: AISizingAdvisor | None = None
        if settings.ai_sizing_enabled and settings.anthropic_api_key:
            self._ai_advisor = AISizingAdvisor()

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
        self,
        signal: TradingSignal,
        portfolio: Portfolio,
        estimated_price: float,
        correlation_matrix: dict[tuple[str, str], float] | None = None,
        regime=None,
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

        # Hourly loss circuit breaker — pause new BUY if losses are accelerating
        if signal.action == SignalAction.BUY:
            now = datetime.now(UTC)
            if self._hourly_loss_paused_until and now < self._hourly_loss_paused_until:
                return RiskDecision(
                    approved=False, signal=signal,
                    reason=f"Hourly loss limit hit - paused until {self._hourly_loss_paused_until.strftime('%H:%M')} UTC",
                )
            current_value = portfolio.account_summary.total_value
            self._hourly_snapshots.append((now, current_value))
            # Prune snapshots older than 1 hour
            cutoff = now.timestamp() - 3600
            self._hourly_snapshots = [
                (t, v) for t, v in self._hourly_snapshots
                if t.timestamp() > cutoff
            ]
            if len(self._hourly_snapshots) >= 2:
                hour_ago_value = self._hourly_snapshots[0][1]
                if hour_ago_value > 0:
                    hourly_loss_pct = ((hour_ago_value - current_value) / hour_ago_value) * 100
                    if hourly_loss_pct > settings.max_hourly_loss_pct:
                        from datetime import timedelta
                        self._hourly_loss_paused_until = now + timedelta(hours=1)
                        logger.warning(
                            "risk.hourly_loss_pause",
                            hourly_loss_pct=round(hourly_loss_pct, 2),
                            limit=settings.max_hourly_loss_pct,
                            paused_until=self._hourly_loss_paused_until.isoformat(),
                        )
                        return RiskDecision(
                            approved=False, signal=signal,
                            reason=f"Hourly loss {hourly_loss_pct:.1f}% exceeds {settings.max_hourly_loss_pct}% - paused 1h",
                        )

        if signal.action == SignalAction.HOLD:
            return RiskDecision(approved=False, signal=signal, reason="HOLD signal, no action")

        # For SELL signals: close existing position (auto-approve) or open short
        if signal.action == SignalAction.SELL:
            has_position = any(p.symbol == signal.symbol for p in portfolio.positions)
            if has_position:
                return RiskDecision(approved=True, signal=signal)
            # Opening a short position — run through same risk checks as BUY
            if not settings.enable_short_selling:
                return RiskDecision(
                    approved=False, signal=signal, reason="Short selling disabled",
                )
            # Check short exposure limit
            short_exposure = sum(
                abs(p.market_value) for p in portfolio.positions if p.quantity < 0
            )
            max_short = portfolio.account_summary.total_value * settings.max_short_exposure_pct / 100
            if short_exposure >= max_short:
                return RiskDecision(
                    approved=False, signal=signal,
                    reason=f"Short exposure {short_exposure:.0f} >= max {max_short:.0f}",
                )
            # Fall through to position sizing below (same as BUY)

        # Regime-aware position sizing
        effective_max_position_pct = self.max_position_pct
        if regime is not None:
            try:
                from app.strategy.regime import MarketRegime
                if regime.regime == MarketRegime.HIGH_VOLATILITY:
                    effective_max_position_pct *= 0.6  # 40% reduction
                elif regime.regime == MarketRegime.RANGING:
                    effective_max_position_pct *= 0.8  # 20% reduction
            except Exception:
                pass

        # AI sizing modifier
        ai_modifier = 1.0
        if self._ai_advisor:
            try:
                portfolio_summary = {
                    "total_value": portfolio.account_summary.total_value,
                    "cash": portfolio.account_summary.cash,
                    "positions": len(portfolio.positions),
                    "unrealized_pnl": portfolio.account_summary.unrealized_pnl,
                }
                features = (
                    signal.features_snapshot if hasattr(signal, "features_snapshot") else None
                )
                sentiment_data = None
                if features and isinstance(features, dict):
                    sentiment_data = {
                        k: v for k, v in features.items()
                        if "sentiment" in k.lower()
                    } or None

                ai_result = await self._ai_advisor.get_modifier(
                    symbol=signal.symbol,
                    signal_confidence=signal.confidence,
                    strategy_name=signal.strategy_name,
                    portfolio_summary=portfolio_summary,
                    features=features if isinstance(features, dict) else None,
                    sentiment=sentiment_data,
                )
                ai_modifier = ai_result.modifier
                logger.info(
                    "risk.ai_sizing",
                    symbol=signal.symbol,
                    modifier=ai_result.modifier,
                    reasoning=ai_result.reasoning,
                    risk_factors=ai_result.risk_factors,
                )
            except Exception:
                logger.warning("risk.ai_sizing_error", symbol=signal.symbol)
                ai_modifier = 1.0

        # Save AI sizing data in features_snapshot for persistence
        if ai_modifier != 1.0:
            if signal.features_snapshot is None:
                signal.features_snapshot = {}
            if isinstance(signal.features_snapshot, dict):
                signal.features_snapshot["ai_sizing_modifier"] = ai_result.modifier
                signal.features_snapshot["ai_sizing_reasoning"] = ai_result.reasoning
                signal.features_snapshot["ai_sizing_risk_factors"] = ai_result.risk_factors

        # Extract ATR-based volatility from signal features for position sizing
        volatility = None
        if signal.features_snapshot and isinstance(signal.features_snapshot, dict):
            atr = signal.features_snapshot.get("atr_14")
            if atr and estimated_price > 0:
                volatility = atr / estimated_price  # Normalized ATR as volatility measure

        # For BUY signals, run full risk checks
        quantity = calculate_position_size(
            portfolio=portfolio,
            price=estimated_price,
            max_position_pct=effective_max_position_pct,
            confidence=signal.confidence,
            volatility=volatility,
            symbol=signal.symbol,
            correlation_matrix=correlation_matrix,
            ai_modifier=ai_modifier,
        )
        order_value = quantity * estimated_price

        # Aggregate exposure check: total open notional + new order must not exceed limit
        total_value = portfolio.account_summary.total_value
        if total_value > 0:
            current_exposure = sum(
                abs(p.market_value) for p in portfolio.positions
            )
            new_exposure = current_exposure + order_value
            max_exposure = total_value * (settings.max_total_exposure_pct / 100.0)
            if new_exposure > max_exposure:
                return RiskDecision(
                    approved=False,
                    signal=signal,
                    reason=f"Total exposure {new_exposure:.0f} would exceed "
                           f"{settings.max_total_exposure_pct}% limit ({max_exposure:.0f})",
                    quantity=0,
                )

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

        return RiskDecision(
            approved=True, signal=signal, adjusted_quantity=quantity, ai_modifier=ai_modifier,
        )

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
            market_open = any(is_market_open(ex, include_extended=settings.extended_hours_enabled) for ex in exchanges)

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

        if not self._db and not self._session_factory:
            logger.warning(
                "risk.event_not_persisted",
                event_type=event_type.value,
                reason="no db session or session_factory configured",
            )
            return

        daily_loss_pct = 0.0
        portfolio_value = 0.0
        if portfolio:
            portfolio_value = portfolio.account_summary.total_value
            if self.daily_start_value > 0:
                daily_loss_pct = (
                    (self.daily_start_value - portfolio_value) / self.daily_start_value
                ) * 100

        def _build_event() -> RiskEvent:
            return RiskEvent(
                event_type=event_type.value,
                severity=severity.value,
                symbol=symbol,
                description=description,
                action_taken=action_taken,
                portfolio_value=portfolio_value,
                daily_loss_pct=round(daily_loss_pct, 2),
            )

        # Prefer an independent short-lived session so the event is committed
        # even if the caller's transaction later rolls back. Fall back to the
        # request-scoped session (flush + commit) when no factory is available.
        try:
            if self._session_factory is not None:
                async with self._session_factory() as session:
                    session.add(_build_event())
                    await session.commit()
            else:
                self._db.add(_build_event())
                await self._db.commit()
        except Exception:
            logger.exception(
                "risk.event_persist_error", event_type=event_type.value
            )
