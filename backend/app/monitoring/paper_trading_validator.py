"""Paper trading validation service for Phase 8.

Tracks daily performance, compares against backtest predictions, detects
anomalies, generates summary reports, and determines live-trading readiness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import structlog
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtest.metrics import calculate_metrics
from app.models.backtest_result import BacktestResult
from app.models.order import Order, OrderStatus
from app.models.portfolio_snapshot import PortfolioSnapshot
from app.models.risk_event import RiskEvent
from app.models.trade import Trade, TradeStatus
from app.models.validation_report import ValidationReport

logger = structlog.get_logger()


def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN/inf floats with None so values can be stored in JSONB."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _finite_or(value: Any, fallback: float) -> float:
    """Return value as float, or fallback if NaN/inf/None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(v) or math.isinf(v):
        return fallback
    return v


# ── Readiness thresholds ────────────────────────────────────

MIN_TRADING_DAYS = 20
MIN_SHARPE_RATIO = 0.5
MAX_DRAWDOWN_PCT = 15.0
MIN_WIN_RATE = 40.0
MIN_PROFIT_FACTOR = 1.2
MAX_BACKTEST_DEVIATION_PCT = 30.0


@dataclass
class ReadinessCheck:
    """Result of a single readiness criterion."""

    name: str
    passed: bool
    required: str
    actual: str
    detail: str = ""


@dataclass
class ReadinessResult:
    """Overall live-trading readiness assessment."""

    ready: bool
    checks: list[ReadinessCheck]
    trading_days: int
    summary: str

    def to_dict(self) -> dict[str, Any]:
        passed_count = sum(1 for c in self.checks if c.passed)
        total_count = len(self.checks)
        overall_score = passed_count / total_count if total_count > 0 else 0.0
        blockers = [c.detail for c in self.checks if not c.passed]
        return {
            "ready": self.ready,
            "trading_days": self.trading_days,
            "summary": self.summary,
            "passed_count": passed_count,
            "total_count": total_count,
            "overall_score": overall_score,
            "recommendation": self.summary,
            "blockers": blockers,
            "criteria": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "required": c.required,
                    "actual": c.actual,
                    "description": c.detail,
                }
                for c in self.checks
            ],
            # Legacy field
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "required": c.required,
                    "actual": c.actual,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


@dataclass
class AnomalyRecord:
    """A detected anomaly in paper-trading behaviour."""

    category: str  # slippage, fill_rate, drawdown, deviation, etc.
    severity: str  # info, warning, critical
    description: str
    value: float
    threshold: float


@dataclass
class DailySummary:
    """Aggregated metrics for a single trading day."""

    date: date
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    daily_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    portfolio_value: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_slippage: float = 0.0
    fill_rate: float = 0.0
    risk_events: int = 0
    anomalies: list[AnomalyRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "daily_pnl": round(self.daily_pnl, 2),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "portfolio_value": round(self.portfolio_value, 2),
            "win_rate": round(self.win_rate, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "profit_factor": round(self.profit_factor, 3),
            "avg_slippage": round(self.avg_slippage, 4),
            "fill_rate": round(self.fill_rate, 2),
            "risk_events": self.risk_events,
            "anomalies": [
                {
                    "category": a.category,
                    "severity": a.severity,
                    "description": a.description,
                    "value": round(a.value, 4),
                    "threshold": round(a.threshold, 4),
                }
                for a in self.anomalies
            ],
        }


class PaperTradingValidator:
    """Validates paper-trading results before allowing live trading.

    Responsibilities:
    - Track daily P&L, win rate, Sharpe ratio during paper trading.
    - Compare actual results versus backtest predictions.
    - Detect anomalies (excessive slippage, low fill rates, etc.).
    - Generate daily summary reports.
    - Determine readiness for live trading based on minimum criteria.
    """

    # Anomaly thresholds (configurable)
    MAX_AVG_SLIPPAGE_PCT: float = 0.5   # 0.5 % average slippage is suspicious
    MIN_FILL_RATE_PCT: float = 85.0      # Below 85 % fill rate is a concern
    DRAWDOWN_WARNING_PCT: float = 10.0   # Warn when drawdown approaches limit

    def __init__(
        self,
        db: AsyncSession,
        initial_capital: float = 5000.0,
    ) -> None:
        self._db = db
        self._initial_capital = initial_capital

    # ── Public API ──────────────────────────────────────────

    async def get_validation_status(self) -> dict[str, Any]:
        """Return the current state of paper-trading validation.

        Includes cumulative metrics, latest daily summary, and anomaly counts.
        """
        daily_summaries = await self._build_daily_summaries()
        trading_days = len(daily_summaries)

        if trading_days == 0:
            return {
                "is_active": False,
                "start_date": None,
                "days_elapsed": 0,
                "min_days_required": MIN_TRADING_DAYS,
                "total_trades": 0,
                "is_complete": False,
                "current_phase": "collecting_data",
                "progress_pct": 0.0,
                # Legacy fields for backwards compatibility
                "phase": "paper_trading",
                "trading_days": 0,
                "status": "collecting_data",
                "message": (
                    f"No trading days recorded yet. "
                    f"Need at least {MIN_TRADING_DAYS} days before readiness evaluation."
                ),
                "metrics": {},
                "anomaly_count": 0,
            }

        cumulative = self._calculate_cumulative_metrics(daily_summaries)
        latest = daily_summaries[-1]
        total_anomalies = sum(len(s.anomalies) for s in daily_summaries)
        critical_anomalies = sum(
            1
            for s in daily_summaries
            for a in s.anomalies
            if a.severity == "critical"
        )
        total_trades = cumulative.get("total_trades", 0)
        is_complete = trading_days >= MIN_TRADING_DAYS
        start_date = daily_summaries[0].date.isoformat() if daily_summaries[0].date else None

        return {
            "is_active": True,
            "start_date": start_date,
            "days_elapsed": trading_days,
            "min_days_required": MIN_TRADING_DAYS,
            "total_trades": total_trades,
            "is_complete": is_complete,
            "current_phase": "evaluation_ready" if is_complete else "validating",
            "progress_pct": round(min(trading_days / MIN_TRADING_DAYS * 100, 100), 1),
            # Legacy fields for backwards compatibility
            "phase": "paper_trading",
            "trading_days": trading_days,
            "status": "validating" if trading_days < MIN_TRADING_DAYS else "evaluation_ready",
            "message": (
                f"{trading_days} of {MIN_TRADING_DAYS} required trading days completed."
                if trading_days < MIN_TRADING_DAYS
                else "Sufficient data collected. Run readiness check."
            ),
            "metrics": cumulative,
            "latest_day": latest.to_dict(),
            "anomaly_count": total_anomalies,
            "critical_anomaly_count": critical_anomalies,
        }

    async def get_performance_report(
        self,
        period: str = "daily",
        days: int = 30,
    ) -> dict[str, Any]:
        """Generate a performance report for the requested period.

        Args:
            period: 'daily' or 'weekly'.
            days: Look-back window in calendar days.
        """
        daily_summaries = await self._build_daily_summaries(lookback_days=days)

        if not daily_summaries:
            return {
                "period": period,
                "days_requested": days,
                "days_available": 0,
                "summaries": [],
                "aggregate": {},
            }

        if period == "weekly":
            summaries_output = self._aggregate_weekly(daily_summaries)
        else:
            summaries_output = [s.to_dict() for s in daily_summaries]

        aggregate = self._calculate_cumulative_metrics(daily_summaries)

        return {
            "period": period,
            "days_requested": days,
            "days_available": len(daily_summaries),
            "summaries": summaries_output,
            "aggregate": aggregate,
        }

    async def is_ready_for_live(self) -> ReadinessResult:
        """Evaluate whether paper-trading results meet minimum criteria for live.

        Criteria:
        - At least 20 trading days of data.
        - Sharpe ratio > 0.5.
        - Max drawdown < 15 %.
        - Win rate > 40 %.
        - Profit factor > 1.2.
        - Results within 30 % of backtest predictions.
        """
        daily_summaries = await self._build_daily_summaries()
        trading_days = len(daily_summaries)
        checks: list[ReadinessCheck] = []

        # 1. Minimum trading days
        checks.append(
            ReadinessCheck(
                name="minimum_trading_days",
                passed=trading_days >= MIN_TRADING_DAYS,
                required=f">= {MIN_TRADING_DAYS}",
                actual=str(trading_days),
                detail=(
                    "Sufficient trading history."
                    if trading_days >= MIN_TRADING_DAYS
                    else f"Need {MIN_TRADING_DAYS - trading_days} more trading days."
                ),
            )
        )

        if trading_days == 0:
            return ReadinessResult(
                ready=False,
                checks=checks,
                trading_days=0,
                summary="No paper-trading data available. Cannot evaluate readiness.",
            )

        cumulative = self._calculate_cumulative_metrics(daily_summaries)
        sharpe = cumulative.get("sharpe_ratio", 0.0)
        max_dd = cumulative.get("max_drawdown_pct", 100.0)
        win_rate = cumulative.get("win_rate", 0.0)
        pf = cumulative.get("profit_factor", 0.0)

        # 2. Sharpe ratio
        checks.append(
            ReadinessCheck(
                name="sharpe_ratio",
                passed=sharpe > MIN_SHARPE_RATIO,
                required=f"> {MIN_SHARPE_RATIO}",
                actual=f"{sharpe:.3f}",
                detail=(
                    "Risk-adjusted return is acceptable."
                    if sharpe > MIN_SHARPE_RATIO
                    else "Risk-adjusted return is too low."
                ),
            )
        )

        # 3. Max drawdown
        checks.append(
            ReadinessCheck(
                name="max_drawdown",
                passed=max_dd < MAX_DRAWDOWN_PCT,
                required=f"< {MAX_DRAWDOWN_PCT}%",
                actual=f"{max_dd:.2f}%",
                detail=(
                    "Drawdown within acceptable limits."
                    if max_dd < MAX_DRAWDOWN_PCT
                    else "Drawdown exceeds acceptable limits."
                ),
            )
        )

        # 4. Win rate
        checks.append(
            ReadinessCheck(
                name="win_rate",
                passed=win_rate > MIN_WIN_RATE,
                required=f"> {MIN_WIN_RATE}%",
                actual=f"{win_rate:.2f}%",
                detail=(
                    "Win rate is acceptable."
                    if win_rate > MIN_WIN_RATE
                    else "Win rate is too low."
                ),
            )
        )

        # 5. Profit factor
        checks.append(
            ReadinessCheck(
                name="profit_factor",
                passed=pf > MIN_PROFIT_FACTOR,
                required=f"> {MIN_PROFIT_FACTOR}",
                actual=f"{pf:.3f}",
                detail=(
                    "Profit factor is acceptable."
                    if pf > MIN_PROFIT_FACTOR
                    else "Profit factor is too low."
                ),
            )
        )

        # 6. Backtest deviation
        deviation = await self._calculate_backtest_deviation(cumulative)
        deviation_ok = deviation is None or abs(deviation) < MAX_BACKTEST_DEVIATION_PCT
        checks.append(
            ReadinessCheck(
                name="backtest_deviation",
                passed=deviation_ok,
                required=f"< {MAX_BACKTEST_DEVIATION_PCT}%",
                actual=f"{deviation:.2f}%" if deviation is not None else "N/A (no backtest)",
                detail=(
                    "Paper results are close to backtest predictions."
                    if deviation_ok
                    else "Paper results deviate significantly from backtest predictions."
                ),
            )
        )

        all_passed = all(c.passed for c in checks)
        failed = [c.name for c in checks if not c.passed]

        if all_passed:
            summary = "All readiness criteria met. System is ready for live trading."
        else:
            summary = f"Not ready for live trading. Failed checks: {', '.join(failed)}."

        logger.info(
            "validation.readiness_check",
            ready=all_passed,
            trading_days=trading_days,
            failed_checks=failed,
        )

        return ReadinessResult(
            ready=all_passed,
            checks=checks,
            trading_days=trading_days,
            summary=summary,
        )

    async def generate_daily_report(self, report_date: date | None = None) -> ValidationReport:
        """Generate and persist a daily validation report.

        Args:
            report_date: The date to report on. Defaults to today (UTC).
        """
        if report_date is None:
            report_date = datetime.now(UTC).date()

        summary = await self._build_single_day_summary(report_date)
        all_summaries = await self._build_daily_summaries()
        cumulative = self._calculate_cumulative_metrics(all_summaries) if all_summaries else {}
        deviation = await self._calculate_backtest_deviation(cumulative) if cumulative else None

        # Count risk events for the day
        day_start = datetime.combine(report_date, datetime.min.time()).replace(tzinfo=UTC)
        day_end = day_start + timedelta(days=1)
        risk_count_result = await self._db.execute(
            select(func.count(RiskEvent.id)).where(
                RiskEvent.timestamp >= day_start,
                RiskEvent.timestamp < day_end,
            )
        )
        risk_count = risk_count_result.scalar() or 0

        # Build risk events summary
        risk_events_result = await self._db.execute(
            select(RiskEvent)
            .where(RiskEvent.timestamp >= day_start, RiskEvent.timestamp < day_end)
            .order_by(desc(RiskEvent.timestamp))
            .limit(20)
        )
        risk_events = risk_events_result.scalars().all()
        risk_summary = [
            {
                "type": e.event_type,
                "severity": e.severity,
                "symbol": e.symbol,
                "description": e.description,
            }
            for e in risk_events
        ]

        # Build narrative summary
        narrative_parts = [
            f"Paper Trading Daily Report - {report_date.isoformat()}",
            (
                f"Trades: {summary.total_trades} "
                f"(W:{summary.winning_trades} / "
                f"L:{summary.losing_trades})"
            ),
            f"Daily P&L: ${summary.daily_pnl:,.2f}",
            f"Cumulative P&L: ${summary.cumulative_pnl:,.2f}",
            f"Portfolio Value: ${summary.portfolio_value:,.2f}",
            f"Win Rate: {summary.win_rate:.1f}%",
            f"Fill Rate: {summary.fill_rate:.1f}%",
        ]
        if summary.anomalies:
            narrative_parts.append(
                f"Anomalies detected: {len(summary.anomalies)} "
                f"({', '.join(a.category for a in summary.anomalies)})"
            )
        if risk_count > 0:
            narrative_parts.append(f"Risk events: {risk_count}")

        report = ValidationReport(
            report_date=day_start,
            report_type="daily",
            total_trades=summary.total_trades,
            winning_trades=summary.winning_trades,
            losing_trades=summary.losing_trades,
            win_rate=summary.win_rate,
            daily_pnl=summary.daily_pnl,
            cumulative_pnl=summary.cumulative_pnl,
            sharpe_ratio=_finite_or(cumulative.get("sharpe_ratio"), 0.0),
            max_drawdown_pct=_finite_or(cumulative.get("max_drawdown_pct"), 0.0),
            profit_factor=_finite_or(cumulative.get("profit_factor"), 0.0),
            portfolio_value=summary.portfolio_value,
            backtest_deviation_pct=deviation,
            anomalies=_json_safe([a.__dict__ for a in summary.anomalies]) if summary.anomalies else None,
            metrics_snapshot=_json_safe(cumulative),
            risk_events_count=risk_count,
            risk_events_summary=_json_safe(risk_summary) if risk_summary else None,
            summary_text="\n".join(narrative_parts),
        )

        self._db.add(report)
        await self._db.commit()
        await self._db.refresh(report)

        logger.info(
            "validation.daily_report_generated",
            report_id=report.id,
            date=report_date.isoformat(),
            trades=summary.total_trades,
            daily_pnl=summary.daily_pnl,
            anomalies=len(summary.anomalies),
        )

        return report

    # ── Internal helpers ────────────────────────────────────

    async def _build_daily_summaries(
        self,
        lookback_days: int | None = None,
    ) -> list[DailySummary]:
        """Build per-day summaries from closed trades and portfolio snapshots.

        If lookback_days is None, returns all available data.
        """
        cutoff = None
        if lookback_days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        # Fetch closed trades
        trade_query = (
            select(Trade)
            .where(Trade.status == TradeStatus.CLOSED)
            .order_by(Trade.closed_at)
        )
        if cutoff:
            trade_query = trade_query.where(Trade.closed_at >= cutoff)
        trade_result = await self._db.execute(trade_query)
        trades = trade_result.scalars().all()

        # Fetch portfolio snapshots for end-of-day values
        snap_query = select(PortfolioSnapshot).order_by(PortfolioSnapshot.timestamp)
        if cutoff:
            snap_query = snap_query.where(PortfolioSnapshot.timestamp >= cutoff)
        snap_result = await self._db.execute(snap_query)
        snapshots = snap_result.scalars().all()

        # Fetch orders for slippage / fill rate calculations
        order_query = select(Order).order_by(Order.submitted_at)
        if cutoff:
            order_query = order_query.where(Order.submitted_at >= cutoff)
        order_result = await self._db.execute(order_query)
        orders = order_result.scalars().all()

        # Fetch risk events
        risk_query = select(RiskEvent).order_by(RiskEvent.timestamp)
        if cutoff:
            risk_query = risk_query.where(RiskEvent.timestamp >= cutoff)
        risk_result = await self._db.execute(risk_query)
        risk_events = risk_result.scalars().all()

        # Group trades by day (closed_at date)
        trades_by_day: dict[date, list[Trade]] = {}
        for t in trades:
            if t.closed_at is None:
                continue
            day = t.closed_at.date()
            trades_by_day.setdefault(day, []).append(t)

        # Group snapshots by day (last snapshot per day)
        snap_by_day: dict[date, PortfolioSnapshot] = {}
        for s in snapshots:
            if s.timestamp:
                snap_by_day[s.timestamp.date()] = s  # last one wins

        # Group orders by day
        orders_by_day: dict[date, list[Order]] = {}
        for o in orders:
            if o.submitted_at:
                day = o.submitted_at.date()
                orders_by_day.setdefault(day, []).append(o)

        # Group risk events by day
        risk_by_day: dict[date, list[RiskEvent]] = {}
        for r in risk_events:
            if r.timestamp:
                day = r.timestamp.date()
                risk_by_day.setdefault(day, []).append(r)

        # Collect all relevant days (union of trades and snapshots)
        all_days = sorted(set(trades_by_day.keys()) | set(snap_by_day.keys()))
        if not all_days:
            return []

        # Build daily summaries
        cumulative_pnl = 0.0
        all_trade_pnls: list[float] = []
        equity_curve: list[float] = [self._initial_capital]
        summaries: list[DailySummary] = []

        for day in all_days:
            day_trades = trades_by_day.get(day, [])
            day_orders = orders_by_day.get(day, [])
            day_risk = risk_by_day.get(day, [])
            snapshot = snap_by_day.get(day)

            winning = [t for t in day_trades if (t.realized_pnl or 0) > 0]
            losing = [t for t in day_trades if (t.realized_pnl or 0) < 0]
            daily_pnl = sum(t.realized_pnl or 0 for t in day_trades)
            cumulative_pnl += daily_pnl

            total = len(day_trades)
            _wr = (len(winning) / total * 100) if total > 0 else 0.0

            # Collect P&Ls for cumulative metrics
            for t in day_trades:
                all_trade_pnls.append(t.realized_pnl or 0)

            # Portfolio value from snapshot or calculated
            pv = snapshot.total_value if snapshot else (self._initial_capital + cumulative_pnl)
            equity_curve.append(pv)

            # Slippage calculation (compare limit/stop price to filled price)
            slippages = self._calculate_slippages(day_orders)
            avg_slippage = float(np.mean(slippages)) if slippages else 0.0

            # Fill rate
            total_orders = len(day_orders)
            filled_orders = len([
                o for o in day_orders
                if o.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
            ])
            fill_rate = (filled_orders / total_orders * 100) if total_orders > 0 else 100.0

            # Calculate rolling cumulative metrics using backtest.metrics
            cum_metrics = {}
            if len(equity_curve) >= 2 and all_trade_pnls:
                cum_metrics = calculate_metrics(
                    trade_pnls=all_trade_pnls,
                    equity_curve=equity_curve,
                    initial_capital=self._initial_capital,
                )

            sharpe = cum_metrics.get("sharpe_ratio", 0.0)
            max_dd = cum_metrics.get("max_drawdown_pct", 0.0)
            gross_profit = sum(p for p in all_trade_pnls if p > 0)
            gross_loss = abs(sum(p for p in all_trade_pnls if p < 0))
            pf = (
                (gross_profit / gross_loss) if gross_loss > 0
                else (
                    float("inf") if gross_profit > 0
                    else 0.0
                )
            )
            cumulative_wr = (
                (len([p for p in all_trade_pnls if p > 0]) / len(all_trade_pnls) * 100)
                if all_trade_pnls
                else 0.0
            )

            # Detect anomalies
            anomalies = self._detect_anomalies(
                avg_slippage=avg_slippage,
                fill_rate=fill_rate,
                max_drawdown=max_dd,
                daily_pnl=daily_pnl,
                portfolio_value=pv,
            )

            summary = DailySummary(
                date=day,
                total_trades=total,
                winning_trades=len(winning),
                losing_trades=len(losing),
                daily_pnl=daily_pnl,
                cumulative_pnl=cumulative_pnl,
                portfolio_value=pv,
                win_rate=cumulative_wr,
                sharpe_ratio=sharpe,
                max_drawdown_pct=max_dd,
                profit_factor=pf if pf != float("inf") else 999.99,
                avg_slippage=avg_slippage,
                fill_rate=fill_rate,
                risk_events=len(day_risk),
                anomalies=anomalies,
            )
            summaries.append(summary)

        return summaries

    async def _build_single_day_summary(self, report_date: date) -> DailySummary:
        """Build a summary for a single date.

        Re-uses _build_daily_summaries and filters to the requested date.
        If the date has no data, returns an empty summary.
        """
        summaries = await self._build_daily_summaries()
        for s in summaries:
            if s.date == report_date:
                return s
        # Return empty if no data for that date
        return DailySummary(date=report_date)

    def _calculate_cumulative_metrics(
        self,
        daily_summaries: list[DailySummary],
    ) -> dict[str, Any]:
        """Calculate aggregate metrics across all daily summaries."""
        if not daily_summaries:
            return {}

        latest = daily_summaries[-1]
        total_trades = sum(s.total_trades for s in daily_summaries)
        total_winning = sum(s.winning_trades for s in daily_summaries)
        total_losing = sum(s.losing_trades for s in daily_summaries)

        return {
            "trading_days": len(daily_summaries),
            "total_trades": total_trades,
            "winning_trades": total_winning,
            "losing_trades": total_losing,
            "win_rate": latest.win_rate,
            "cumulative_pnl": latest.cumulative_pnl,
            "portfolio_value": latest.portfolio_value,
            "sharpe_ratio": latest.sharpe_ratio,
            "max_drawdown_pct": latest.max_drawdown_pct,
            "profit_factor": latest.profit_factor,
            "avg_daily_pnl": round(
                sum(s.daily_pnl for s in daily_summaries) / len(daily_summaries), 2
            ),
            "best_day": round(max(s.daily_pnl for s in daily_summaries), 2),
            "worst_day": round(min(s.daily_pnl for s in daily_summaries), 2),
            "avg_fill_rate": round(
                sum(s.fill_rate for s in daily_summaries) / len(daily_summaries), 2
            ),
            "total_risk_events": sum(s.risk_events for s in daily_summaries),
            "total_anomalies": sum(len(s.anomalies) for s in daily_summaries),
        }

    async def _calculate_backtest_deviation(
        self,
        cumulative_metrics: dict[str, Any],
    ) -> float | None:
        """Compare paper-trading results to the most recent backtest.

        Returns the percentage deviation of cumulative return relative to
        the backtest's total return, or None if no backtest is available.
        """
        if not cumulative_metrics:
            return None

        # Fetch most recent completed backtest
        result = await self._db.execute(
            select(BacktestResult)
            .where(BacktestResult.metrics.isnot(None))
            .order_by(desc(BacktestResult.created_at))
            .limit(1)
        )
        bt = result.scalar_one_or_none()

        if bt is None or bt.metrics is None:
            return None

        # Backtest may have "status": "error" or "status": "running"
        if bt.metrics.get("status") in ("error", "running"):
            return None

        bt_return = bt.metrics.get("total_return_pct")
        if bt_return is None:
            return None

        # Paper trading return %
        paper_pnl = cumulative_metrics.get("cumulative_pnl", 0.0)
        paper_return = (
            (paper_pnl / self._initial_capital) * 100
            if self._initial_capital > 0 else 0.0
        )

        if bt_return == 0:
            return abs(paper_return) * 100  # Large deviation if backtest was zero

        deviation_pct = ((paper_return - bt_return) / abs(bt_return)) * 100

        logger.debug(
            "validation.backtest_deviation",
            paper_return_pct=round(paper_return, 2),
            backtest_return_pct=round(bt_return, 2),
            deviation_pct=round(deviation_pct, 2),
        )

        return round(deviation_pct, 2)

    @staticmethod
    def _calculate_slippages(orders: list[Order]) -> list[float]:
        """Calculate slippage percentages for filled orders.

        Slippage is defined as the difference between the expected price
        (limit_price or stop_price) and the actual filled price, as a
        percentage of the expected price.  Market orders without a
        reference price are excluded.
        """
        slippages = []
        for o in orders:
            if o.filled_price is None:
                continue
            expected = o.limit_price or o.stop_price
            if expected is None or expected == 0:
                continue
            slippage = abs(o.filled_price - expected) / expected * 100
            slippages.append(slippage)
        return slippages

    def _detect_anomalies(
        self,
        avg_slippage: float,
        fill_rate: float,
        max_drawdown: float,
        daily_pnl: float,
        portfolio_value: float,
    ) -> list[AnomalyRecord]:
        """Detect anomalies in daily trading metrics."""
        anomalies: list[AnomalyRecord] = []

        # High slippage
        if avg_slippage > self.MAX_AVG_SLIPPAGE_PCT:
            severity = "critical" if avg_slippage > self.MAX_AVG_SLIPPAGE_PCT * 2 else "warning"
            anomalies.append(
                AnomalyRecord(
                    category="slippage",
                    severity=severity,
                    description=(
                        f"Average slippage of {avg_slippage:.3f}% exceeds "
                        f"threshold of {self.MAX_AVG_SLIPPAGE_PCT}%."
                    ),
                    value=avg_slippage,
                    threshold=self.MAX_AVG_SLIPPAGE_PCT,
                )
            )

        # Low fill rate
        if fill_rate < self.MIN_FILL_RATE_PCT:
            severity = "critical" if fill_rate < self.MIN_FILL_RATE_PCT * 0.7 else "warning"
            anomalies.append(
                AnomalyRecord(
                    category="fill_rate",
                    severity=severity,
                    description=(
                        f"Fill rate of {fill_rate:.1f}% is below "
                        f"minimum threshold of {self.MIN_FILL_RATE_PCT}%."
                    ),
                    value=fill_rate,
                    threshold=self.MIN_FILL_RATE_PCT,
                )
            )

        # Drawdown approaching limit
        if max_drawdown > self.DRAWDOWN_WARNING_PCT:
            severity = "critical" if max_drawdown > MAX_DRAWDOWN_PCT else "warning"
            anomalies.append(
                AnomalyRecord(
                    category="drawdown",
                    severity=severity,
                    description=(
                        f"Maximum drawdown of {max_drawdown:.2f}% "
                        f"{'exceeds' if max_drawdown > MAX_DRAWDOWN_PCT else 'is approaching'} "
                        f"the {MAX_DRAWDOWN_PCT}% limit."
                    ),
                    value=max_drawdown,
                    threshold=MAX_DRAWDOWN_PCT,
                )
            )

        # Large single-day loss (> 3% of portfolio)
        if portfolio_value > 0:
            daily_loss_pct = abs(daily_pnl) / portfolio_value * 100
            if daily_pnl < 0 and daily_loss_pct > 3.0:
                severity = "critical" if daily_loss_pct > 5.0 else "warning"
                anomalies.append(
                    AnomalyRecord(
                        category="daily_loss",
                        severity=severity,
                        description=(
                            f"Single-day loss of {daily_loss_pct:.2f}% of portfolio "
                            f"exceeds 3% warning threshold."
                        ),
                        value=daily_loss_pct,
                        threshold=3.0,
                    )
                )

        return anomalies

    def _aggregate_weekly(
        self,
        daily_summaries: list[DailySummary],
    ) -> list[dict[str, Any]]:
        """Roll up daily summaries into weekly buckets (Mon-Sun)."""
        if not daily_summaries:
            return []

        weeks: dict[str, list[DailySummary]] = {}
        for s in daily_summaries:
            # ISO week: Monday is day 1
            week_start = s.date - timedelta(days=s.date.weekday())
            week_key = week_start.isoformat()
            weeks.setdefault(week_key, []).append(s)

        result = []
        for week_start, days in sorted(weeks.items()):
            week_end = (date.fromisoformat(week_start) + timedelta(days=6)).isoformat()
            total_trades = sum(d.total_trades for d in days)
            winning = sum(d.winning_trades for d in days)
            losing = sum(d.losing_trades for d in days)
            weekly_pnl = sum(d.daily_pnl for d in days)

            result.append({
                "week_start": week_start,
                "week_end": week_end,
                "trading_days": len(days),
                "total_trades": total_trades,
                "winning_trades": winning,
                "losing_trades": losing,
                "win_rate": round(winning / total_trades * 100, 2) if total_trades > 0 else 0.0,
                "weekly_pnl": round(weekly_pnl, 2),
                "cumulative_pnl": round(days[-1].cumulative_pnl, 2),
                "portfolio_value": round(days[-1].portfolio_value, 2),
                "sharpe_ratio": round(days[-1].sharpe_ratio, 3),
                "max_drawdown_pct": round(days[-1].max_drawdown_pct, 2),
                "anomalies": sum(len(d.anomalies) for d in days),
            })

        return result
