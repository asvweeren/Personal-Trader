"""Daily report generator for paper trading validation.

Summarises trades, P&L, and risk events.  Calculates rolling performance
metrics.  Can be scheduled via APScheduler and persists reports to the
database through :class:`PaperTradingValidator`.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import structlog
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.database import async_session as session_factory
from app.models.validation_report import ValidationReport
from app.monitoring.alerts import send_alert
from app.monitoring.paper_trading_validator import PaperTradingValidator

logger = structlog.get_logger()


class DailyReporter:
    """Generates and distributes daily paper-trading reports.

    Responsibilities:
    - Orchestrate report generation via :class:`PaperTradingValidator`.
    - Retrieve historical reports for comparison.
    - Send alert notifications for critical anomalies.
    - Provide a scheduler-compatible entry point for APScheduler.
    """

    def __init__(
        self,
        db: AsyncSession,
        initial_capital: float | None = None,
    ) -> None:
        self._db = db
        self._initial_capital = initial_capital or settings.initial_capital
        self._validator = PaperTradingValidator(
            db=self._db,
            initial_capital=self._initial_capital,
        )

    # ── Main entry points ───────────────────────────────────

    async def generate_and_store_report(
        self,
        report_date: date | None = None,
    ) -> ValidationReport:
        """Generate a daily report, persist it, and send alerts if needed.

        Args:
            report_date: The date to report on. Defaults to today (UTC).

        Returns:
            The persisted :class:`ValidationReport` instance.
        """
        if report_date is None:
            report_date = datetime.now(timezone.utc).date()

        logger.info("daily_reporter.generating", date=report_date.isoformat())

        report = await self._validator.generate_daily_report(report_date)

        # Check for critical anomalies and notify
        if report.anomalies:
            critical = [a for a in report.anomalies if a.get("severity") == "critical"]
            if critical:
                await self._send_anomaly_alert(report, critical)

        # Check readiness and log
        readiness = await self._validator.is_ready_for_live()
        logger.info(
            "daily_reporter.readiness",
            ready=readiness.ready,
            trading_days=readiness.trading_days,
        )

        # Send Telegram + email alert when system is ready for live trading
        if readiness.ready:
            passed = sum(1 for c in readiness.checks if c.passed)
            total = len(readiness.checks)
            await send_alert(
                "Paper Trading Validation Complete",
                f"The system has passed all readiness criteria after "
                f"{readiness.trading_days} trading days.\n\n"
                f"Score: {passed}/{total} checks passed\n"
                f"Summary: {readiness.summary}\n\n"
                f"You can now consider switching to live trading.",
                critical=True,
            )

        logger.info(
            "daily_reporter.complete",
            report_id=report.id,
            date=report_date.isoformat(),
            daily_pnl=report.daily_pnl,
            cumulative_pnl=report.cumulative_pnl,
            anomalies=len(report.anomalies) if report.anomalies else 0,
        )

        # Send daily summary via Telegram
        await self._send_daily_summary(report, readiness)

        return report

    async def get_recent_reports(
        self,
        limit: int = 30,
        report_type: str = "daily",
    ) -> list[dict[str, Any]]:
        """Fetch recent reports from the database.

        Args:
            limit: Maximum number of reports to return.
            report_type: Filter by report type ('daily' or 'weekly').

        Returns:
            List of report dictionaries, most recent first.
        """
        result = await self._db.execute(
            select(ValidationReport)
            .where(ValidationReport.report_type == report_type)
            .order_by(desc(ValidationReport.report_date))
            .limit(limit)
        )
        reports = result.scalars().all()

        return [self._report_to_dict(r) for r in reports]

    async def get_rolling_metrics(self, window_days: int = 20) -> dict[str, Any]:
        """Calculate rolling performance metrics over the last N trading days.

        Args:
            window_days: Number of recent trading days to consider.

        Returns:
            Dictionary of rolling metrics.
        """
        result = await self._db.execute(
            select(ValidationReport)
            .where(ValidationReport.report_type == "daily")
            .order_by(desc(ValidationReport.report_date))
            .limit(window_days)
        )
        reports = list(reversed(result.scalars().all()))

        if not reports:
            return {
                "period_days": window_days,
                "available_days": 0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
                "total_trades": 0,
                "avg_trade_pnl": 0.0,
                "expectancy": 0.0,
                "volatility": 0.0,
                "calmar_ratio": 0.0,
                "updated_at": "",
                # Legacy
                "window_days": window_days,
                "metrics": {},
            }

        daily_pnls = [r.daily_pnl for r in reports]
        _cumulative_pnls = [r.cumulative_pnl for r in reports]
        _win_rates = [r.win_rate for r in reports if r.total_trades > 0]
        total_trades = sum(r.total_trades for r in reports)
        total_winning = sum(r.winning_trades for r in reports)
        total_anomalies = sum(len(r.anomalies) if r.anomalies else 0 for r in reports)

        avg_daily_pnl = sum(daily_pnls) / len(daily_pnls) if daily_pnls else 0.0

        # Simple daily return std for rolling Sharpe approximation
        import numpy as np

        pnl_arr = np.array(daily_pnls, dtype=float)
        portfolio_values = [r.portfolio_value for r in reports if r.portfolio_value > 0]
        avg_pv = np.mean(portfolio_values) if portfolio_values else self._initial_capital
        daily_returns = pnl_arr / avg_pv if avg_pv > 0 else pnl_arr
        rolling_sharpe = 0.0
        if len(daily_returns) >= 2:
            std = float(np.std(daily_returns, ddof=1))
            if std > 0:
                rolling_sharpe = float(np.mean(daily_returns) / std * np.sqrt(252))

        # Streak tracking
        current_streak = 0
        streak_type = "none"
        for pnl in reversed(daily_pnls):
            if pnl > 0:
                if streak_type == "none":
                    streak_type = "win"
                if streak_type == "win":
                    current_streak += 1
                else:
                    break
            elif pnl < 0:
                if streak_type == "none":
                    streak_type = "loss"
                if streak_type == "loss":
                    current_streak += 1
                else:
                    break
            else:
                break

        latest = reports[-1]

        total_pnl = round(sum(daily_pnls), 2)
        win_rate = round(total_winning / total_trades * 100, 2) if total_trades > 0 else 0.0
        max_dd = round(latest.max_drawdown_pct, 2)
        avg_trade_pnl = round(total_pnl / total_trades, 2) if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = sum(p for p in daily_pnls if p > 0)
        gross_loss = abs(sum(p for p in daily_pnls if p < 0))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0

        # Volatility (annualised daily return std)
        volatility = (
            round(
                float(np.std(daily_returns, ddof=1)
                      * np.sqrt(252) * 100),
                2,
            )
            if len(daily_returns) >= 2 else 0.0
        )

        # Calmar ratio
        calmar_ratio = round(rolling_sharpe / max_dd, 2) if max_dd > 0 else 0.0

        return {
            "period_days": window_days,
            "available_days": len(reports),
            "sharpe_ratio": round(rolling_sharpe, 3),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "avg_trade_pnl": avg_trade_pnl,
            "expectancy": avg_trade_pnl,
            "volatility": volatility,
            "calmar_ratio": calmar_ratio,
            "updated_at": latest.report_date.isoformat() if latest.report_date else "",
            # Legacy fields
            "window_days": window_days,
            "metrics": {
                "avg_daily_pnl": round(avg_daily_pnl, 2),
                "total_pnl": total_pnl,
                "best_day": round(max(daily_pnls), 2),
                "worst_day": round(min(daily_pnls), 2),
                "positive_days": len([p for p in daily_pnls if p > 0]),
                "negative_days": len([p for p in daily_pnls if p < 0]),
                "flat_days": len([p for p in daily_pnls if p == 0]),
                "total_trades": total_trades,
                "overall_win_rate": win_rate,
                "rolling_sharpe": round(rolling_sharpe, 3),
                "current_streak": current_streak,
                "streak_type": streak_type,
                "latest_cumulative_pnl": round(latest.cumulative_pnl, 2),
                "latest_portfolio_value": round(latest.portfolio_value, 2),
                "latest_max_drawdown_pct": max_dd,
                "total_anomalies": total_anomalies,
            },
            "date_range": {
                "start": reports[0].report_date.isoformat() if reports[0].report_date else None,
                "end": latest.report_date.isoformat() if latest.report_date else None,
            },
        }

    async def get_comparison_report(self) -> dict[str, Any]:
        """Generate a comparison of paper trading vs backtest predictions.

        Uses the validator's backtest deviation calculation and enriches it
        with additional context.
        """
        status = await self._validator.get_validation_status()
        readiness = await self._validator.is_ready_for_live()

        # Find the backtest deviation check
        bt_deviation_check = None
        for c in readiness.checks:
            if c.name == "backtest_deviation":
                bt_deviation_check = {
                    "passed": c.passed,
                    "required": c.required,
                    "actual": c.actual,
                    "detail": c.detail,
                }
                break

        trading_days = status.get("trading_days", 0)
        return {
            "has_data": trading_days > 0,
            "metrics": [],
            "overall_deviation": 0.0,
            "correlation": 0.0,
            "assessment": "Not enough data" if trading_days == 0 else "Collecting data",
            # Legacy fields
            "trading_days": trading_days,
            "paper_metrics": status.get("metrics", {}),
            "backtest_deviation": bt_deviation_check,
            "readiness": readiness.to_dict(),
        }

    # ── Scheduler integration ───────────────────────────────

    @staticmethod
    async def scheduled_daily_report() -> None:
        """APScheduler-compatible entry point for generating the daily report.

        Creates its own database session and DailyReporter instance, so it
        can be called directly from the scheduler without external state.
        """
        logger.info("daily_reporter.scheduled_run_starting")
        async with session_factory() as db:
            try:
                reporter = DailyReporter(db=db)
                report = await reporter.generate_and_store_report()
                logger.info(
                    "daily_reporter.scheduled_run_complete",
                    report_id=report.id,
                )
            except Exception:
                logger.exception("daily_reporter.scheduled_run_error")

    # ── Internal helpers ────────────────────────────────────

    async def _send_daily_summary(
        self,
        report: ValidationReport,
        readiness,
    ) -> None:
        """Send a daily Telegram summary of paper trading results."""
        pnl_emoji = "\U0001f7e2" if report.daily_pnl >= 0 else "\U0001f534"
        cum_emoji = "\U0001f7e2" if report.cumulative_pnl >= 0 else "\U0001f534"
        passed = sum(1 for c in readiness.checks if c.passed)
        total = len(readiness.checks)
        anomaly_count = len(report.anomalies) if report.anomalies else 0

        lines = [
            f"Date: {report.report_date}",
            "",
            f"{pnl_emoji} Daily P&L: \u20ac{report.daily_pnl:,.2f}",
            f"{cum_emoji} Cumulative P&L: \u20ac{report.cumulative_pnl:,.2f}",
            f"Portfolio Value: \u20ac{report.portfolio_value:,.2f}",
            "",
            f"Trades: {report.total_trades} ({report.winning_trades}W / {report.losing_trades}L)",
            f"Win Rate: {report.win_rate:.1f}%",
            f"Max Drawdown: {report.max_drawdown_pct:.2f}%",
        ]

        if anomaly_count:
            lines.append(f"\u26a0\ufe0f Anomalies: {anomaly_count}")

        lines.append("")
        lines.append(f"Readiness: {passed}/{total} checks passed")
        lines.append(f"Trading days: {readiness.trading_days}")

        await send_alert(
            title="Daily Paper Trading Report",
            message="\n".join(lines),
        )

    async def _send_anomaly_alert(
        self,
        report: ValidationReport,
        critical_anomalies: list[dict],
    ) -> None:
        """Send an alert for critical anomalies detected in the daily report."""
        anomaly_descriptions = "\n".join(
            f"  - [{a.get('category', 'unknown')}] {a.get('description', '')}"
            for a in critical_anomalies
        )
        message = (
            f"Date: {report.report_date}\n"
            f"Daily P&L: ${report.daily_pnl:,.2f}\n"
            f"Portfolio Value: ${report.portfolio_value:,.2f}\n"
            f"\nCritical Anomalies:\n{anomaly_descriptions}"
        )
        await send_alert(
            title="Paper Trading - Critical Anomalies Detected",
            message=message,
            critical=True,
        )

    @staticmethod
    def _report_to_dict(report: ValidationReport) -> dict[str, Any]:
        """Convert a ValidationReport ORM instance to a dictionary."""
        return {
            "id": report.id,
            "report_date": report.report_date.isoformat() if report.report_date else None,
            "report_type": report.report_type,
            "total_trades": report.total_trades,
            "winning_trades": report.winning_trades,
            "losing_trades": report.losing_trades,
            "win_rate": report.win_rate,
            "daily_pnl": report.daily_pnl,
            "cumulative_pnl": report.cumulative_pnl,
            "sharpe_ratio": report.sharpe_ratio,
            "max_drawdown_pct": report.max_drawdown_pct,
            "profit_factor": report.profit_factor,
            "portfolio_value": report.portfolio_value,
            "backtest_deviation_pct": report.backtest_deviation_pct,
            "anomalies": report.anomalies,
            "risk_events_count": report.risk_events_count,
            "risk_events_summary": report.risk_events_summary,
            "summary_text": report.summary_text,
            "created_at": report.created_at.isoformat() if report.created_at else None,
        }
