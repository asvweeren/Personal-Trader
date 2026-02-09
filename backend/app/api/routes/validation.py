"""API endpoints for paper-trading validation (Phase 8)."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_db
from app.monitoring.daily_reporter import DailyReporter
from app.monitoring.paper_trading_validator import PaperTradingValidator

router = APIRouter()


def _get_validator(db: AsyncSession = Depends(get_db)) -> PaperTradingValidator:
    return PaperTradingValidator(db=db, initial_capital=settings.initial_capital)


def _get_reporter(db: AsyncSession = Depends(get_db)) -> DailyReporter:
    return DailyReporter(db=db, initial_capital=settings.initial_capital)


@router.get("/validation/status")
async def get_validation_status(
    validator: PaperTradingValidator = Depends(_get_validator),
):
    """Current paper-trading validation status.

    Returns the number of trading days completed, cumulative metrics,
    latest daily summary, and anomaly counts.
    """
    return await validator.get_validation_status()


@router.get("/validation/report")
async def get_validation_report(
    period: str = Query("daily", regex="^(daily|weekly)$"),
    days: int = Query(30, ge=1, le=365),
    validator: PaperTradingValidator = Depends(_get_validator),
):
    """Daily or weekly performance report for paper trading.

    Args:
        period: 'daily' or 'weekly' aggregation.
        days: Look-back window in calendar days.
    """
    return await validator.get_performance_report(period=period, days=days)


@router.get("/validation/readiness")
async def get_readiness(
    validator: PaperTradingValidator = Depends(_get_validator),
):
    """Live-trading readiness check.

    Evaluates whether paper-trading results meet the minimum criteria
    for transitioning to live trading:
    - >= 20 trading days of data
    - Sharpe ratio > 0.5
    - Max drawdown < 15%
    - Win rate > 40%
    - Profit factor > 1.2
    - Results within 30% of backtest predictions
    """
    result = await validator.is_ready_for_live()
    return result.to_dict()


@router.get("/validation/reports/history")
async def get_report_history(
    limit: int = Query(30, ge=1, le=200),
    report_type: str = Query("daily", regex="^(daily|weekly)$"),
    reporter: DailyReporter = Depends(_get_reporter),
):
    """Fetch historical validation reports from the database."""
    return await reporter.get_recent_reports(limit=limit, report_type=report_type)


@router.get("/validation/metrics/rolling")
async def get_rolling_metrics(
    window: int = Query(20, ge=5, le=100),
    reporter: DailyReporter = Depends(_get_reporter),
):
    """Rolling performance metrics over the last N trading days."""
    return await reporter.get_rolling_metrics(window_days=window)


@router.get("/validation/comparison")
async def get_backtest_comparison(
    reporter: DailyReporter = Depends(_get_reporter),
):
    """Compare paper-trading results to backtest predictions."""
    return await reporter.get_comparison_report()


@router.post("/validation/report/generate")
async def trigger_daily_report(
    reporter: DailyReporter = Depends(_get_reporter),
):
    """Manually trigger generation of today's daily validation report.

    Normally this runs automatically via APScheduler, but this endpoint
    allows manual invocation for testing or on-demand reporting.
    """
    report = await reporter.generate_and_store_report()
    return {
        "report_id": report.id,
        "date": report.report_date.isoformat() if report.report_date else None,
        "daily_pnl": report.daily_pnl,
        "cumulative_pnl": report.cumulative_pnl,
        "anomalies": len(report.anomalies) if report.anomalies else 0,
        "summary": report.summary_text,
    }
