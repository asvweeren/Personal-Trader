"""Database model for paper trading validation reports."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class ValidationReport(Base):
    __tablename__ = "validation_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    report_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="daily"
    )  # daily, weekly, summary

    # Core metrics
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    win_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    daily_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cumulative_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    profit_factor: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    portfolio_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Comparison with backtest
    backtest_deviation_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Anomalies detected
    anomalies: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Full metrics snapshot for the day
    metrics_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Risk events summary
    risk_events_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    risk_events_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Narrative summary
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
