from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Enum, Float, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class RiskEventType(StrEnum):
    DAILY_LOSS_TRIGGERED = "DAILY_LOSS_TRIGGERED"
    POSITION_SIZE_EXCEEDED = "POSITION_SIZE_EXCEEDED"
    MAX_POSITIONS_EXCEEDED = "MAX_POSITIONS_EXCEEDED"
    CASH_RESERVE_LOW = "CASH_RESERVE_LOW"
    MARKET_CLOSED = "MARKET_CLOSED"
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING"
    CONCENTRATION_WARNING = "CONCENTRATION_WARNING"
    SIGNAL_REJECTED = "SIGNAL_REJECTED"
    TRADING_HALTED = "TRADING_HALTED"
    TRADING_RESUMED = "TRADING_RESUMED"


class RiskEventSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class RiskEvent(Base):
    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(
        Enum(
            *[e.value for e in RiskEventType],
            name="riskeventtype",
        ),
        nullable=False,
    )
    severity: Mapped[str] = mapped_column(
        Enum(
            *[e.value for e in RiskEventSeverity],
            name="riskeventseverity",
        ),
        nullable=False,
        default=RiskEventSeverity.WARNING.value,
    )
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    action_taken: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    portfolio_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    daily_loss_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
