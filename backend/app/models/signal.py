import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class SignalAction(enum.StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Signal(Base):
    __tablename__ = "signals"
    __table_args__ = (
        Index("ix_signal_symbol_created", "symbol", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    action: Mapped[SignalAction] = mapped_column(Enum(SignalAction), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    features_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
