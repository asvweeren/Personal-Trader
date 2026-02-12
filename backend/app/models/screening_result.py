"""Database model for daily stock screening results."""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Integer, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class ScreeningResult(Base):
    __tablename__ = "screening_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    screening_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    total_scanned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    candidates: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
