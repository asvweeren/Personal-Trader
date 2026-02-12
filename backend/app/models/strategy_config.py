from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class StrategyConfig(Base):
    __tablename__ = "strategy_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    active_strategies: Mapped[list] = mapped_column(JSONB, nullable=False)
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.6)
    ensemble_method: Mapped[str] = mapped_column(
        String(50), nullable=False, default="weighted_average",
    )
    weights: Mapped[dict] = mapped_column(JSONB, nullable=False)
    symbols: Mapped[list] = mapped_column(JSONB, nullable=False)
    trading_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
