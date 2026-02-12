"""SQLAlchemy model for tracking ML model experiments."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class ModelExperiment(Base):
    __tablename__ = "model_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    hyperparameters: Mapped[dict] = mapped_column(JSONB, default=dict)
    train_metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    val_metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    test_metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    feature_importance: Mapped[dict] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
