"""Add strategy config table

Revision ID: 005
Revises: 004
Create Date: 2026-02-10
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "strategy_config",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("active_strategies", postgresql.JSONB(), nullable=False),
        sa.Column("confidence_threshold", sa.Float(), nullable=False, server_default="0.6"),
        sa.Column("ensemble_method", sa.String(50), nullable=False, server_default="weighted_average"),
        sa.Column("weights", postgresql.JSONB(), nullable=False),
        sa.Column("symbols", postgresql.JSONB(), nullable=False),
        sa.Column("trading_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    # Insert default row
    op.execute(
        """INSERT INTO strategy_config (id, active_strategies, confidence_threshold, ensemble_method, weights, symbols, trading_enabled)
        VALUES (1, '["ml_xgboost", "sentiment"]'::jsonb, 0.6, 'weighted_average',
                '{"ml_xgboost": 0.5, "sentiment": 0.3, "nn_lstm": 0.2}'::jsonb,
                '["SPY","QQQ","AAPL","MSFT","GOOGL","NVDA","AMZN","META","IWM","EFA","VGK"]'::jsonb,
                false)"""
    )


def downgrade() -> None:
    op.drop_table("strategy_config")
