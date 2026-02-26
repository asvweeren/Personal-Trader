"""Add risk events table

Revision ID: 003
Revises: 002
Create Date: 2026-02-09
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    riskeventtype = sa.Enum(
        "DAILY_LOSS_TRIGGERED",
        "POSITION_SIZE_EXCEEDED",
        "MAX_POSITIONS_EXCEEDED",
        "CASH_RESERVE_LOW",
        "MARKET_CLOSED",
        "DRAWDOWN_WARNING",
        "CONCENTRATION_WARNING",
        "SIGNAL_REJECTED",
        "TRADING_HALTED",
        "TRADING_RESUMED",
        name="riskeventtype",
    )
    riskeventseverity = sa.Enum(
        "INFO",
        "WARNING",
        "CRITICAL",
        name="riskeventseverity",
    )

    op.create_table(
        "risk_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_type", riskeventtype, nullable=False),
        sa.Column("severity", riskeventseverity, nullable=False, server_default="WARNING"),
        sa.Column("symbol", sa.String(20), nullable=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("action_taken", sa.Text(), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("portfolio_value", sa.Float(), nullable=True),
        sa.Column("daily_loss_pct", sa.Float(), nullable=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_risk_events_timestamp", "risk_events", ["timestamp"])
    op.create_index("ix_risk_events_event_type", "risk_events", ["event_type"])


def downgrade() -> None:
    op.drop_index("ix_risk_events_event_type", table_name="risk_events")
    op.drop_index("ix_risk_events_timestamp", table_name="risk_events")
    op.drop_table("risk_events")
    sa.Enum(name="riskeventtype").drop(op.get_bind())
    sa.Enum(name="riskeventseverity").drop(op.get_bind())
