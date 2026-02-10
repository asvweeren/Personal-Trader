"""Add validation reports table

Revision ID: 004
Revises: 003
Create Date: 2026-02-10
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "validation_reports",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("report_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("report_type", sa.String(20), nullable=False, server_default="daily"),
        sa.Column("total_trades", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("winning_trades", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("losing_trades", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("win_rate", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("daily_pnl", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("cumulative_pnl", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("sharpe_ratio", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("max_drawdown_pct", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("profit_factor", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("portfolio_value", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("backtest_deviation_pct", sa.Float(), nullable=True),
        sa.Column("anomalies", postgresql.JSONB(), nullable=True),
        sa.Column("metrics_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column("risk_events_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("risk_events_summary", postgresql.JSONB(), nullable=True),
        sa.Column("summary_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_validation_reports_report_date", "validation_reports", ["report_date"])


def downgrade() -> None:
    op.drop_index("ix_validation_reports_report_date", table_name="validation_reports")
    op.drop_table("validation_reports")
