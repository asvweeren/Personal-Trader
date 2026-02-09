"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-02-09
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Trades table
    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column(
            "side", sa.Enum("BUY", "SELL", name="tradeside"), nullable=False
        ),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("stop_loss", sa.Float(), nullable=True),
        sa.Column("take_profit", sa.Float(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("PENDING", "OPEN", "CLOSED", "CANCELLED", name="tradestatus"),
            nullable=False,
        ),
        sa.Column("strategy_name", sa.String(50), nullable=False),
        sa.Column("signal_id", sa.Integer(), nullable=True),
        sa.Column("realized_pnl", sa.Float(), nullable=True),
        sa.Column("commission", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trades_symbol", "trades", ["symbol"])

    # Orders table
    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trade_id", sa.Integer(), nullable=False),
        sa.Column("broker_order_id", sa.String(50), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column(
            "order_type",
            sa.Enum("MARKET", "LIMIT", "STOP", "STOP_LIMIT", name="ordertype"),
            nullable=False,
        ),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("limit_price", sa.Float(), nullable=True),
        sa.Column("stop_price", sa.Float(), nullable=True),
        sa.Column("filled_price", sa.Float(), nullable=True),
        sa.Column("filled_quantity", sa.Integer(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "SUBMITTED",
                "FILLED",
                "PARTIALLY_FILLED",
                "CANCELLED",
                "REJECTED",
                "ERROR",
                name="orderstatus",
            ),
            nullable=False,
        ),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["trade_id"], ["trades.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_orders_trade_id", "orders", ["trade_id"])

    # Signals table
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("strategy_name", sa.String(50), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column(
            "action",
            sa.Enum("BUY", "SELL", "HOLD", name="signalaction"),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("features_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_signals_strategy_name", "signals", ["strategy_name"])
    op.create_index("ix_signals_symbol", "signals", ["symbol"])

    # Portfolio snapshots table
    op.create_table(
        "portfolio_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("total_value", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("positions_value", sa.Float(), nullable=False),
        sa.Column("unrealized_pnl", sa.Float(), nullable=False),
        sa.Column("realized_pnl", sa.Float(), nullable=False),
        sa.Column("daily_pnl", sa.Float(), nullable=False),
        sa.Column("positions_detail", postgresql.JSONB(), nullable=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_portfolio_snapshots_timestamp", "portfolio_snapshots", ["timestamp"]
    )

    # Backtest results table
    op.create_table(
        "backtest_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("strategy_name", sa.String(50), nullable=False),
        sa.Column("params", postgresql.JSONB(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("trades_summary", postgresql.JSONB(), nullable=True),
        sa.Column("equity_curve", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("backtest_results")
    op.drop_table("portfolio_snapshots")
    op.drop_table("signals")
    op.drop_table("orders")
    op.drop_table("trades")

    # Drop enums
    sa.Enum(name="tradeside").drop(op.get_bind())
    sa.Enum(name="tradestatus").drop(op.get_bind())
    sa.Enum(name="ordertype").drop(op.get_bind())
    sa.Enum(name="orderstatus").drop(op.get_bind())
    sa.Enum(name="signalaction").drop(op.get_bind())
