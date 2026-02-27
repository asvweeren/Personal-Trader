"""Add performance indexes to trades table.

Adds indexes on created_at, closed_at, and strategy_name for faster
trade listing, validator queries, and strategy performance aggregation.

Revision ID: 009
Revises: 008
"""

from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_trade_created_at", "trades", ["created_at"])
    op.create_index("ix_trade_closed_at", "trades", ["closed_at"])
    op.create_index("ix_trade_strategy_name", "trades", ["strategy_name"])


def downgrade() -> None:
    op.drop_index("ix_trade_strategy_name", table_name="trades")
    op.drop_index("ix_trade_closed_at", table_name="trades")
    op.drop_index("ix_trade_created_at", table_name="trades")
