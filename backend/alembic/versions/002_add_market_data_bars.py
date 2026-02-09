"""Add market data bars table

Revision ID: 002
Revises: 001
Create Date: 2026-02-09
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "market_data_bars",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bar_size", sa.String(20), nullable=False, server_default="1h"),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "timestamp", "bar_size", name="uq_bar_symbol_ts_size"),
    )
    op.create_index("ix_market_data_bars_symbol", "market_data_bars", ["symbol"])
    op.create_index("ix_market_data_bars_timestamp", "market_data_bars", ["timestamp"])


def downgrade() -> None:
    op.drop_index("ix_market_data_bars_timestamp", table_name="market_data_bars")
    op.drop_index("ix_market_data_bars_symbol", table_name="market_data_bars")
    op.drop_table("market_data_bars")
