"""Add expected_price and slippage columns to orders

Revision ID: 007
Revises: 006
Create Date: 2026-02-12
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("orders", sa.Column("expected_price", sa.Float(), nullable=True))
    op.add_column("orders", sa.Column("slippage", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("orders", "slippage")
    op.drop_column("orders", "expected_price")
