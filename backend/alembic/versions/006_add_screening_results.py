"""Add screening_results table

Revision ID: 006
Revises: 005
Create Date: 2026-02-12
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "screening_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("screening_date", sa.Date(), nullable=False),
        sa.Column("total_scanned", sa.Integer(), nullable=False),
        sa.Column("candidates", postgresql.JSONB(), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_screening_results_screening_date",
        "screening_results",
        ["screening_date"],
    )


def downgrade() -> None:
    op.drop_index("ix_screening_results_screening_date", table_name="screening_results")
    op.drop_table("screening_results")
