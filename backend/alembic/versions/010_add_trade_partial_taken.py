"""Add partial_taken column to trades table.

Persists the partial profit-taking flag so it survives engine restarts.
Previously this was a runtime-only attribute that was lost on restart,
causing double partial closes.

Revision ID: 010
Revises: 009
"""

from alembic import op
import sqlalchemy as sa

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "trades",
        sa.Column("partial_taken", sa.Boolean(), server_default="false", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("trades", "partial_taken")
