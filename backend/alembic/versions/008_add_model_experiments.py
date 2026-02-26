"""Add model_experiments table for ML experiment tracking

Revision ID: 008
Revises: 007
Create Date: 2026-02-12
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_experiments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("strategy_name", sa.String(50), nullable=False, index=True),
        sa.Column("version", sa.String(20), nullable=False),
        sa.Column("hyperparameters", JSONB(), server_default="{}"),
        sa.Column("train_metrics", JSONB(), server_default="{}"),
        sa.Column("val_metrics", JSONB(), server_default="{}"),
        sa.Column("test_metrics", JSONB(), server_default="{}"),
        sa.Column("feature_importance", JSONB(), server_default="{}"),
        sa.Column("is_active", sa.Boolean(), server_default="false"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("model_experiments")
