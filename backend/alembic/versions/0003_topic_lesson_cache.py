"""Add plan_topics.lesson_cache for cached per-topic lessons

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-30
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("plan_topics", sa.Column("lesson_cache", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("plan_topics", "lesson_cache")
