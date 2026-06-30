"""Add documents.content_hash for ingestion dedup

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-30
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("content_hash", sa.String(64), nullable=True))
    op.create_index("idx_documents_content_hash", "documents", ["content_hash"])


def downgrade() -> None:
    op.drop_index("idx_documents_content_hash", table_name="documents")
    op.drop_column("documents", "content_hash")
