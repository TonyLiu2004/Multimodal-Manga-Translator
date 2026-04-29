"""panels: add panel_url

Revision ID: 20260429_000007
Revises: 20260429_000006
Create Date: 2026-04-29
"""

from alembic import op
import sqlalchemy as sa


revision = "20260429_000007"
down_revision = "20260429_000006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("panels", sa.Column("panel_url", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("panels", "panel_url")

