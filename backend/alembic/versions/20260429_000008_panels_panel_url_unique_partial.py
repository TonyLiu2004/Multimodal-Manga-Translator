"""panels: global unique panel_url (when set)

Revision ID: 20260429_000008
Revises: 20260429_000007
Create Date: 2026-04-29
"""

from alembic import op


revision = "20260429_000008"
down_revision = "20260429_000007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Postgres: enforce uniqueness only when panel_url is present (still allows many NULLs).
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_panels_panel_url_not_null
        ON panels (panel_url)
        WHERE panel_url IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_panels_panel_url_not_null;")
