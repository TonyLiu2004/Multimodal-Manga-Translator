"""single provider: drop chapters.provider_id; panels.page_number nullable

Revision ID: 20260429_000006
Revises: 20260429_000005
Create Date: 2026-04-29
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision = "20260429_000006"
down_revision = "20260429_000005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # --- chapters: drop provider_id + replace unique constraint ---
    # Drop old constraint/index if present.
    conn.execute(text("ALTER TABLE chapters DROP CONSTRAINT IF EXISTS uq_chapters_manga_provider_chapter"))
    conn.execute(text("DROP INDEX IF EXISTS ix_chapters_provider_id"))

    # Drop the column if it exists.
    conn.execute(
        text(
            """
            DO $$
            BEGIN
              IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='chapters' AND column_name='provider_id'
              ) THEN
                ALTER TABLE chapters DROP COLUMN provider_id;
              END IF;
            END
            $$;
            """
        )
    )

    # Ensure new uniqueness.
    conn.execute(text("ALTER TABLE chapters DROP CONSTRAINT IF EXISTS uq_chapters_manga_chapter"))
    conn.execute(text("ALTER TABLE chapters ADD CONSTRAINT uq_chapters_manga_chapter UNIQUE (manga_id, chapter_number)"))

    # --- panels: allow page_number NULL ---
    op.alter_column(
        "panels",
        "page_number",
        existing_type=sa.Integer(),
        nullable=True,
    )


def downgrade() -> None:
    raise NotImplementedError("Not supporting downgrade for this schema simplification.")

