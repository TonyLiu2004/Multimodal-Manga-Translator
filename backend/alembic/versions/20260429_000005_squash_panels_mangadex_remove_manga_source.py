"""squash: panels refactor + mangadex ids + remove manga_source

Revision ID: 20260429_000005
Revises: 34d55d0d8469
Create Date: 2026-04-29

This squashes the following revisions into one:
  - 20260429_000001_segments_match_translated_segment_shape.py
  - 20260429_000002_panels_drop_pages.py
  - 20260429_000003_add_mangadex_chapter_id_to_chapters_panels.py
  - 20260429_000004_remove_manga_source.py

Target end-state:
  - pages/segments removed; panels table used instead
  - chapters and panels include mangadex_chapter_id
  - manga includes mangadex_manga_id; manga_source removed
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision = "20260429_000005"
down_revision = "34d55d0d8469"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # --- 1) segments: rename segment_index -> bubble_index; add width/height ---
    # Handle both "segment_index" (older) and already-renamed "bubble_index" (idempotent).
    conn.execute(
        text(
            """
            DO $$
            BEGIN
              IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='segments' AND column_name='segment_index'
              ) AND NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='segments' AND column_name='bubble_index'
              ) THEN
                ALTER TABLE segments RENAME COLUMN segment_index TO bubble_index;
              END IF;
            END
            $$;
            """
        )
    )

    conn.execute(
        text(
            """
            DO $$
            BEGIN
              IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='segments') THEN
                IF NOT EXISTS (
                  SELECT 1 FROM information_schema.columns
                  WHERE table_name='segments' AND column_name='width'
                ) THEN
                  ALTER TABLE segments ADD COLUMN width INTEGER NULL;
                END IF;
                IF NOT EXISTS (
                  SELECT 1 FROM information_schema.columns
                  WHERE table_name='segments' AND column_name='height'
                ) THEN
                  ALTER TABLE segments ADD COLUMN height INTEGER NULL;
                END IF;
              END IF;
            END
            $$;
            """
        )
    )

    # Ensure composite index matches the renamed column (best-effort).
    conn.execute(text("DROP INDEX IF EXISTS ix_segments_page_segment"))
    conn.execute(text("DROP INDEX IF EXISTS ix_segments_page_bubble"))
    conn.execute(
        text(
            """
            DO $$
            BEGIN
              IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='segments') THEN
                CREATE INDEX IF NOT EXISTS ix_segments_page_bubble ON segments (page_id, bubble_index);
              END IF;
            END
            $$;
            """
        )
    )

    # --- 2) create panels and migrate data from segments+pages ---
    op.create_table(
        "panels",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chapter_id", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("mangadex_chapter_id", sa.String(), nullable=True),
        sa.Column("bubble_index", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("x1", sa.Float(), nullable=False),
        sa.Column("y1", sa.Float(), nullable=False),
        sa.Column("x2", sa.Float(), nullable=False),
        sa.Column("y2", sa.Float(), nullable=False),
        sa.Column("original_text", sa.String(), nullable=False),
        sa.Column("translated_text", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_panels_chapter_id", "panels", ["chapter_id"], unique=False)
    op.create_index("ix_panels_page_number", "panels", ["page_number"], unique=False)
    op.create_index(
        "ix_panels_chapter_page_bubble",
        "panels",
        ["chapter_id", "page_number", "bubble_index"],
        unique=False,
    )
    op.create_index(
        "ix_panels_mangadex_chapter_id",
        "panels",
        ["mangadex_chapter_id"],
        unique=False,
    )

    # Copy all existing segment rows into panels (if the old tables exist).
    conn.execute(
        text(
            """
            DO $$
            BEGIN
              IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='segments')
                 AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='pages')
              THEN
                INSERT INTO panels
                    (chapter_id, page_number, bubble_index, width, height,
                     x1, y1, x2, y2, original_text, translated_text, created_at)
                SELECT
                    p.chapter_id,
                    p.page_number,
                    s.bubble_index,
                    s.width,
                    s.height,
                    s.x1, s.y1, s.x2, s.y2,
                    s.original_text,
                    s.translated_text,
                    s.created_at
                FROM segments s
                INNER JOIN pages p ON p.id = s.page_id;
              END IF;
            END
            $$;
            """
        )
    )

    # --- 3) add mangadex_chapter_id to chapters (nullable) ---
    conn.execute(
        text(
            """
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='chapters' AND column_name='mangadex_chapter_id'
              ) THEN
                ALTER TABLE chapters ADD COLUMN mangadex_chapter_id VARCHAR NULL;
              END IF;
            END
            $$;
            """
        )
    )
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chapters_mangadex_chapter_id ON chapters (mangadex_chapter_id)"))

    # Copy chapter mangadex id into panels (best-effort backfill).
    conn.execute(
        text(
            """
            UPDATE panels p
            SET mangadex_chapter_id = c.mangadex_chapter_id
            FROM chapters c
            WHERE p.chapter_id = c.id
              AND p.mangadex_chapter_id IS NULL
              AND c.mangadex_chapter_id IS NOT NULL;
            """
        )
    )

    # --- 4) manga: add mangadex_manga_id and backfill from manga_source, then drop manga_source ---
    conn.execute(
        text(
            """
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='manga' AND column_name='mangadex_manga_id'
              ) THEN
                ALTER TABLE manga ADD COLUMN mangadex_manga_id VARCHAR NULL;
              END IF;
            END
            $$;
            """
        )
    )
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_manga_mangadex_manga_id ON manga (mangadex_manga_id)"))
    conn.execute(
        text(
            """
            DO $$
            BEGIN
              IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='manga_source') THEN
                UPDATE manga m
                SET mangadex_manga_id = ms.external_manga_id
                FROM (
                  SELECT manga_id, MIN(external_manga_id) AS external_manga_id
                  FROM manga_source
                  WHERE provider_id = 'mangadex'
                    AND external_manga_id IS NOT NULL
                    AND external_manga_id NOT LIKE 'legacy-%'
                    AND external_manga_id NOT LIKE 'local-%'
                  GROUP BY manga_id
                ) ms
                WHERE m.id = ms.manga_id
                  AND (m.mangadex_manga_id IS NULL OR m.mangadex_manga_id = '');
              END IF;
            END
            $$;
            """
        )
    )
    conn.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_manga_mangadex_manga_id ON manga (mangadex_manga_id)"
        )
    )
    conn.execute(text("DROP TABLE IF EXISTS manga_source CASCADE"))

    # --- 5) drop old pages/segments tables (if present) ---
    conn.execute(text("DROP TABLE IF EXISTS segments CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS pages CASCADE"))


def downgrade() -> None:
    raise NotImplementedError("Squashed migration is not designed to downgrade.")

