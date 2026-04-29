"""Data migration: merge duplicate manga rows sharing the same MangaDex series id.

Revision ID: 20260429_000009
Revises: 20260429_000008
Create Date: 2026-04-29

When the same series was added to several reading lists in parallel, multiple
``manga`` rows could be created with the same ``mangadex_manga_id`` (possibly
differing only by case). This migration:

- Keeps the smallest ``manga.id`` per ``lower(trim(mangadex_manga_id))`` group
- Points ``reading_list_item`` and ``chapters`` at that canonical row
- Merges overlapping chapters (same ``chapter_number``) by moving panels and
  dropping duplicate chapters; deletes source panels that would duplicate a
  non-null ``panel_url`` already present on the target chapter
- Removes extra ``reading_list_item`` rows that share the same
  ``(reading_list_id, manga_id)`` after the merge (keeps best progress)

Requires PostgreSQL. Irreversible: duplicate rows are deleted, not restored.
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import text


revision = "20260429_000009"
down_revision = "20260429_000008"
branch_labels = None
depends_on = None


def _merge_one_duplicate_manga(conn, canonical_id: int, dup_id: int) -> None:
    """Remap FKs from dup_id onto canonical_id, then delete dup_id manga row."""
    conn.execute(
        text(
            "UPDATE reading_list_item SET manga_id = :canon WHERE manga_id = :dup",
        ),
        {"canon": canonical_id, "dup": dup_id},
    )

    chapters = conn.execute(
        text(
            "SELECT id, chapter_number FROM chapters WHERE manga_id = :d ORDER BY id",
        ),
        {"d": dup_id},
    ).fetchall()

    for ch_id, ch_num in chapters:
        row = conn.execute(
            text(
                "SELECT id FROM chapters WHERE manga_id = :c AND chapter_number = :n "
                "LIMIT 1",
            ),
            {"c": canonical_id, "n": ch_num},
        ).fetchone()

        if row is not None:
            canon_ch_id = row[0]
            # Drop source panels whose panel_url already exists on the target chapter
            # (partial unique index uq_panels_panel_url_not_null).
            conn.execute(
                text(
                    """
                    DELETE FROM panels p
                    WHERE p.chapter_id = :old_ch
                      AND p.panel_url IS NOT NULL
                      AND EXISTS (
                        SELECT 1 FROM panels p2
                        WHERE p2.chapter_id = :new_ch
                          AND p2.panel_url IS NOT NULL
                          AND p2.panel_url = p.panel_url
                      )
                    """,
                ),
                {"old_ch": ch_id, "new_ch": canon_ch_id},
            )
            conn.execute(
                text(
                    "UPDATE panels SET chapter_id = :cc WHERE chapter_id = :oc",
                ),
                {"cc": canon_ch_id, "oc": ch_id},
            )
            conn.execute(text("DELETE FROM chapters WHERE id = :oc"), {"oc": ch_id})
        else:
            conn.execute(
                text("UPDATE chapters SET manga_id = :c WHERE id = :cid"),
                {"c": canonical_id, "cid": ch_id},
            )

    conn.execute(text("DELETE FROM manga WHERE id = :d"), {"d": dup_id})


def upgrade() -> None:
    conn = op.get_bind()

    groups = conn.execute(
        text(
            """
            SELECT array_agg(id ORDER BY id) AS ids
            FROM manga
            WHERE mangadex_manga_id IS NOT NULL
              AND btrim(mangadex_manga_id) <> ''
            GROUP BY lower(btrim(mangadex_manga_id))
            HAVING COUNT(*) > 1
            """,
        ),
    ).fetchall()

    for (ids_arr,) in groups:
        if not ids_arr:
            continue
        ids = [int(x) for x in ids_arr]
        canonical_id = min(ids)
        dup_ids = [i for i in ids if i != canonical_id]
        for dup_id in dup_ids:
            _merge_one_duplicate_manga(conn, canonical_id, dup_id)

    # Same list may now reference the same manga twice — keep one row (best progress).
    conn.execute(
        text(
            """
            DELETE FROM reading_list_item
            WHERE id IN (
              SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                         PARTITION BY reading_list_id, manga_id
                         ORDER BY COALESCE(last_chapter_number, -1) DESC NULLS LAST,
                                  updated_at DESC NULLS LAST,
                                  id ASC
                       ) AS rn
                FROM reading_list_item
              ) sub
              WHERE rn > 1
            )
            """,
        ),
    )


def downgrade() -> None:
    raise NotImplementedError(
        "Dedupe migration is irreversible (merged rows are deleted).",
    )
