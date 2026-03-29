"""reading_list_item: store provider + external manga id, drop FK to public.manga

Revision ID: b7e2a1c0d9f8
Revises: 34d55d0d8469
Create Date: 2026-03-29

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision = "b7e2a1c0d9f8"
down_revision = "34d55d0d8469"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "reading_list_item",
        sa.Column("provider_id", sa.String(), nullable=True),
    )
    op.add_column(
        "reading_list_item",
        sa.Column("external_manga_id", sa.String(), nullable=True),
    )
    op.add_column(
        "reading_list_item",
        sa.Column("manga_title", sa.String(length=500), nullable=True),
    )

    conn = op.get_bind()
    conn.execute(
        text(
            """
            UPDATE reading_list_item AS rli
            SET
                provider_id = ms.provider_id,
                external_manga_id = ms.external_manga_id,
                manga_title = m.manga_title
            FROM manga AS m
            INNER JOIN manga_source AS ms
                ON ms.manga_id = m.id
                AND ms.provider_id = 'mangadex'
            WHERE rli.manga_id = m.id
            """
        )
    )
    conn.execute(text("DELETE FROM reading_list_item WHERE external_manga_id IS NULL"))

    op.drop_constraint(
        "reading_list_item_manga_id_fkey",
        "reading_list_item",
        type_="foreignkey",
    )
    op.drop_constraint("uq_rli_list_manga", "reading_list_item", type_="unique")
    op.drop_index(op.f("ix_reading_list_item_manga_id"), table_name="reading_list_item")
    op.drop_column("reading_list_item", "manga_id")

    op.alter_column(
        "reading_list_item",
        "provider_id",
        existing_type=sa.String(),
        nullable=False,
    )
    op.alter_column(
        "reading_list_item",
        "external_manga_id",
        existing_type=sa.String(),
        nullable=False,
    )
    op.alter_column(
        "reading_list_item",
        "manga_title",
        existing_type=sa.String(length=500),
        nullable=False,
    )

    op.create_index(
        op.f("ix_reading_list_item_external_manga_id"),
        "reading_list_item",
        ["external_manga_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_reading_list_item_provider_id"),
        "reading_list_item",
        ["provider_id"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_rli_list_provider_external",
        "reading_list_item",
        ["reading_list_id", "provider_id", "external_manga_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_rli_list_provider_external", "reading_list_item", type_="unique"
    )
    op.drop_index(
        op.f("ix_reading_list_item_provider_id"), table_name="reading_list_item"
    )
    op.drop_index(
        op.f("ix_reading_list_item_external_manga_id"),
        table_name="reading_list_item",
    )

    op.add_column(
        "reading_list_item",
        sa.Column("manga_id", sa.Integer(), nullable=True),
    )

    conn = op.get_bind()
    conn.execute(
        text(
            """
            UPDATE reading_list_item AS rli
            SET manga_id = ms.manga_id
            FROM manga_source AS ms
            WHERE rli.provider_id = ms.provider_id
              AND rli.external_manga_id = ms.external_manga_id
            """
        )
    )
    conn.execute(text("DELETE FROM reading_list_item WHERE manga_id IS NULL"))

    op.alter_column(
        "reading_list_item", "manga_id", existing_type=sa.Integer(), nullable=False
    )
    op.create_foreign_key(
        "reading_list_item_manga_id_fkey",
        "reading_list_item",
        "manga",
        ["manga_id"],
        ["id"],
    )
    op.create_index(
        op.f("ix_reading_list_item_manga_id"),
        "reading_list_item",
        ["manga_id"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_rli_list_manga", "reading_list_item", ["reading_list_id", "manga_id"]
    )

    op.drop_column("reading_list_item", "manga_title")
    op.drop_column("reading_list_item", "external_manga_id")
    op.drop_column("reading_list_item", "provider_id")
