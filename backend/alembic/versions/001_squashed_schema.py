"""squashed schema (manga, chapters, pages, segments)

Revision ID: 001_squashed
Revises: None
Create Date: 2026-03-09

Single migration representing full current schema.
For existing DBs: run 'alembic stamp 001_squashed' (do not run upgrade).
For fresh DBs: run 'alembic upgrade head'.
"""
from alembic import op
import sqlalchemy as sa


revision = '001_squashed'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'manga',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('provider_id', sa.String(), nullable=False),
        sa.Column('manga_title', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('provider_id', 'manga_title', name='uq_manga_provider_title'),
    )
    op.create_index('ix_manga_provider_id', 'manga', ['provider_id'])
    op.create_index('ix_manga_manga_title', 'manga', ['manga_title'])

    op.create_table(
        'chapters',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('manga_id', sa.Integer(), nullable=False),
        sa.Column('chapter_number', sa.Float(), nullable=False),
        sa.Column('language_code', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['manga_id'], ['manga.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('manga_id', 'chapter_number', name='uq_chapters_manga_chapter'),
    )
    op.create_index('ix_chapters_chapter_number', 'chapters', ['chapter_number'])
    op.create_index('ix_chapters_manga_id', 'chapters', ['manga_id'])

    op.create_table(
        'pages',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('chapter_id', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['chapter_id'], ['chapters.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('chapter_id', 'page_number', name='uq_pages_chapter_page'),
    )
    op.create_index('ix_pages_chapter_id', 'pages', ['chapter_id'])
    op.create_index('ix_pages_page_number', 'pages', ['page_number'])

    op.create_table(
        'segments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('page_id', sa.Integer(), nullable=False),
        sa.Column('segment_index', sa.Integer(), nullable=False),
        sa.Column('x1', sa.Float(), nullable=False),
        sa.Column('y1', sa.Float(), nullable=False),
        sa.Column('x2', sa.Float(), nullable=False),
        sa.Column('y2', sa.Float(), nullable=False),
        sa.Column('original_text', sa.String(), nullable=False),
        sa.Column('translated_text', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['page_id'], ['pages.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_segments_page_id', 'segments', ['page_id'])
    op.create_index('ix_segments_page_segment', 'segments', ['page_id', 'segment_index'])


def downgrade() -> None:
    op.drop_index('ix_segments_page_segment', table_name='segments')
    op.drop_index('ix_segments_page_id', table_name='segments')
    op.drop_table('segments')
    op.drop_index('ix_pages_page_number', table_name='pages')
    op.drop_index('ix_pages_chapter_id', table_name='pages')
    op.drop_table('pages')
    op.drop_index('ix_chapters_manga_id', table_name='chapters')
    op.drop_index('ix_chapters_chapter_number', table_name='chapters')
    op.drop_table('chapters')
    op.drop_index('ix_manga_manga_title', table_name='manga')
    op.drop_index('ix_manga_provider_id', table_name='manga')
    op.drop_table('manga')
