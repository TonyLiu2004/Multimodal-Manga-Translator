"""
Manga table: stores manga metadata
  id: int, primary key
  provider_id: str
  manga_title: str
  created_at: datetime

Chapter table: stores manga chapter info
  id: int, primary key
  chapter_id: int, foreign key → chapters.id
  page_number: int
  created_at: datetime

Segment table: stores text segments (bubbles) within a page
  id: int, primary key
  page_id: int, foreign key → pages.id
  segment_index: int
  x1: float
  y1: float
  x2: float
  y2: float
  original_text: str
  translated_text: str
  created_at: datetime
"""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Index
from sqlmodel import Field, SQLModel, UniqueConstraint


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

# Models for the database, these are auto added to metadata in (SQLModel.metadata.create_all(engine))


class Manga(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("provider_id", "manga_title", name="uq_manga_provider_title"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    provider_id: str = Field(index=True)
    manga_title: str = Field(index=True)
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    updated_at: Optional[datetime] = Field(default_factory=_utc_now)


class Chapters(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("manga_id", "chapter_number", name="uq_chapters_manga_chapter"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    manga_id: int = Field(foreign_key="manga.id", index=True)
    chapter_number: float = Field(index=True)
    language_code: str
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    updated_at: Optional[datetime] = Field(default_factory=_utc_now)


class Pages(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("chapter_id", "page_number", name="uq_pages_chapter_page"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter_id: int = Field(foreign_key="chapters.id", index=True)
    page_number: int = Field(index=True)
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    updated_at: Optional[datetime] = Field(default_factory=_utc_now)


class Segments(SQLModel, table=True):
    __table_args__ = (
        Index("ix_segments_page_segment", "page_id", "segment_index"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    page_id: int = Field(foreign_key="pages.id", index=True)
    segment_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    original_text: str
    translated_text: str
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
