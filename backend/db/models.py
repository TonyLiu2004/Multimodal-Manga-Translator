"""
User table: app profile keyed by Supabase auth user id (UUID)
  id: UUID, primary key — use auth.users.id when using Supabase Auth
  email: optional, unique when set
  display_name: optional
  created_at / updated_at: datetime

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

Chapter table: chapter data scoped by provider (same umbrella manga can have chapters per source)

Page / Segment tables: unchanged hierarchy under chapters

ReadingListEntry: one row per user + umbrella manga (manga_id)
"""
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import Index
from sqlmodel import Field, SQLModel, UniqueConstraint


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Users(SQLModel, table=True):
    __table_args__ = ({"extend_existing": True},)

    id: UUID = Field(primary_key=True)
    email: Optional[str] = Field(default=None, unique=True)
    display_name: Optional[str] = Field(default=None)
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    updated_at: Optional[datetime] = Field(default_factory=_utc_now)


class Manga(SQLModel, table=True):
    """Umbrella series; use MangaSource for per-provider external ids."""

    __table_args__ = ({"extend_existing": True},)

    id: Optional[int] = Field(default=None, primary_key=True)
    manga_title: str = Field(index=True)
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    updated_at: Optional[datetime] = Field(default_factory=_utc_now)


class MangaSource(SQLModel, table=True):
    """Links an umbrella manga to a provider catalog id (e.g. MangaDex UUID)."""

    __tablename__ = "manga_source"
    __table_args__ = (
        UniqueConstraint("provider_id", "external_manga_id", name="uq_manga_source_provider_external"),
        UniqueConstraint("manga_id", "provider_id", name="uq_manga_source_manga_provider"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    manga_id: int = Field(foreign_key="manga.id", index=True)
    provider_id: str = Field(index=True)
    external_manga_id: str = Field(index=True) #if there is no external manga id, it will be the same as the manga_id with local in front
    created_at: Optional[datetime] = Field(default_factory=_utc_now)


class ReadingListEntry(SQLModel, table=True):
    __tablename__ = "reading_list"
    __table_args__ = (
        UniqueConstraint("user_id", "manga_id", name="uq_reading_list_user_manga"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: UUID = Field(foreign_key="users.id")
    manga_id: int = Field(foreign_key="manga.id", index=True)
    last_chapter_number: Optional[float] = Field(default=None)
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    updated_at: Optional[datetime] = Field(default_factory=_utc_now)


class Chapters(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("manga_id", "provider_id", "chapter_number", name="uq_chapters_manga_provider_chapter"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    manga_id: int = Field(foreign_key="manga.id", index=True)
    provider_id: str = Field(index=True)
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
