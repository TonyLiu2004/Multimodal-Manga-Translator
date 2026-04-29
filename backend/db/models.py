"""
User table: app profile keyed by Supabase auth user id (UUID)
  id: UUID, primary key — same as auth.users.id when using Supabase Auth
  email: optional, unique when set
  display_name: optional
  created_at / updated_at: datetime

Manga table: stores manga metadata
  id: int, primary key
  manga_title: str
  created_at: datetime

Chapter table: stores manga chapter info
  id: int, primary key
  chapter_id: int, foreign key → chapters.id
  page_number: int
  created_at: datetime

Chapter table: chapter data (single-provider setup)

ReadingListCollection: named list per user (e.g. "Want to read").
ReadingListItem: one row per named list + umbrella manga (manga_id).
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
    """Umbrella series.

    We keep MangaDex series id directly on this table (optional) since MangaDex
    is the primary external provider used by the app.
    """

    __table_args__ = ({"extend_existing": True},)

    id: Optional[int] = Field(default=None, primary_key=True)
    manga_title: str = Field(index=True)
    mangadex_manga_id: Optional[str] = Field(default=None, index=True, unique=True)
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    updated_at: Optional[datetime] = Field(default_factory=_utc_now)


class ReadingListCollection(SQLModel, table=True):
    """User-owned named reading list (container for manga)."""

    __tablename__ = "reading_list_collection"
    __table_args__ = ({"extend_existing": True},)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: UUID = Field(foreign_key="users.id", index=True)
    name: str = Field(max_length=200)
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    updated_at: Optional[datetime] = Field(default_factory=_utc_now)


class ReadingListItem(SQLModel, table=True):
    """One umbrella manga inside a named reading list."""

    __tablename__ = "reading_list_item"
    __table_args__ = (
        UniqueConstraint("reading_list_id", "manga_id", name="uq_rli_list_manga"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    reading_list_id: int = Field(foreign_key="reading_list_collection.id", index=True)
    manga_id: int = Field(foreign_key="manga.id", index=True)
    last_chapter_number: Optional[float] = Field(default=None)
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
    # MangaDex chapter UUID (single-provider setup)
    mangadex_chapter_id: Optional[str] = Field(default=None, index=True)
    language_code: str
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    updated_at: Optional[datetime] = Field(default_factory=_utc_now)


class Panels(SQLModel, table=True):
    __tablename__ = "panels"
    __table_args__ = (
        Index("ix_panels_chapter_page_bubble", "chapter_id", "page_number", "bubble_index"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    chapter_id: int = Field(foreign_key="chapters.id", index=True)
    page_number: Optional[int] = Field(default=None, index=True)
    # Denormalized copy of MangaDex chapter UUID for convenience (optional).
    mangadex_chapter_id: Optional[str] = Field(default=None, index=True)
    # Optional URL for this panel's image (e.g. MangaDex at-home URL or proxied URL).
    panel_url: Optional[str] = Field(default=None)
    # bubble index within a page (0..N-1)
    bubble_index: int
    # page dimensions for this segment's coordinate space (optional; helpful for clients)
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    x1: float
    y1: float
    x2: float
    y2: float
    original_text: str
    translated_text: str
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
