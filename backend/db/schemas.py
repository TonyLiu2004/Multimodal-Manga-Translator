"""Response schemas for API endpoints. Pure Pydantic models (no table)."""

import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class MangaOut(SQLModel):
    """Umbrella manga list response."""

    id: Optional[int] = None
    manga_title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ChapterListOut(SQLModel):
    """Chapter list response (includes manga_title)."""

    manga_title: str
    id: int
    chapter_number: float
    mangadex_chapter_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ReadingListCollectionOut(SQLModel):
    """A named reading list owned by the user."""

    id: int
    name: str
    manga_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    #: MangaDex id of the most recently *added* list item (for cover thumbnails).
    latest_external_manga_id: Optional[str] = None


class ReadingListCollectionCreateIn(SQLModel):
    name: str


class ReadingListCollectionRenameIn(SQLModel):
    name: str


class UserDisplayNamePatchIn(SQLModel):
    """Display name stored in public.users and mirrored from Supabase Auth user_metadata."""

    display_name: str = Field(default="", max_length=200)


class ReadingListItemOut(SQLModel):
    """One manga row inside a named list (with umbrella title + optional MangaDex id)."""

    id: int
    reading_list_id: int
    manga_id: int
    manga_title: str
    external_manga_id: Optional[str] = None
    last_chapter_number: Optional[float] = None
    updated_at: Optional[datetime] = None


class ReadingListAddIn(SQLModel):
    """Add a title by MangaDex series id (UUID)."""

    external_manga_id: str
    manga_title: str
    last_chapter_number: Optional[float] = None


class ReadingListProgressPatchIn(SQLModel):
    """Update furthest chapter read for a title already on a list."""

    last_chapter_number: float


class PanelListOut(SQLModel):
    """Panel list response (join of Panels + Chapters + Manga)."""

    id: int
    manga_title: str
    chapter_number: float
    page_number: Optional[int] = None
    mangadex_chapter_id: Optional[str] = None
    bubble_index: int
    width: Optional[int] = None
    height: Optional[int] = None
    x1: float
    y1: float
    x2: float
    y2: float
    original_text: str
    translated_text: str
    panel_url: Optional[str] = None
    language_code: str
    created_at: Optional[datetime] = None
