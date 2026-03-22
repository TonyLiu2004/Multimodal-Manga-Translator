"""Response schemas for API endpoints. Pure Pydantic models (no table)."""

import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel


class MangaOut(SQLModel):
    """Umbrella manga list response."""

    id: Optional[int] = None
    manga_title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ChapterListOut(SQLModel):
    """Chapter list response (includes manga_title, provider_id from chapter)."""

    manga_title: str
    provider_id: str
    id: int
    chapter_number: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ReadingListEntryOut(SQLModel):
    """One saved umbrella manga on a user's reading list."""

    id: int
    user_id: uuid.UUID
    manga_id: int
    last_chapter_number: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SegmentListOut(SQLModel):
    """Segment list response (join of Segments + Pages + Chapters + Manga)."""

    id: int
    provider_id: str
    manga_title: str
    chapter_number: float
    page_number: int
    segment_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    original_text: str
    translated_text: str
    language_code: str
    created_at: Optional[datetime] = None
