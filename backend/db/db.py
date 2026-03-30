"""
Database: SQL (PostgreSQL) via SQLModel.

Schema: Users, reading_list_collection, reading_list_item; Manga → Chapters → Pages → Segments

Utility Functions:

_get_db_url(db_url=None):
    Returns the database URL string.
    If db_url is provided, uses that; otherwise, attempts to read from the DATABASE_URL environment variable.
    Raises ValueError if no URL is found.

get_engine(db_url=None):
    Returns a SQLModel engine connected to the specified database.
    Uses a cached engine if db_url is not provided. If db_url is provided, creates and returns a new engine.

get_connection(db_url=None):
    Returns a new Session object (database connection) for the target database engine.
    The caller should use the session as a context manager or close it manually.

init_db(db_url=None):
    Initializes the database schema by creating tables for all SQLModel models
    (Users, ReadingListCollection, ReadingListItem, Manga, Chapters, Pages, Segments).

Other Core Database Functions:

save_page_translation(provider_id, manga_title, chapter_number, page_number, bubbles, language_code, db_url=None):
    Saves text bubble/segment data for a specific manga page to the database.

get_segments(provider_id=None, manga_title=None, chapter_number=None, page_number=None, db_url=None):
    Returns all segments (bubbles) matching the specified filters.

get_chapter_segments(provider_id, manga_title, chapter_number, db_url=None):
    Returns all segments for every page in a single chapter.

list_entries(db_url=None, order_by="created_at", order_desc=True):
    Lists chapters with provider_id, manga_title, chapter_number, and last_updated.

delete_page_segments(provider_id, manga_title, chapter_number, page_number, db_url=None):
    Removes all segments for a specific page in a chapter.

delete_chapter_segments(provider_id, manga_title, chapter_number, db_url=None):
    Removes all pages and segments for an entire chapter.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID
from sqlalchemy import desc, func
from sqlmodel import Session, SQLModel, create_engine, select

from .const import PROVIDER_IDS, PROVIDER_LOCAL, PROVIDER_MANGADEX
from .models import (
    Manga,
    MangaSource,
    Chapters,
    Pages,
    Segments,
    ReadingListCollection,
    ReadingListItem,
    Users,
)
from .schemas import (
    ChapterListOut,
    ReadingListCollectionOut,
    ReadingListItemOut,
    SegmentListOut,
)


def _validate_provider_id(provider_id: str) -> None:
    if provider_id not in PROVIDER_IDS:
        raise ValueError(f"provider_id must be one of {sorted(PROVIDER_IDS)}, got {provider_id!r}")


try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

_engine = None


def _get_db_url(db_url=None):
    url = db_url or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError(
            "No database URL. Set DATABASE_URL (e.g. postgresql://user:password@host:5432/dbname) or pass db_url=..."
        )
    return url


def get_engine(db_url=None):
    global _engine
    url = _get_db_url(db_url)
    if db_url is not None:
        return create_engine(url, pool_size=5, max_overflow=10)
    if _engine is None:
        _engine = create_engine(url, pool_size=5, max_overflow=10)
    return _engine


def get_connection(db_url=None):
    """Return a new Session (caller should use as context manager or close)."""
    return Session(get_engine(db_url))


def init_db(db_url=None):
    """Create tables: users, reading_list_collection, reading_list_item, manga, manga_source, chapters, pages, segments."""
    engine = get_engine(db_url)
    SQLModel.metadata.create_all(engine)


def _is_placeholder_external_manga_id(external_manga_id: str) -> bool:
    """Synthetic ids we generate when the provider has no catalog key (local-*, legacy-*)."""
    return external_manga_id.startswith("legacy-") or external_manga_id.startswith("local-")


def _synthetic_external_manga_id(provider_id: str, manga_id: int) -> str:
    """Stable per-umbrella id for manga_source when no real external id was passed."""
    if provider_id == PROVIDER_LOCAL:
        return f"local-{manga_id}"
    return f"legacy-{manga_id}"


def _ensure_manga_source(session, manga_id: int, provider_id: str, external_manga_id: str) -> None:
    stmt = select(MangaSource).where(
        MangaSource.manga_id == manga_id,
        MangaSource.provider_id == provider_id,
    )
    row = session.exec(stmt).first()
    if row:
        if external_manga_id and not _is_placeholder_external_manga_id(external_manga_id):
            row.external_manga_id = external_manga_id
            session.add(row)
            session.flush()
        return
    session.add(
        MangaSource(
            manga_id=manga_id,
            provider_id=provider_id,
            external_manga_id=external_manga_id,
        )
    )
    session.flush()


def resolve_manga_id(
    session,
    provider_id: str,
    manga_title: str,
    external_manga_id: Optional[str],
) -> int:
    """Find or create umbrella manga and ensure a MangaSource row for this provider."""
    ext_in = (external_manga_id or "").strip()
    if ext_in:
        src = session.exec(
            select(MangaSource).where(
                MangaSource.provider_id == provider_id,
                MangaSource.external_manga_id == ext_in,
            )
        ).first()
        if src:
            return src.manga_id

    row = session.exec(select(Manga).where(Manga.manga_title == manga_title)).first()
    if row:
        synth = _synthetic_external_manga_id(provider_id, row.id)
        _ensure_manga_source(session, row.id, provider_id, ext_in or synth)
        return row.id

    m = Manga(manga_title=manga_title)
    session.add(m)
    session.flush()
    synth = _synthetic_external_manga_id(provider_id, m.id)
    _ensure_manga_source(session, m.id, provider_id, ext_in or synth)
    return m.id


def resolve_manga_id_by_source(provider_id: str, external_manga_id: str, db_url=None) -> Optional[int]:
    """Return umbrella manga id for a provider catalog id, if mapped."""
    _validate_provider_id(provider_id)
    ext = external_manga_id.strip()
    if not ext:
        return None
    engine = get_engine(db_url)
    with Session(engine) as session:
        src = session.exec(
            select(MangaSource).where(
                MangaSource.provider_id == provider_id,
                MangaSource.external_manga_id == ext,
            )
        ).first()
        return src.manga_id if src else None


def _get_or_create_chapter(
    session, manga_id: int, provider_id: str, chapter_number: float, language_code: str
) -> int:
    stmt = select(Chapters).where(
        Chapters.manga_id == manga_id,
        Chapters.provider_id == provider_id,
        Chapters.chapter_number == chapter_number,
    )
    row = session.exec(stmt).first()
    if row:
        return row.id
    ch = Chapters(
        manga_id=manga_id,
        provider_id=provider_id,
        chapter_number=chapter_number,
        language_code=language_code,
    )
    session.add(ch)
    session.flush()
    return ch.id


def _get_or_create_page(session, chapter_id: int, page_number: int) -> int:
    stmt = select(Pages).where(Pages.chapter_id == chapter_id, Pages.page_number == page_number)
    row = session.exec(stmt).first()
    if row:
        return row.id
    p = Pages(chapter_id=chapter_id, page_number=page_number)
    session.add(p)
    session.flush()
    return p.id


def save_page_translation(
    provider_id: str,
    manga_title: str,
    chapter_number: float,
    page_number: int,
    bubbles: list[dict],
    language_code: str,
    *,
    external_manga_id: Optional[str] = None,
    db_url=None,
) -> None:
    """Save one page's segments (replace existing for this provider/manga/chapter/page)."""
    _validate_provider_id(provider_id)
    engine = get_engine(db_url)
    with Session(engine) as session:
        manga_id = resolve_manga_id(session, provider_id, manga_title, external_manga_id)
        chapter_id = _get_or_create_chapter(session, manga_id, provider_id, chapter_number, language_code)
        page_id = _get_or_create_page(session, chapter_id, page_number)
        for s in session.exec(select(Segments).where(Segments.page_id == page_id)).all():
            session.delete(s)
        session.flush()
        for b in bubbles:
            seg = Segments(
                page_id=page_id,
                segment_index=b["bubble_index"],
                x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
                original_text=b["original_text"],
                translated_text=b["translated_text"],
            )
            session.add(seg)
        now = datetime.now(timezone.utc)
        page = session.get(Pages, page_id)
        if page:
            page.updated_at = now
            chapter = session.get(Chapters, page.chapter_id)
            if chapter:
                chapter.updated_at = now
                manga = session.get(Manga, chapter.manga_id)
                if manga:
                    manga.updated_at = now
        session.commit()


def delete_page_segments(
    provider_id: str,
    manga_title: str,
    chapter_number: float,
    page_number: int,
    db_url=None,
) -> None:
    """Delete all segments for one page and the page row."""
    _validate_provider_id(provider_id)
    engine = get_engine(db_url)
    with Session(engine) as session:
        m = session.exec(select(Manga).where(Manga.manga_title == manga_title)).first()
        if not m:
            return
        ch_stmt = select(Chapters).where(
            Chapters.manga_id == m.id,
            Chapters.provider_id == provider_id,
            Chapters.chapter_number == chapter_number,
        )
        ch = session.exec(ch_stmt).first()
        if not ch:
            return
        page_stmt = select(Pages).where(Pages.chapter_id == ch.id, Pages.page_number == page_number)
        page = session.exec(page_stmt).first()
        if not page:
            return
        for seg in session.exec(select(Segments).where(Segments.page_id == page.id)).all():
            session.delete(seg)
        session.flush()
        session.delete(page)
        session.commit()


def delete_chapter_segments(
    provider_id: str,
    manga_title: str,
    chapter_number: float,
    db_url=None,
) -> None:
    """Delete chapter and all its pages/segments (explicit deletes; DB may not have CASCADE)."""
    _validate_provider_id(provider_id)
    engine = get_engine(db_url)
    with Session(engine) as session:
        m = session.exec(select(Manga).where(Manga.manga_title == manga_title)).first()
        if not m:
            return
        ch = session.exec(
            select(Chapters).where(
                Chapters.manga_id == m.id,
                Chapters.provider_id == provider_id,
                Chapters.chapter_number == chapter_number,
            )
        ).first()
        if not ch:
            return
        pages = list(session.exec(select(Pages).where(Pages.chapter_id == ch.id)).all())
        for page in pages:
            for seg in session.exec(select(Segments).where(Segments.page_id == page.id)).all():
                session.delete(seg)
        session.flush()
        for page in pages:
            session.delete(page)
        session.flush()
        session.delete(ch)
        session.commit()


def delete_all_manga(db_url=None) -> None:
    """Delete all segments, pages, chapters, reading-list rows, manga_source rows, and manga rows."""
    engine = get_engine(db_url)
    with Session(engine) as session:
        for seg in session.exec(select(Segments)).all():
            session.delete(seg)
        for p in session.exec(select(Pages)).all():
            session.delete(p)
        for ch in session.exec(select(Chapters)).all():
            session.delete(ch)
        for rli in session.exec(select(ReadingListItem)).all():
            session.delete(rli)
        for rlc in session.exec(select(ReadingListCollection)).all():
            session.delete(rlc)
        for ms in session.exec(select(MangaSource)).all():
            session.delete(ms)
        for m in session.exec(select(Manga)).all():
            session.delete(m)
        session.commit()


def get_segments(
    provider_id: Optional[str] = None,
    manga_title: Optional[str] = None,
    chapter_number: Optional[float] = None,
    page_number: Optional[int] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    db_url=None,
) -> list[SegmentListOut]:
    """Query segments; provider_id filters on chapter. Supports limit/offset for pagination."""
    if provider_id is not None:
        _validate_provider_id(provider_id)
    engine = get_engine(db_url)
    with Session(engine) as session:
        stmt = (
            select(
                Segments.id,
                Chapters.provider_id,
                Manga.manga_title,
                Chapters.chapter_number,
                Pages.page_number,
                Segments.segment_index,
                Segments.x1, Segments.y1, Segments.x2, Segments.y2,
                Segments.original_text,
                Segments.translated_text,
                Chapters.language_code,
                Segments.created_at,
            )
            .select_from(Segments)
            .join(Pages, Segments.page_id == Pages.id)
            .join(Chapters, Pages.chapter_id == Chapters.id)
            .join(Manga, Chapters.manga_id == Manga.id)
        )
        if provider_id is not None:
            stmt = stmt.where(Chapters.provider_id == provider_id)
        if manga_title is not None:
            stmt = stmt.where(Manga.manga_title == manga_title)
        if chapter_number is not None:
            stmt = stmt.where(Chapters.chapter_number == chapter_number)
        if page_number is not None:
            stmt = stmt.where(Pages.page_number == page_number)
        stmt = stmt.order_by(Chapters.chapter_number, Pages.page_number, Segments.segment_index)
        if offset > 0:
            stmt = stmt.offset(offset)
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)
        rows = session.exec(stmt).all()
        return [
            SegmentListOut(
                id=r[0], provider_id=r[1], manga_title=r[2], chapter_number=r[3],
                page_number=r[4], segment_index=r[5], x1=r[6], y1=r[7], x2=r[8],
                y2=r[9], original_text=r[10], translated_text=r[11], language_code=r[12],
                created_at=r[13],
            )
            for r in rows
        ]


def get_chapter_segments(
    provider_id: str,
    manga_title: str,
    chapter_number: float,
    limit: Optional[int] = None,
    offset: int = 0,
    db_url=None,
) -> list[SegmentListOut]:
    """Get all segments for a chapter. Supports limit/offset for pagination."""
    _validate_provider_id(provider_id)
    return get_segments(
        provider_id=provider_id,
        manga_title=manga_title,
        chapter_number=chapter_number,
        limit=limit,
        offset=offset,
        db_url=db_url,
    )


def list_chapters(
    manga_title: str,
    provider_id: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    db_url=None,
) -> list[ChapterListOut]:
    """List chapters for an umbrella manga_title; optionally filter by chapter provider_id."""
    engine = get_engine(db_url)
    with Session(engine) as session:
        stmt = (
            select(Manga.manga_title, Chapters.provider_id, Chapters.id, Chapters.chapter_number, Chapters.created_at, Chapters.updated_at)
            .select_from(Chapters)
            .join(Manga, Chapters.manga_id == Manga.id)
            .where(Manga.manga_title == manga_title)
        )
        if provider_id is not None:
            _validate_provider_id(provider_id)
            stmt = stmt.where(Chapters.provider_id == provider_id)
        if offset > 0:
            stmt = stmt.offset(offset)
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)
        rows = session.exec(stmt).all()
        return [
            ChapterListOut(
                manga_title=r[0], provider_id=r[1], id=r[2], chapter_number=r[3],
                created_at=r[4], updated_at=r[5],
            )
            for r in rows
        ]


def list_mangas(
    db_url=None,
    order_by: str = "created_at",
    order_desc: bool = True,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[Manga]:
    """
    List umbrella Manga rows.
    order_by: one of "manga_title", "created_at", "updated_at"
    """
    engine = get_engine(db_url)
    order_map = {
        "manga_title": Manga.manga_title,
        "created_at": Manga.created_at,
        "updated_at": Manga.updated_at,
    }
    col = order_map.get(order_by, Manga.updated_at)
    if order_desc:
        col = desc(col)
    with Session(engine) as session:
        stmt = select(Manga).order_by(col)
        if offset > 0:
            stmt = stmt.offset(offset)
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)
        return list(session.exec(stmt).all())


def _normalize_reading_list_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        raise ValueError("Name is required")
    if len(n) > 200:
        raise ValueError("Name is too long")
    return n


def ensure_app_user(
    user_id: UUID, email: Optional[str] = None, db_url=None
) -> None:
    """Ensure public.users has a row for this Supabase auth user (required for reading_list FK)."""
    engine = get_engine(db_url)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        row = session.exec(select(Users).where(Users.id == user_id)).first()
        if row is not None:
            if email and row.email != email:
                row.email = email
                row.updated_at = now
                session.add(row)
                session.commit()
            return
        session.add(
            Users(
                id=user_id,
                email=email,
                display_name=None,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()


def update_app_user_display_name(
    user_id: UUID,
    display_name: Optional[str],
    email: Optional[str] = None,
    db_url=None,
) -> None:
    """Sync display_name on public.users (client updates Supabase Auth metadata separately)."""
    trimmed = (display_name or "").strip() or None
    if trimmed and len(trimmed) > 200:
        raise ValueError("Display name is too long")
    ensure_app_user(user_id, email=email, db_url=db_url)
    engine = get_engine(db_url)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        row = session.exec(select(Users).where(Users.id == user_id)).first()
        if row is None:
            return
        row.display_name = trimmed
        row.updated_at = now
        session.add(row)
        session.commit()


def get_reading_list_collection(
    user_id: UUID, collection_id: int, db_url=None
) -> Optional[ReadingListCollection]:
    engine = get_engine(db_url)
    with Session(engine) as session:
        return session.exec(
            select(ReadingListCollection).where(
                ReadingListCollection.id == collection_id,
                ReadingListCollection.user_id == user_id,
            )
        ).first()


def create_reading_list_collection(
    user_id: UUID, name: str, db_url=None, *, user_email: Optional[str] = None
) -> ReadingListCollection:
    ensure_app_user(user_id, email=user_email, db_url=db_url)
    n = _normalize_reading_list_name(name)
    now = datetime.now(timezone.utc)
    col = ReadingListCollection(user_id=user_id, name=n, created_at=now, updated_at=now)
    engine = get_engine(db_url)
    with Session(engine) as session:
        session.add(col)
        session.commit()
        session.refresh(col)
        return col


def update_reading_list_collection(
    user_id: UUID, collection_id: int, name: str, db_url=None
) -> Optional[ReadingListCollection]:
    n = _normalize_reading_list_name(name)
    engine = get_engine(db_url)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        col = session.exec(
            select(ReadingListCollection).where(
                ReadingListCollection.id == collection_id,
                ReadingListCollection.user_id == user_id,
            )
        ).first()
        if not col:
            return None
        col.name = n
        col.updated_at = now
        session.add(col)
        session.commit()
        session.refresh(col)
        return col


def delete_reading_list_collection(user_id: UUID, collection_id: int, db_url=None) -> bool:
    engine = get_engine(db_url)
    with Session(engine) as session:
        col = session.exec(
            select(ReadingListCollection).where(
                ReadingListCollection.id == collection_id,
                ReadingListCollection.user_id == user_id,
            )
        ).first()
        if not col:
            return False
        for item in session.exec(
            select(ReadingListItem).where(
                ReadingListItem.reading_list_id == collection_id
            )
        ).all():
            session.delete(item)
        session.delete(col)
        session.commit()
        return True


def list_reading_list_collections_with_counts(
    user_id: UUID, db_url=None
) -> list[ReadingListCollectionOut]:
    engine = get_engine(db_url)
    with Session(engine) as session:
        cols = list(
            session.exec(
                select(ReadingListCollection)
                .where(ReadingListCollection.user_id == user_id)
                .order_by(desc(ReadingListCollection.updated_at))
            ).all()
        )
        if not cols:
            return []
        ids = [c.id for c in cols if c.id is not None]
        cnt_stmt = (
            select(ReadingListItem.reading_list_id, func.count(ReadingListItem.id))
            .where(ReadingListItem.reading_list_id.in_(ids))
            .group_by(ReadingListItem.reading_list_id)
        )
        count_map = {rid: int(n) for rid, n in session.exec(cnt_stmt).all()}
        latest_ext: dict[int, Optional[str]] = {}
        if ids:
            rows = list(
                session.exec(
                    select(ReadingListItem, MangaSource)
                    .join(Manga, ReadingListItem.manga_id == Manga.id)
                    .outerjoin(
                        MangaSource,
                        (MangaSource.manga_id == Manga.id)
                        & (MangaSource.provider_id == PROVIDER_MANGADEX),
                    )
                    .where(ReadingListItem.reading_list_id.in_(ids))
                    .order_by(
                        ReadingListItem.reading_list_id,
                        desc(ReadingListItem.created_at),
                    )
                ).all()
            )
            for item, src in rows:
                lid = item.reading_list_id
                if lid in latest_ext:
                    continue
                ext: Optional[str] = None
                if (
                    src
                    and src.external_manga_id
                    and not _is_placeholder_external_manga_id(
                        src.external_manga_id
                    )
                ):
                    ext = src.external_manga_id
                latest_ext[lid] = ext
    return [
        ReadingListCollectionOut(
            id=c.id,
            name=c.name,
            created_at=c.created_at,
            updated_at=c.updated_at,
            manga_count=count_map.get(c.id, 0),
            latest_external_manga_id=latest_ext.get(c.id),
        )
        for c in cols
    ]


def list_reading_list_items_with_manga(
    user_id: UUID,
    reading_list_id: int,
    *,
    db_url=None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[ReadingListItemOut]:
    if get_reading_list_collection(user_id, reading_list_id, db_url) is None:
        return []
    engine = get_engine(db_url)
    with Session(engine) as session:
        stmt = (
            select(ReadingListItem, Manga, MangaSource)
            .join(Manga, ReadingListItem.manga_id == Manga.id)
            .outerjoin(
                MangaSource,
                (MangaSource.manga_id == Manga.id)
                & (MangaSource.provider_id == PROVIDER_MANGADEX),
            )
            .where(ReadingListItem.reading_list_id == reading_list_id)
            .order_by(desc(ReadingListItem.updated_at))
        )
        if offset > 0:
            stmt = stmt.offset(offset)
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)
        rows = session.exec(stmt).all()
    out: list[ReadingListItemOut] = []
    for item, manga, src in rows:
        ext = None
        if src and src.external_manga_id and not _is_placeholder_external_manga_id(
            src.external_manga_id
        ):
            ext = src.external_manga_id
        out.append(
            ReadingListItemOut(
                id=item.id,
                reading_list_id=reading_list_id,
                manga_id=manga.id,
                manga_title=manga.manga_title,
                external_manga_id=ext,
                last_chapter_number=item.last_chapter_number,
                updated_at=item.updated_at,
            )
        )
    return out


def upsert_reading_list_item(
    user_id: UUID,
    reading_list_id: int,
    manga_id: int,
    *,
    last_chapter_number: Optional[float] = None,
    db_url=None,
) -> ReadingListItem:
    engine = get_engine(db_url)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        col = session.exec(
            select(ReadingListCollection).where(
                ReadingListCollection.id == reading_list_id,
                ReadingListCollection.user_id == user_id,
            )
        ).first()
        if not col:
            raise ValueError("Reading list not found")
        stmt = select(ReadingListItem).where(
            ReadingListItem.reading_list_id == reading_list_id,
            ReadingListItem.manga_id == manga_id,
        )
        row = session.exec(stmt).first()
        if row:
            if last_chapter_number is not None:
                row.last_chapter_number = last_chapter_number
            row.updated_at = now
            session.add(row)
            col.updated_at = now
            session.add(col)
            session.commit()
            session.refresh(row)
            return row
        entry = ReadingListItem(
            reading_list_id=reading_list_id,
            manga_id=manga_id,
            last_chapter_number=last_chapter_number,
            created_at=now,
            updated_at=now,
        )
        session.add(entry)
        col.updated_at = now
        session.add(col)
        session.commit()
        session.refresh(entry)
        return entry


def remove_reading_list_item_from_list(
    user_id: UUID, reading_list_id: int, manga_id: int, db_url=None
) -> bool:
    engine = get_engine(db_url)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        col = session.exec(
            select(ReadingListCollection).where(
                ReadingListCollection.id == reading_list_id,
                ReadingListCollection.user_id == user_id,
            )
        ).first()
        if not col:
            return False
        stmt = select(ReadingListItem).where(
            ReadingListItem.reading_list_id == reading_list_id,
            ReadingListItem.manga_id == manga_id,
        )
        row = session.exec(stmt).first()
        if not row:
            return False
        session.delete(row)
        col.updated_at = now
        session.add(col)
        session.commit()
        return True


def add_reading_list_item_by_source(
    user_id: UUID,
    reading_list_id: int,
    provider_id: str,
    external_manga_id: str,
    manga_title: str,
    *,
    last_chapter_number: Optional[float] = None,
    db_url=None,
) -> ReadingListItem:
    _validate_provider_id(provider_id)
    ext = (external_manga_id or "").strip()
    title = (manga_title or "").strip() or "Untitled"
    if not ext:
        raise ValueError("external_manga_id is required")
    engine = get_engine(db_url)
    with Session(engine) as session:
        manga_id = resolve_manga_id(session, provider_id, title, ext)
        session.commit()
    return upsert_reading_list_item(
        user_id,
        reading_list_id,
        manga_id,
        last_chapter_number=last_chapter_number,
        db_url=db_url,
    )
