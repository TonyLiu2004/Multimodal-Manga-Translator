"""
Database: SQL (PostgreSQL) via SQLModel.

Schema: Users, reading_list_collection, reading_list_item; Manga → Chapters → Panels

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
    (Users, ReadingListCollection, ReadingListItem, Manga, Chapters, Panels).

Other Core Database Functions:

save_panels_translation(manga_title, chapter_number, page_number, panels, language_code, db_url=None):
    Saves text bubble/panel data for a specific manga page to the database.

get_panels(manga_title=None, chapter_number=None, page_number=None, db_url=None):
    Returns all panels (bubbles) matching the specified filters.

get_chapter_panels(manga_title, chapter_number, db_url=None):
    Returns all panels for every page in a single chapter.

list_entries(db_url=None, order_by="created_at", order_desc=True):
    Lists chapters with manga_title, chapter_number, and last_updated.

delete_page_panels(manga_title, chapter_number, page_number, db_url=None):
    Removes all panels for a specific page in a chapter.

delete_chapter_panels(manga_title, chapter_number, db_url=None):
    Removes all panels for an entire chapter.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID
from sqlalchemy import desc, func
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, create_engine, select

# Single-provider setup (MangaDex).
from .models import (
    Manga,
    Chapters,
    Panels,
    ReadingListCollection,
    ReadingListItem,
    Users,
)
from .schemas import (
    ChapterListOut,
    ReadingListCollectionOut,
    ReadingListItemOut,
    PanelListOut,
)


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
    """Create tables: users, reading_list_collection, reading_list_item, manga, chapters, panels."""
    engine = get_engine(db_url)
    SQLModel.metadata.create_all(engine)


def resolve_manga_id(
    session,
    manga_title: str,
    external_manga_id: Optional[str],
) -> int:
    """Find or create umbrella manga.

    If external_manga_id is provided, it is treated as MangaDex series id and stored
    on Manga.mangadex_manga_id for later lookups.

    MangaDex ids are normalized to lowercase so lookups stay stable. Concurrent
    inserts for the same series id are merged via DB uniqueness + IntegrityError
    handling (important when one title is added to several lists in parallel).
    """
    title_stripped = (manga_title or "").strip() or "Untitled"
    ext_raw = (external_manga_id or "").strip()
    norm_ext = ext_raw.lower() if ext_raw else ""

    if norm_ext:
        row = session.exec(
            select(Manga).where(func.lower(Manga.mangadex_manga_id) == norm_ext),
        ).first()
        if row:
            return row.id

    row = session.exec(
        select(Manga).where(Manga.manga_title == title_stripped),
    ).first()
    if row:
        if norm_ext:
            existing_norm = (row.mangadex_manga_id or "").strip().lower()
            if existing_norm != norm_ext:
                row.mangadex_manga_id = norm_ext
                row.updated_at = datetime.now(timezone.utc)
                session.add(row)
                session.flush()
        return row.id

    if norm_ext:
        try:
            with session.begin_nested():
                m = Manga(manga_title=title_stripped, mangadex_manga_id=norm_ext)
                session.add(m)
                session.flush()
                return m.id
        except IntegrityError:
            dup = session.exec(
                select(Manga).where(
                    func.lower(Manga.mangadex_manga_id) == norm_ext,
                ),
            ).first()
            if dup is None:
                raise
            return dup.id

    m = Manga(manga_title=title_stripped, mangadex_manga_id=None)
    session.add(m)
    session.flush()
    return m.id


def _get_or_create_chapter(
    session,
    manga_id: int,
    chapter_number: float,
    language_code: str,
    *,
    mangadex_chapter_id: Optional[str] = None,
) -> int:
    stmt = select(Chapters).where(
        Chapters.manga_id == manga_id,
        Chapters.chapter_number == chapter_number,
    )
    row = session.exec(stmt).first()
    if row:
        if mangadex_chapter_id is not None:
            v = mangadex_chapter_id.strip()
            if v and row.mangadex_chapter_id != v:
                row.mangadex_chapter_id = v
                row.updated_at = datetime.now(timezone.utc)
                session.add(row)
                session.flush()
        return row.id
    ch = Chapters(
        manga_id=manga_id,
        chapter_number=chapter_number,
        mangadex_chapter_id=(mangadex_chapter_id.strip() if mangadex_chapter_id else None),
        language_code=language_code,
    )
    session.add(ch)
    session.flush()
    return ch.id


def save_panels_translation(
    manga_title: str,
    chapter_number: float,
    page_number: Optional[int],
    panels: list[dict],
    language_code: str,
    *,
    external_manga_id: Optional[str] = None,
    mangadex_chapter_id: Optional[str] = None,
    db_url=None,
) -> None:
    """Save panels (replace existing for this manga/chapter/page)."""
    engine = get_engine(db_url)
    with Session(engine) as session:
        manga_id = resolve_manga_id(session, manga_title, external_manga_id)
        chapter_id = _get_or_create_chapter(
            session,
            manga_id,
            chapter_number,
            language_code,
            mangadex_chapter_id=mangadex_chapter_id,
        )
        for p in session.exec(
            select(Panels).where(Panels.chapter_id == chapter_id, Panels.page_number == page_number)
        ).all():
            session.delete(p)
        session.flush()
        for panel in panels:
            row = Panels(
                chapter_id=chapter_id,
                page_number=page_number,
                mangadex_chapter_id=(mangadex_chapter_id.strip() if mangadex_chapter_id else None),
                bubble_index=panel["bubble_index"],
                width=panel.get("width"),
                height=panel.get("height"),
                panel_url=panel.get("panel_url"),
                x1=panel["x1"],
                y1=panel["y1"],
                x2=panel["x2"],
                y2=panel["y2"],
                original_text=panel["original_text"],
                translated_text=panel["translated_text"],
            )
            session.add(row)
        now = datetime.now(timezone.utc)
        chapter = session.get(Chapters, chapter_id)
        if chapter:
            chapter.updated_at = now
            manga = session.get(Manga, chapter.manga_id)
            if manga:
                manga.updated_at = now
        session.commit()


# Backwards-compatible alias (older code/tests may still import this name).
save_panel_translation = save_panels_translation


def upsert_panel_translation(
    manga_title: str,
    chapter_number: float,
    page_number: Optional[int],
    panel: dict,
    language_code: str,
    *,
    external_manga_id: Optional[str] = None,
    mangadex_chapter_id: Optional[str] = None,
    db_url=None,
) -> None:
    """Insert/update a single panel without deleting other panels.

    If panel_url is provided, uniqueness is treated as (chapter_id, page_number, panel_url).
    Otherwise falls back to (chapter_id, page_number, bubble_index).
    """
    engine = get_engine(db_url)
    with Session(engine) as session:
        manga_id = resolve_manga_id(session, manga_title, external_manga_id)
        chapter_id = _get_or_create_chapter(
            session,
            manga_id,
            chapter_number,
            language_code,
            mangadex_chapter_id=mangadex_chapter_id,
        )

        bubble_index = panel["bubble_index"]
        panel_url = panel.get("panel_url")
        panel_url = panel_url.strip() if isinstance(panel_url, str) else None
        if panel_url:
            existing = session.exec(
                select(Panels).where(
                    Panels.chapter_id == chapter_id,
                    Panels.page_number == page_number,
                    Panels.panel_url == panel_url,
                )
            ).first()
        else:
            existing = session.exec(
                select(Panels).where(
                    Panels.chapter_id == chapter_id,
                    Panels.page_number == page_number,
                    Panels.bubble_index == bubble_index,
                )
            ).first()

        mdx = mangadex_chapter_id.strip() if mangadex_chapter_id else None
        if existing:
            existing.mangadex_chapter_id = mdx
            existing.panel_url = panel_url
            existing.bubble_index = bubble_index
            existing.width = panel.get("width")
            existing.height = panel.get("height")
            existing.x1 = panel["x1"]
            existing.y1 = panel["y1"]
            existing.x2 = panel["x2"]
            existing.y2 = panel["y2"]
            existing.original_text = panel["original_text"]
            existing.translated_text = panel["translated_text"]
            session.add(existing)
        else:
            row = Panels(
                chapter_id=chapter_id,
                page_number=page_number,
                mangadex_chapter_id=mdx,
                bubble_index=bubble_index,
                width=panel.get("width"),
                height=panel.get("height"),
                panel_url=panel_url,
                x1=panel["x1"],
                y1=panel["y1"],
                x2=panel["x2"],
                y2=panel["y2"],
                original_text=panel["original_text"],
                translated_text=panel["translated_text"],
            )
            session.add(row)

        now = datetime.now(timezone.utc)
        chapter = session.get(Chapters, chapter_id)
        if chapter:
            chapter.updated_at = now
            manga = session.get(Manga, chapter.manga_id)
            if manga:
                manga.updated_at = now
        session.commit()


def delete_page_panels(
    manga_title: str,
    chapter_number: float,
    page_number: Optional[int],
    db_url=None,
) -> None:
    """Delete all panels for one page."""
    engine = get_engine(db_url)
    with Session(engine) as session:
        m = session.exec(select(Manga).where(Manga.manga_title == manga_title)).first()
        if not m:
            return
        ch_stmt = select(Chapters).where(Chapters.manga_id == m.id, Chapters.chapter_number == chapter_number)
        ch = session.exec(ch_stmt).first()
        if not ch:
            return
        for p in session.exec(
            select(Panels).where(Panels.chapter_id == ch.id, Panels.page_number == page_number)
        ).all():
            session.delete(p)
        session.commit()


def delete_chapter_panels(
    manga_title: str,
    chapter_number: float,
    db_url=None,
) -> None:
    """Delete chapter and all its panels (explicit deletes; DB may not have CASCADE)."""
    engine = get_engine(db_url)
    with Session(engine) as session:
        m = session.exec(select(Manga).where(Manga.manga_title == manga_title)).first()
        if not m:
            return
        ch = session.exec(select(Chapters).where(Chapters.manga_id == m.id, Chapters.chapter_number == chapter_number)).first()
        if not ch:
            return
        for p in session.exec(select(Panels).where(Panels.chapter_id == ch.id)).all():
            session.delete(p)
        session.flush()
        session.delete(ch)
        session.commit()


def delete_all_manga(db_url=None) -> None:
    """Delete all panels, chapters, reading-list rows, and manga rows."""
    engine = get_engine(db_url)
    with Session(engine) as session:
        for p in session.exec(select(Panels)).all():
            session.delete(p)
        for ch in session.exec(select(Chapters)).all():
            session.delete(ch)
        for rli in session.exec(select(ReadingListItem)).all():
            session.delete(rli)
        for rlc in session.exec(select(ReadingListCollection)).all():
            session.delete(rlc)
        for m in session.exec(select(Manga)).all():
            session.delete(m)
        session.commit()


def get_panels(
    manga_title: Optional[str] = None,
    chapter_number: Optional[float] = None,
    page_number: Optional[int] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    db_url=None,
) -> list[PanelListOut]:
    """Query panels. Supports limit/offset for pagination.
    """
    engine = get_engine(db_url)
    with Session(engine) as session:
        stmt = (
            select(
                Panels.id,
                Manga.manga_title,
                Chapters.chapter_number,
                Chapters.mangadex_chapter_id,
                Panels.page_number,
                Panels.bubble_index,
                Panels.width,
                Panels.height,
                Panels.x1, Panels.y1, Panels.x2, Panels.y2,
                Panels.original_text,
                Panels.translated_text,
                Panels.panel_url,
                Chapters.language_code,
                Panels.created_at,
            )
            .select_from(Panels)
            .join(Chapters, Panels.chapter_id == Chapters.id)
            .join(Manga, Chapters.manga_id == Manga.id)
        )
        if manga_title is not None:
            stmt = stmt.where(Manga.manga_title == manga_title)
        if chapter_number is not None:
            stmt = stmt.where(Chapters.chapter_number == chapter_number)
        if page_number is not None:
            stmt = stmt.where(Panels.page_number == page_number)
        stmt = stmt.order_by(
            Chapters.chapter_number,
            func.coalesce(Panels.page_number, 0),
            Panels.bubble_index,
        )
        if offset > 0:
            stmt = stmt.offset(offset)
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)
        rows = session.exec(stmt).all()
        return [
            PanelListOut(
                id=r[0],
                manga_title=r[1],
                chapter_number=r[2],
                mangadex_chapter_id=r[3],
                page_number=r[4],
                bubble_index=r[5],
                width=r[6],
                height=r[7],
                x1=r[8],
                y1=r[9],
                x2=r[10],
                y2=r[11],
                original_text=r[12],
                translated_text=r[13],
                panel_url=r[14],
                language_code=r[15],
                created_at=r[16],
            )
            for r in rows
        ]


def get_panel_by_panel_url(
    panel_url: str,
    *,
    manga_title: Optional[str] = None,
    chapter_number: Optional[float] = None,
    mangadex_chapter_id: Optional[str] = None,
    page_number: Optional[int] = None,
    db_url=None,
) -> Optional[PanelListOut]:
    """Return the first matching panel for panel_url (or None)."""
    url = (panel_url or "").strip()
    if not url:
        return None
    engine = get_engine(db_url)
    with Session(engine) as session:
        stmt = (
            select(
                Panels.id,
                Manga.manga_title,
                Chapters.chapter_number,
                Chapters.mangadex_chapter_id,
                Panels.page_number,
                Panels.bubble_index,
                Panels.width,
                Panels.height,
                Panels.x1,
                Panels.y1,
                Panels.x2,
                Panels.y2,
                Panels.original_text,
                Panels.translated_text,
                Panels.panel_url,
                Chapters.language_code,
                Panels.created_at,
            )
            .select_from(Panels)
            .join(Chapters, Panels.chapter_id == Chapters.id)
            .join(Manga, Chapters.manga_id == Manga.id)
            .where(Panels.panel_url == url)
        )
        if manga_title is not None:
            stmt = stmt.where(Manga.manga_title == manga_title)
        if chapter_number is not None:
            stmt = stmt.where(Chapters.chapter_number == chapter_number)
        if mangadex_chapter_id is not None:
            v = mangadex_chapter_id.strip()
            if v:
                stmt = stmt.where(Chapters.mangadex_chapter_id == v)
        if page_number is not None:
            stmt = stmt.where(Panels.page_number == page_number)
        stmt = stmt.order_by(
            Chapters.chapter_number,
            func.coalesce(Panels.page_number, 0),
            Panels.bubble_index,
        ).limit(1)
        r = session.exec(stmt).first()
        if not r:
            return None
        return PanelListOut(
            id=r[0],
            manga_title=r[1],
            chapter_number=r[2],
            mangadex_chapter_id=r[3],
            page_number=r[4],
            bubble_index=r[5],
            width=r[6],
            height=r[7],
            x1=r[8],
            y1=r[9],
            x2=r[10],
            y2=r[11],
            original_text=r[12],
            translated_text=r[13],
            panel_url=r[14],
            language_code=r[15],
            created_at=r[16],
        )


def delete_panel_by_panel_url(
    panel_url: str,
    *,
    manga_title: Optional[str] = None,
    chapter_number: Optional[float] = None,
    mangadex_chapter_id: Optional[str] = None,
    page_number: Optional[int] = None,
    db_url=None,
) -> int:
    """Delete panel rows matching panel_url (optionally scoped).

    Returns number of deleted rows.
    """
    url = (panel_url or "").strip()
    if not url:
        return 0
    engine = get_engine(db_url)
    with Session(engine) as session:
        stmt = select(Panels).where(Panels.panel_url == url)
        if (
            manga_title is not None
            or chapter_number is not None
            or mangadex_chapter_id is not None
        ):
            stmt = stmt.join(Chapters, Panels.chapter_id == Chapters.id).join(
                Manga, Chapters.manga_id == Manga.id
            )
            if manga_title is not None:
                stmt = stmt.where(Manga.manga_title == manga_title)
            if chapter_number is not None:
                stmt = stmt.where(Chapters.chapter_number == chapter_number)
            if mangadex_chapter_id is not None:
                v = mangadex_chapter_id.strip()
                if v:
                    stmt = stmt.where(Chapters.mangadex_chapter_id == v)
        if page_number is not None:
            stmt = stmt.where(Panels.page_number == page_number)

        panels = session.exec(stmt).all()
        if not panels:
            return 0

        chapter_ids = {p.chapter_id for p in panels}
        for p in panels:
            session.delete(p)
        session.flush()

        now = datetime.now(timezone.utc)
        for cid in chapter_ids:
            chapter = session.get(Chapters, cid)
            if chapter:
                chapter.updated_at = now
                manga = session.get(Manga, chapter.manga_id)
                if manga:
                    manga.updated_at = now
        session.commit()
        return len(panels)


def get_chapter_panels(
    manga_title: str,
    chapter_number: float,
    limit: Optional[int] = None,
    offset: int = 0,
    db_url=None,
) -> list[PanelListOut]:
    """Get all panels for a chapter. Supports limit/offset for pagination."""
    return get_panels(
        manga_title=manga_title,
        chapter_number=chapter_number,
        limit=limit,
        offset=offset,
        db_url=db_url,
    )



def list_chapters(
    manga_title: str,
    limit: Optional[int] = None,
    offset: int = 0,
    db_url=None,
) -> list[ChapterListOut]:
    """List chapters for an umbrella manga_title."""
    engine = get_engine(db_url)
    with Session(engine) as session:
        stmt = (
            select(
                Manga.manga_title,
                Chapters.id,
                Chapters.chapter_number,
                Chapters.mangadex_chapter_id,
                Chapters.created_at,
                Chapters.updated_at,
            )
            .select_from(Chapters)
            .join(Manga, Chapters.manga_id == Manga.id)
            .where(Manga.manga_title == manga_title)
        )
        if offset > 0:
            stmt = stmt.offset(offset)
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)
        rows = session.exec(stmt).all()
        return [
            ChapterListOut(
                manga_title=r[0],
                id=r[1],
                chapter_number=r[2],
                mangadex_chapter_id=r[3],
                created_at=r[4],
                updated_at=r[5],
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
                    select(ReadingListItem, Manga)
                    .join(Manga, ReadingListItem.manga_id == Manga.id)
                    .where(ReadingListItem.reading_list_id.in_(ids))
                    .order_by(
                        ReadingListItem.reading_list_id,
                        desc(ReadingListItem.created_at),
                    )
                ).all()
            )
            for item, manga in rows:
                lid = item.reading_list_id
                if lid in latest_ext:
                    continue
                latest_ext[lid] = manga.mangadex_manga_id
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
            select(ReadingListItem, Manga)
            .join(Manga, ReadingListItem.manga_id == Manga.id)
            .where(ReadingListItem.reading_list_id == reading_list_id)
            .order_by(desc(ReadingListItem.updated_at))
        )
        if offset > 0:
            stmt = stmt.offset(offset)
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)
        rows = session.exec(stmt).all()
    out: list[ReadingListItemOut] = []
    for item, manga in rows:
        ext = manga.mangadex_manga_id
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
                new_val = float(last_chapter_number)
                prev = row.last_chapter_number
                if prev is None or new_val > prev:
                    row.last_chapter_number = new_val
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


def update_reading_list_item_last_read(
    user_id: UUID,
    reading_list_id: int,
    manga_id: int,
    last_chapter_number: float,
    *,
    db_url=None,
) -> bool:
    """Set last-read chapter for one list row (only increases stored value)."""
    if get_reading_list_collection(user_id, reading_list_id, db_url) is None:
        return False
    engine = get_engine(db_url)
    now = datetime.now(timezone.utc)
    new_val = float(last_chapter_number)
    with Session(engine) as session:
        row = session.exec(
            select(ReadingListItem).where(
                ReadingListItem.reading_list_id == reading_list_id,
                ReadingListItem.manga_id == manga_id,
            )
        ).first()
        if row is None:
            return False
        prev = row.last_chapter_number
        if prev is None or new_val > prev:
            row.last_chapter_number = new_val
        row.updated_at = now
        session.add(row)
        col = session.exec(
            select(ReadingListCollection).where(
                ReadingListCollection.id == reading_list_id,
                ReadingListCollection.user_id == user_id,
            )
        ).first()
        if col:
            col.updated_at = now
            session.add(col)
        session.commit()
        return True


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
    external_manga_id: str,
    manga_title: str,
    *,
    last_chapter_number: Optional[float] = None,
    db_url=None,
) -> ReadingListItem:
    ext = (external_manga_id or "").strip()
    title = (manga_title or "").strip() or "Untitled"
    if not ext:
        raise ValueError("external_manga_id is required")
    engine = get_engine(db_url)
    with Session(engine) as session:
        manga_id = resolve_manga_id(session, title, ext)
        session.commit()
    return upsert_reading_list_item(
        user_id,
        reading_list_id,
        manga_id,
        last_chapter_number=last_chapter_number,
        db_url=db_url,
    )
