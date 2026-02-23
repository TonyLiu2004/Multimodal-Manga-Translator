"""
Database: SQL (PostgreSQL).

Database schema:
  - chapters table(provider_id, manga_title, chapter_number, language_code)  → one row per chapter
  - pages table(chapter_id, page_number)                                    → one row per page
  - segments table(page_id, segment_index, x1, y1, x2, y2, original_text, translated_text)

Query "all segments of a chapter": join segments → pages → chapters, filter by chapter_id.
No repeated provider/manga/chapter on every segment row; indexes on chapter_id and page_id.
"""

import os
from contextlib import contextmanager
from pathlib import Path

def _load_env():
    """Load DATABASE_URL from backend/.env if not already in environment."""
    if os.environ.get("DATABASE_URL"):
        return
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key == "DATABASE_URL" and value:
                    # Remove surrounding quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ["DATABASE_URL"] = value
                    return

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass
_load_env()

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None
    RealDictCursor = None


def _get_db_url(db_url=None):
    url = db_url or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError(
            "No database URL. Set DATABASE_URL (e.g. postgresql://user:password@host:5432/dbname) or pass db_url=..."
        )
    return url


def get_connection(db_url=None):
    if psycopg2 is None:
        raise ImportError("Install psycopg2-binary: pip install psycopg2-binary")
    return psycopg2.connect(_get_db_url(db_url))


@contextmanager
def _cursor(db_url=None):
    conn = get_connection(db_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    finally:
        conn.close()


def init_db(db_url=None):
    """initialize the database tables: chapters → pages → segments."""
    with _cursor(db_url) as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chapters (
                id SERIAL PRIMARY KEY,
                provider_id TEXT NOT NULL,
                manga_title TEXT NOT NULL,
                chapter_number DOUBLE PRECISION NOT NULL,
                language_code TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider_id, manga_title, chapter_number)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id SERIAL PRIMARY KEY,
                chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chapter_id, page_number)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                id SERIAL PRIMARY KEY,
                page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                segment_index INTEGER NOT NULL,
                x1 DOUBLE PRECISION NOT NULL,
                y1 DOUBLE PRECISION NOT NULL,
                x2 DOUBLE PRECISION NOT NULL,
                y2 DOUBLE PRECISION NOT NULL,
                original_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_chapter ON pages(chapter_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_segments_page ON segments(page_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chapters_lookup ON chapters(provider_id, manga_title, chapter_number)")


def _get_or_create_chapter(provider_id: str, manga_title: str, chapter_number: float, language_code: str, cur) -> int:
    cur.execute(
        "SELECT id FROM chapters WHERE provider_id = %s AND manga_title = %s AND chapter_number = %s",
        (provider_id, manga_title, chapter_number),
    )
    row = cur.fetchone()
    if row:
        return row["id"]
    cur.execute(
        "INSERT INTO chapters (provider_id, manga_title, chapter_number, language_code) VALUES (%s, %s, %s, %s) RETURNING id",
        (provider_id, manga_title, chapter_number, language_code),
    )
    return cur.fetchone()["id"]


def _get_or_create_page(chapter_id: int, page_number: int, cur) -> int:
    cur.execute("SELECT id FROM pages WHERE chapter_id = %s AND page_number = %s", (chapter_id, page_number))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur.execute("INSERT INTO pages (chapter_id, page_number) VALUES (%s, %s) RETURNING id", (chapter_id, page_number))
    return cur.fetchone()["id"]


def save_page_translation(
    provider_id: str,
    manga_title: str,
    chapter_number: float,
    page_number: int,
    bubbles: list[dict],
    language_code: str,
    db_url=None,
) -> None:
    """
    Save one page's segments (replace existing for this provider/manga/chapter/page).
    bubbles: list of dicts with bubble_index, x1, y1, x2, y2, original_text, translated_text.
    """
    with _cursor(db_url) as cur:
        chapter_id = _get_or_create_chapter(provider_id, manga_title, chapter_number, language_code, cur)
        page_id = _get_or_create_page(chapter_id, page_number, cur)
        cur.execute("DELETE FROM segments WHERE page_id = %s", (page_id,))
        for b in bubbles:
            cur.execute(
                """
                INSERT INTO segments (page_id, segment_index, x1, y1, x2, y2, original_text, translated_text)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    page_id,
                    b["bubble_index"],
                    b["x1"],
                    b["y1"],
                    b["x2"],
                    b["y2"],
                    b["original_text"],
                    b["translated_text"],
                ),
            )


def delete_page_segments(
    provider_id: str,
    manga_title: str,
    chapter_number: float,
    page_number: int,
    db_url=None,
) -> None:
    """Delete all segments for one page (and the page row if empty)."""
    with _cursor(db_url) as cur:
        cur.execute(
            "SELECT p.id FROM pages p JOIN chapters c ON p.chapter_id = c.id "
            "WHERE c.provider_id = %s AND c.manga_title = %s AND c.chapter_number = %s AND p.page_number = %s",
            (provider_id, manga_title, chapter_number, page_number),
        )
        row = cur.fetchone()
        if row:
            cur.execute("DELETE FROM segments WHERE page_id = %s", (row["id"],))
            cur.execute("DELETE FROM pages WHERE id = %s", (row["id"],))


def delete_chapter_segments(
    provider_id: str,
    manga_title: str,
    chapter_number: float,
    db_url=None,
) -> None:
    """Delete all segments and pages for a chapter (CASCADE removes segments)."""
    with _cursor(db_url) as cur:
        cur.execute(
            "DELETE FROM chapters WHERE provider_id = %s AND manga_title = %s AND chapter_number = %s",
            (provider_id, manga_title, chapter_number),
        )


def get_segments(
    provider_id: str = None,
    manga_title: str = None,
    chapter_number: float = None,
    page_number: int = None,
    db_url=None,
) -> list[dict]:
    """
    Query segments. Pass provider_id, manga_title, chapter_number, page_number to filter.
    Returns list of rows with provider_id, manga_title, chapter_number, page_number,
    segment_index, x1, y1, x2, y2, original_text, translated_text, language_code, created_at.
    """
    with _cursor(db_url) as cur:
        conditions = []
        params = []
        if provider_id is not None:
            conditions.append("c.provider_id = %s")
            params.append(provider_id)
        if manga_title is not None:
            conditions.append("c.manga_title = %s")
            params.append(manga_title)
        if chapter_number is not None:
            conditions.append("c.chapter_number = %s")
            params.append(chapter_number)
        if page_number is not None:
            conditions.append("p.page_number = %s")
            params.append(page_number)
        where = " AND ".join(conditions) if conditions else "1=1"
        cur.execute(
            f"""
            SELECT s.id, c.provider_id, c.manga_title, c.chapter_number, p.page_number, s.segment_index,
                   s.x1, s.y1, s.x2, s.y2, s.original_text, s.translated_text, c.language_code, s.created_at
            FROM segments s
            JOIN pages p ON s.page_id = p.id
            JOIN chapters c ON p.chapter_id = c.id
            WHERE {where}
            ORDER BY c.chapter_number, p.page_number, s.segment_index
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]


def get_chapter_segments(provider_id: str, manga_title: str, chapter_number: float, db_url=None) -> list[dict]:
    """
    Get all segments for a chapter (single query, efficient).
    Returns same row shape as get_segments.
    """
    return get_segments(provider_id=provider_id, manga_title=manga_title, chapter_number=chapter_number, db_url=db_url)


def list_entries(db_url=None) -> list[dict]:
    """List chapters (provider_id, manga_title, chapter_number, last_updated)."""
    with _cursor(db_url) as cur:
        cur.execute("""
            SELECT c.provider_id, c.manga_title, c.chapter_number, c.created_at AS last_updated
            FROM chapters c
            ORDER BY c.created_at DESC
        """)
        return [dict(r) for r in cur.fetchall()]
