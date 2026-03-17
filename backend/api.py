"""
Read-only API for the frontend. Wraps db list_entries, get_segments, get_chapter_segments.
Run from backend:  uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import proxy
from services import mangadex_service
import httpx
import db
from sqlmodel import Session, text
from db.models import Manga
from db.schemas import ChapterListOut, SegmentListOut

app = FastAPI(
    title="Manga Translator API",
    description="Read endpoints for chapters and segments",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Currently allow all origins, should be restricted to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """API root - confirms the API is running."""
    return {
        "message": "Manga Translator API is working",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }

@app.get("/health/db")
def health_db():
    engine = db.get_engine()
    try:
        with Session(engine) as session:
            session.exec(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/mangas", response_model=list[Manga])
def list_mangas(
    order_by: str = Query("created_at", description="provider_id | manga_title | created_at | updated_at"),
    order_desc: bool = Query(True, description="Sort descending"),
    limit: int | None = Query(None, ge=1, le=500, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """List mangas (provider_id, manga_title, created_at, updated_at). Supports pagination."""
    return db.list_mangas(order_by=order_by, order_desc=order_desc, limit=limit, offset=offset)

@app.get("/chapters", response_model=list[ChapterListOut])
def list_chapters(
    manga_title: str = Query(...),
    provider_id: str | None = Query(None, description="e.g. local, mangadex"),
    limit: int | None = Query(None, ge=1, le=500, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """List chapters (id, chapter_number, created_at, updated_at). Filters by manga_title; optionally by provider_id."""
    return db.list_chapters(manga_title, provider_id, limit=limit, offset=offset)


@app.get("/segments", response_model=list[SegmentListOut])
def get_segments(
    provider_id: str | None = Query(None, description="e.g. local, mangadex"),
    manga_title: str | None = Query(None),
    chapter_number: float | None = Query(None),
    page_number: int | None = Query(None),
    limit: int | None = Query(None, ge=1, le=1000, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """Get segments with optional filters. Supports pagination."""
    return db.get_segments(
        provider_id=provider_id,
        manga_title=manga_title,
        chapter_number=chapter_number,
        page_number=page_number,
        limit=limit,
        offset=offset,
    )


@app.get("/chapters/segments", response_model=list[SegmentListOut])
def get_chapter_segments(
    provider_id: str = Query(..., description="e.g. local, mangadex"),
    manga_title: str = Query(...),
    chapter_number: float = Query(...),
    limit: int | None = Query(None, ge=1, le=1000, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """Get all segments for one chapter. Supports pagination."""
    return db.get_chapter_segments(provider_id, manga_title, chapter_number, limit=limit, offset=offset)


# to make sure api is running and responding
@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.exception_handler(ValueError)
def value_error_handler(request, exc):
    """Return 400 for invalid provider_id etc."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})

###########
###########
###########



@app.get("/api/manga/chapter/{chapter_id}/page/{page_index}")
async def proxy_manga_page(chapter_id: str, page_index: int):
    urls = mangadex_service.get_chapter_panel_urls(chapter_id)
    if not urls or page_index >= len(urls):
        return {"error": "Page not found"}, 404

    return await proxy.get_manga_page_stream(urls[page_index])

@app.get("/api/manga/search")
async def get_popular_manga(
    title: str = "",
    limit: int = 15,
    offset: int = 0,
    order_by: str = "followedCount",
    order_direction: str = "desc",
    cover_art: bool = True
):
    results = await mangadex_service.search_manga(
        title=title,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_direction=order_direction,
        cover_art=cover_art
    )
    return results


@app.get("/api/manga/{manga_id}/chapters")
async def get_chapters(
    manga_id: str,
    limit: int = 100,
    translatedLanguage: list[str] = Query(['en']),
    offset: int = 0,
    order_by: str = "chapter",
    order_direction: str = "desc",
    content_rating: list[str] = Query(["safe", "suggestive"]), #, "erotica", "pornographic"], #oh hell naw
    includeEmptyPages: int = 0
):
    results = await mangadex_service.get_manga_chapters(
        manga_id=manga_id,
        limit=limit,
        languages=translatedLanguage,
        offset=offset,
        order_by=order_by,
        order_direction=order_direction,
        content_rating=content_rating,
        include_empty=includeEmptyPages
    )
    return results