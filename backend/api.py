"""
Read-only API for the frontend. Wraps db list_entries, get_segments, get_chapter_segments.
Run from backend:  uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import proxy
from services import mangadex_service

import db

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


@app.get("/mangas")
def list_mangas(
    order_by: str = Query("created_at", description="provider_id | manga_title | created_at | updated_at"),
    order_desc: bool = Query(True, description="Sort descending"),
    limit: int | None = Query(None, ge=1, le=500, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """List mangas (provider_id, manga_title, created_at, updated_at). Supports pagination."""
    entries = db.list_mangas(order_by=order_by, order_desc=order_desc, limit=limit, offset=offset)
    return _serialize(entries)

@app.get("/chapters")
def list_chapters(
    manga_title: str = Query(...),
    provider_id: str | None = Query(None, description="e.g. local, mangadex"),
    limit: int | None = Query(None, ge=1, le=500, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """List chapters (id, chapter_number, created_at, updated_at). Filters by manga_title; optionally by provider_id."""
    chapters = db.list_chapters(manga_title, provider_id, limit=limit, offset=offset)
    return _serialize(chapters)


@app.get("/segments")
def get_segments(
    provider_id: str | None = Query(None, description="e.g. local, mangadex"),
    manga_title: str | None = Query(None),
    chapter_number: float | None = Query(None),
    page_number: int | None = Query(None),
    limit: int | None = Query(None, ge=1, le=1000, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """Get segments with optional filters. Supports pagination."""
    segments = db.get_segments(
        provider_id=provider_id,
        manga_title=manga_title,
        chapter_number=chapter_number,
        page_number=page_number,
        limit=limit,
        offset=offset,
    )
    return _serialize(segments)


@app.get("/chapters/segments")
def get_chapter_segments(
    provider_id: str = Query(..., description="e.g. local, mangadex"),
    manga_title: str = Query(...),
    chapter_number: float = Query(...),
    limit: int | None = Query(None, ge=1, le=1000, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """Get all segments for one chapter. Supports pagination."""
    segments = db.get_chapter_segments(provider_id, manga_title, chapter_number, limit=limit, offset=offset)
    return _serialize(segments)


def _serialize(entries: list[dict]) -> list[dict]:
    """Convert datetime fields to ISO strings for JSON."""
    result = []
    for e in entries:
        row = dict(e)
        for key in ("last_updated", "created_at", "updated_at"):
            if key in row and row[key] is not None:
                row[key] = row[key].isoformat()
        result.append(row)
    return result


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

@app.get("/proxy/chapter/{chapter_id}/page/{page_index}")
async def proxy_manga_page(chapter_id: str, page_index: int):
    urls = mangadex_service.get_chapter_panel_urls(chapter_id)
    if not urls or page_index >= len(urls):
        return {"error": "Page not found"}, 404
        
    return await proxy.get_manga_page_stream(urls[page_index])