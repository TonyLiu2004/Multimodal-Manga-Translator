"""
Read-only API for the frontend. Wraps db list_entries, get_panels, get_chapter_panels.
Run from backend:  uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import APIRouter, FastAPI, HTTPException, Query, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import proxy
from services import mangadex_service
from services.image_processor import ImageProcessor
from services.bubble_detector_kiuyha_service import Bubble_Detector_Kiuyha_Service
from services.translate_qwen_service import Translate_Qwen_Service
from manga_ocr import MangaOcr
import httpx
import urllib.parse
import db
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, text
from db.models import Manga
from auth_supabase import CurrentAuthContext, CurrentUserId
from db.schemas import (
    ChapterListOut,
    ReadingListAddIn,
    ReadingListCollectionCreateIn,
    ReadingListCollectionOut,
    ReadingListCollectionRenameIn,
    ReadingListItemOut,
    ReadingListProgressPatchIn,
    PanelListOut,
    UserDisplayNamePatchIn,
)
from fastapi import Query
from typing import Optional

# app = FastAPI(
#     title="Manga Translator API",
#     description="Read endpoints for chapters and segments",
#     version="1.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # Currently allow all origins, should be restricted to specific origins in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

router = APIRouter()
bubble_detector = Bubble_Detector_Kiuyha_Service()
ocr_model = MangaOcr()
translate_model = Translate_Qwen_Service()
processor = ImageProcessor(
    bubble_detector=bubble_detector,
    ocr_model=ocr_model,
    translate_model=translate_model
)

BACKEND_URL = "https://tonyliu404-manglify-backend.hf.space"

@router.get("/")
def root():
    """API root - confirms the API is running."""
    return {
        "message": "Manga Translator API is working",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }

@router.get("/health/db")
def health_db():
    engine = db.get_engine()
    try:
        with Session(engine) as session:
            session.exec(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/mangas", response_model=list[Manga])
def list_mangas(
    order_by: str = Query("created_at", description="manga_title | created_at | updated_at"),
    order_desc: bool = Query(True, description="Sort descending"),
    limit: int | None = Query(None, ge=1, le=500, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """List mangas (manga_title, created_at, updated_at). Supports pagination."""
    return db.list_mangas(order_by=order_by, order_desc=order_desc, limit=limit, offset=offset)

@router.get("/chapters", response_model=list[ChapterListOut])
def list_chapters(
    manga_title: str = Query(...),
    limit: int | None = Query(None, ge=1, le=500, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """List chapters (id, chapter_number, created_at, updated_at). Filters by manga_title."""
    return db.list_chapters(manga_title, limit=limit, offset=offset)


@router.get("/panels", response_model=list[PanelListOut])
def get_panels(
    manga_title: str | None = Query(None),
    chapter_number: float | None = Query(None),
    page_number: int | None = Query(None),
    limit: int | None = Query(None, ge=1, le=1000, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """Get panels with optional filters. Supports pagination."""
    return db.get_panels(
        manga_title=manga_title,
        chapter_number=chapter_number,
        page_number=page_number,
        limit=limit,
        offset=offset,
    )


@router.get("/chapters/panels", response_model=list[PanelListOut])
def get_chapter_panels(
    manga_title: str = Query(...),
    chapter_number: float = Query(...),
    limit: int | None = Query(None, ge=1, le=1000, description="Max results (default: all)"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """Get all panels for one chapter. Supports pagination."""
    return db.get_chapter_panels(manga_title, chapter_number, limit=limit, offset=offset)




# to make sure api is running and responding
@router.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@router.get("/reading-lists", response_model=list[ReadingListCollectionOut])
def get_reading_lists(user_id: CurrentUserId):
    """List named reading lists for the signed-in user (with manga counts)."""
    return db.list_reading_list_collections_with_counts(user_id)


@router.post("/reading-lists", response_model=ReadingListCollectionOut)
def post_reading_list_collection(
    ctx: CurrentAuthContext, body: ReadingListCollectionCreateIn
):
    """Create a new named reading list."""
    try:
        col = db.create_reading_list_collection(
            ctx.user_id, body.name, user_email=ctx.email
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except IntegrityError as e:
        raise HTTPException(
            status_code=400,
            detail="Could not create reading list (database constraint).",
        ) from e
    if col.id is None:
        raise HTTPException(status_code=500, detail="Reading list id missing after save")
    return ReadingListCollectionOut(
        id=col.id,
        name=col.name,
        manga_count=0,
        created_at=col.created_at,
        updated_at=col.updated_at,
        latest_external_manga_id=None,
    )


@router.patch("/reading-lists/{reading_list_id}", response_model=ReadingListCollectionOut)
def patch_reading_list_collection(
    user_id: CurrentUserId,
    reading_list_id: int,
    body: ReadingListCollectionRenameIn,
):
    """Rename a reading list."""
    try:
        col = db.update_reading_list_collection(user_id, reading_list_id, body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if col is None:
        raise HTTPException(status_code=404, detail="Reading list not found")
    counts = db.list_reading_list_collections_with_counts(user_id)
    for c in counts:
        if c.id == col.id:
            return c
    return ReadingListCollectionOut(
        id=col.id,
        name=col.name,
        manga_count=0,
        created_at=col.created_at,
        updated_at=col.updated_at,
        latest_external_manga_id=None,
    )


@router.delete("/reading-lists/{reading_list_id}")
def delete_reading_list_collection(user_id: CurrentUserId, reading_list_id: int):
    """Delete a named list and all manga entries in it."""
    ok = db.delete_reading_list_collection(user_id, reading_list_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Reading list not found")
    return {"ok": True}


@router.get("/reading-lists/{reading_list_id}/items", response_model=list[ReadingListItemOut])
def get_reading_list_items(
    user_id: CurrentUserId,
    reading_list_id: int,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Manga rows for one named list."""
    if db.get_reading_list_collection(user_id, reading_list_id) is None:
        raise HTTPException(status_code=404, detail="Reading list not found")
    return db.list_reading_list_items_with_manga(
        user_id, reading_list_id, limit=limit, offset=offset
    )


@router.post("/reading-lists/{reading_list_id}/items", response_model=ReadingListItemOut)
def post_reading_list_item(
    user_id: CurrentUserId, reading_list_id: int, body: ReadingListAddIn
):
    """Add or update a manga on a named list (by MangaDex series id)."""
    if db.get_reading_list_collection(user_id, reading_list_id) is None:
        raise HTTPException(status_code=404, detail="Reading list not found")
    try:
        row = db.add_reading_list_item_by_source(
            user_id,
            reading_list_id,
            body.external_manga_id,
            body.manga_title,
            last_chapter_number=body.last_chapter_number,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    items = db.list_reading_list_items_with_manga(user_id, reading_list_id, limit=500)
    for it in items:
        if it.id == row.id or it.manga_id == row.manga_id:
            return it
    raise HTTPException(status_code=500, detail="Item not found after save")


@router.delete("/reading-lists/{reading_list_id}/items/{manga_id}")
def delete_reading_list_item(
    user_id: CurrentUserId, reading_list_id: int, manga_id: int
):
    """Remove one manga from a named list."""
    ok = db.remove_reading_list_item_from_list(user_id, reading_list_id, manga_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Not on this reading list")
    return {"ok": True}


@router.patch(
    "/reading-lists/{reading_list_id}/items/{manga_id}",
    response_model=ReadingListItemOut,
)
def patch_reading_list_item_progress(
    user_id: CurrentUserId,
    reading_list_id: int,
    manga_id: int,
    body: ReadingListProgressPatchIn,
):
    """Update last-read chapter for one manga row (stored value only increases)."""
    ok = db.update_reading_list_item_last_read(
        user_id,
        reading_list_id,
        manga_id,
        body.last_chapter_number,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Not on this reading list")
    items = db.list_reading_list_items_with_manga(user_id, reading_list_id, limit=500)
    for it in items:
        if it.manga_id == manga_id:
            return it
    raise HTTPException(status_code=404, detail="Item not found after update")


@router.patch("/users/me")
def patch_me_display_name(ctx: CurrentAuthContext, body: UserDisplayNamePatchIn):
    """Update display_name on public.users for the signed-in user."""
    try:
        db.update_app_user_display_name(
            ctx.user_id,
            body.display_name,
            email=ctx.email,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return Response(status_code=204)


###########
###########
###########

# @router.get("/api/manga/chapter/{chapter_id}/page/{page_index}")
# async def proxy_manga_page(chapter_id: str, page_index: int):
#     urls = mangadex_service.get_chapter_panel_urls(chapter_id)
#     if not urls or page_index >= len(urls):
#         return {"error": "Page not found"}, 404

#     return await proxy.get_manga_page_stream(urls[page_index])


@router.get("/api/proxy/image")
async def proxy_image(
    background_tasks: BackgroundTasks,
    target_url: str = Query(...), 
    chapter_id: Optional[str] = Query(None),
    page: Optional[int] = Query(None)
):
    image_bytes = await proxy.get_manga_page_stream(target_url)
    
    if chapter_id is not None and page is not None:
        background_tasks.add_task(
            processor.process_from_memory,
            image_bytes,
            target_url,
            chapter_id,
            page
        )
    
    return Response(content=image_bytes, media_type="image/jpeg")

@router.get("/api/manga/chapter/{chapter_id}/pages")
def get_chapter_page_urls(chapter_id: str):
    """MangaDex at-home CDN URLs for every page in a chapter (for client readers)."""
    urls = mangadex_service.get_chapter_panel_urls(chapter_id)
    if not urls:
        raise HTTPException(status_code=404, detail="No pages for this chapter")
    
    proxied_urls = [
        f"{BACKEND_URL}/api/proxy/image?target_url={urllib.parse.quote_plus(url)}&chapter_id={chapter_id}&page={i}" 
        for i, url in enumerate(urls)
    ]

    print(proxied_urls)
    return {"urls": proxied_urls}

@router.get("/api/manga/cover_art")
async def proxy_manga_cover_art(manga_id: str, file_name: str, size: int = 256):
    url = f"https://uploads.mangadex.org/covers/{manga_id}/{file_name}.{size}.jpg"
    print(url)
    return await proxy.get_manga_page_stream(url)


@router.get("/api/manga/{manga_id}/cover")
async def get_manga_cover_json(manga_id: str):
    """MangaDex manga UUID → 256px cover URL for list UIs (no DB storage)."""
    cover_url = await mangadex_service.get_manga_cover_url_256(manga_id)
    return {"cover_url": cover_url}


@router.get("/api/manga/{manga_id}/info")
async def get_manga_info_json(manga_id: str):
    """MangaDex title + synopsis for reading-list rows (no DB storage)."""
    info = await mangadex_service.get_manga_public_info(manga_id)
    return info


@router.get("/api/manga/search")
async def get_popular_manga(
    title: str = "",
    limit: int = 15,
    offset: int = 0,
    order_by: str = "followedCount",
    order_direction: str = "desc",
    cover_art: bool = True,
    includedTags: list[str] = Query(default=[]),
    excludedTags: list[str] = Query(default=[]),
    status: str = "",
):
    results = await mangadex_service.search_manga(
        title=title,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_direction=order_direction,
        cover_art=cover_art,
        included_tags=includedTags,
        excluded_tags=excludedTags,
        status=status or None,
    )
    return results


@router.get("/api/manga/{manga_id}/chapters")
async def get_chapters(
    manga_id: str,
    limit: int = 100,
    translatedLanguage: list[str] = Query(None),
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


# Standalone ASGI app for `uvicorn api:app` (no ML / translation stack from main.py).
app = FastAPI(
    title="Manga Translator API",
    description="Read endpoints for chapters and segments",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(ValueError)
def _value_error_handler(request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})