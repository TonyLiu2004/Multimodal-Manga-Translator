"""To use db functions,
    import like this: from db
    import init_db,
    save_page_translation,
    get_segments,
    get_chapter_segments,
    list_entries,
    delete_page_segments,
    delete_chapter_segments,
    get_connection,
    get_engine
"""
from .const import PROVIDER_IDS, PROVIDER_LOCAL, PROVIDER_MANGADEX
from .db import (
    init_db,
    save_page_translation,
    get_segments,
    get_chapter_segments,
    list_mangas,
    list_chapters,
    delete_page_segments,
    delete_chapter_segments,
    delete_all_manga,
    get_connection,
    get_engine,
)
