"""
Insert sample data into the database for testing.
Run from backend:  python -m db.test_functions.create_data
Requires DATABASE_URL in backend/.env.

Uses umbrella Manga + MangaSource + Chapters (per provider_id). Optional external_manga_id
on save_page_translation links a provider catalog id (e.g. MangaDex UUID).
"""

import db
from db.const import PROVIDER_LOCAL, PROVIDER_MANGADEX

# Demo external ids (MangaDex uses UUID strings; local can use legacy-* from DB or omit)
NARUTO_MANGADEX_ID = "931b3112-7b75-43e4-a13f-908f0a251ae2"  # example placeholder UUID


def _bubbles(*pairs: tuple[str, str]) -> list[dict]:
    """Create fake text segments from (original, translated) pairs."""
    result = []
    for i, (orig, trans) in enumerate(pairs):
        result.append({
            "bubble_index": i,
            "x1": 10.0 + i * 5,
            "y1": 20.0 + i * 40,
            "x2": 100.0,
            "y2": 50.0 + i * 40,
            "original_text": orig,
            "translated_text": trans,
        })
    return result


def seed():
    db.init_db()

    # Umbrella manga 1: One Piece — local files only (no API id → omit external_manga_id, DB uses legacy-*)
    db.save_page_translation(
        provider_id=PROVIDER_LOCAL,
        manga_title="One Piece",
        chapter_number=1.0,
        page_number=1,
        bubbles=_bubbles(("海賊王に俺はなる！", "I'm gonna be King of the Pirates!")),
        language_code="en",
    )
    db.save_page_translation(
        provider_id=PROVIDER_LOCAL,
        manga_title="One Piece",
        chapter_number=1.0,
        page_number=2,
        bubbles=_bubbles(("麦わらのルフィ", "Monkey D. Luffy"), ("ゼファ", "Zeff")),
        language_code="en",
    )
    db.save_page_translation(
        provider_id=PROVIDER_LOCAL,
        manga_title="One Piece",
        chapter_number=2.0,
        page_number=1,
        bubbles=_bubbles(("冒険の始まり", "The adventure begins")),
        language_code="en",
    )

    # Umbrella manga 2: Naruto — MangaDex catalog id stored in manga_source via external_manga_id
    db.save_page_translation(
        provider_id=PROVIDER_MANGADEX,
        manga_title="Naruto",
        chapter_number=1.0,
        page_number=1,
        bubbles=_bubbles(("忍たま乱太郎", "Ninja Academy"), ("うずまきナルト", "Uzumaki Naruto")),
        language_code="en",
        external_manga_id=NARUTO_MANGADEX_ID,
    )

    print("Seed data inserted.\n")

    # --- Queries: umbrella manga (no provider on Manga row) ---
    mangas = db.list_mangas(order_by="updated_at", order_desc=True)
    print("=== list_mangas (umbrella) ===")
    for m in mangas:
        print(f"  id={m.id}  title={m.manga_title!r}  updated_at={m.updated_at}")

    # --- Chapters are scoped by provider: same title can exist per source ---
    print("\n=== list_chapters('One Piece', provider_id=local) ===")
    for c in db.list_chapters("One Piece", provider_id=PROVIDER_LOCAL):
        print(f"  ch.id={c.id}  provider={c.provider_id!r}  ch={c.chapter_number}  updated={c.updated_at}")

    print("\n=== list_chapters('Naruto', provider_id=mangadex) ===")
    for c in db.list_chapters("Naruto", provider_id=PROVIDER_MANGADEX):
        print(f"  ch.id={c.id}  provider={c.provider_id!r}  ch={c.chapter_number}  updated={c.updated_at}")

    # Resolve umbrella id from provider + external catalog id
    mid = db.resolve_manga_id_by_source(PROVIDER_MANGADEX, NARUTO_MANGADEX_ID)
    print("\n=== resolve_manga_id_by_source(mangadex, external_manga_id) ===")
    print(f"  umbrella manga_id for Naruto sample UUID: {mid}")

    # Sample segment query (provider on chapter)
    segs = db.get_chapter_segments(PROVIDER_LOCAL, "One Piece", 1.0)
    print("\n=== get_chapter_segments(local, 'One Piece', 1.0) ===")
    print(f"  segment count: {len(segs)}")
    if segs:
        s0 = segs[0]
        print(f"  first: provider_id={s0.provider_id!r}  page={s0.page_number}  translated={s0.translated_text!r}")


if __name__ == "__main__":
    seed()
