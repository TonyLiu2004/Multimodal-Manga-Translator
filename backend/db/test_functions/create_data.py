"""
Insert sample data into the database for testing.
Run from backend:  python -m db.test_functions.create_data
Requires DATABASE_URL in backend/.env.

Uses umbrella Manga + Chapters. Optional external_manga_id (MangaDex series id)
on save_panels_translation stores MangaDex series id.
"""

import db

# Demo external ids (MangaDex uses UUID strings; local can use legacy-* from DB or omit)
NARUTO_MANGADEX_ID = "931b3112-7b75-43e4-a13f-908f0a251ae2"  # example placeholder UUID


def _slug(title: str) -> str:
    s = title.strip().lower().replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch == "_")


def _bubbles(
    manga_title: str,
    chapter_number: float,
    page_number: int,
    *pairs: tuple[str, str],
) -> list[dict]:
    """Create fake panels from (original, translated) pairs."""
    slug = _slug(manga_title)
    result = []
    for i, (orig, trans) in enumerate(pairs):
        result.append({
            "bubble_index": i,
            # Page coordinate space (example values)
            "width": 1080,
            "height": 1522,
            # Globally unique demo URLs (matches global uniqueness on panel_url).
            "panel_url": (
                "https://example.invalid/panels/"
                f"{slug}/ch{chapter_number:g}/p{page_number}/b{i}.jpg"
            ),
            "x1": 120.0 + i * 18,
            "y1": 140.0 + i * 55,
            "x2": 420.0 + i * 18,
            "y2": 260.0 + i * 55,
            "original_text": orig,
            "translated_text": trans,
        })
    return result


def seed():
    db.init_db()

    # Umbrella manga 1: One Piece — MangaDex provider (example seed)
    db.save_panels_translation(
        manga_title="One Piece",
        chapter_number=1.0,
        page_number=1,
        panels=_bubbles(
            "One Piece",
            1.0,
            1,
            ("海賊王に俺はなる！", "I'm gonna be King of the Pirates!"),
        ),
        language_code="en",
    )
    db.save_panels_translation(
        manga_title="One Piece",
        chapter_number=1.0,
        page_number=2,
        panels=_bubbles(
            "One Piece",
            1.0,
            2,
            ("麦わらのルフィ", "Monkey D. Luffy"),
            ("ゼファ", "Zeff"),
        ),
        language_code="en",
    )
    db.save_panels_translation(
        manga_title="One Piece",
        chapter_number=2.0,
        page_number=1,
        panels=_bubbles(
            "One Piece",
            2.0,
            1,
            ("冒険の始まり", "The adventure begins"),
        ),
        language_code="en",
    )

    # Umbrella manga 2: Naruto — MangaDex series id stored on Manga via external_manga_id
    db.save_panels_translation(
        manga_title="Naruto",
        chapter_number=1.0,
        page_number=1,
        panels=_bubbles(
            "Naruto",
            1.0,
            1,
            ("忍たま乱太郎", "Ninja Academy"),
            ("うずまきナルト", "Uzumaki Naruto"),
        ),
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
    print("\n=== list_chapters('One Piece') ===")
    for c in db.list_chapters("One Piece"):
        print(f"  ch.id={c.id}  ch={c.chapter_number}  updated={c.updated_at}")

    print("\n=== list_chapters('Naruto') ===")
    for c in db.list_chapters("Naruto"):
        print(f"  ch.id={c.id}  ch={c.chapter_number}  updated={c.updated_at}")

    print("\n=== MangaDex external_manga_id mapping ===")
    print("  Stored on manga.mangadex_manga_id")

    # Sample panel query
    panels = db.get_chapter_panels("One Piece", 1.0)
    print("\n=== get_chapter_panels('One Piece', 1.0) ===")
    print(f"  panel count: {len(panels)}")
    if panels:
        p0 = panels[0]
        print(f"  first: page={p0.page_number}  translated={p0.translated_text!r}")


if __name__ == "__main__":
    seed()
