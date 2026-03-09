"""
Insert sample data into the database for testing.
Run from backend:  python -m db.test_functions.create_data ***
Requires DATABASE_URL in backend/.env.
"""

import db


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
    # Manga 1: One Piece, local provider
    db.save_page_translation(
        provider_id=db.PROVIDER_LOCAL,
        manga_title="One Piece",
        chapter_number=1.0,
        page_number=1,
        bubbles=_bubbles(("海賊王に俺はなる！", "I'm gonna be King of the Pirates!")),
        language_code="en",
    )
    db.save_page_translation(
        provider_id=db.PROVIDER_LOCAL,
        manga_title="One Piece",
        chapter_number=1.0,
        page_number=2,
        bubbles=_bubbles(("麦わらのルフィ", "Monkey D. Luffy"), ("ゼファ", "Zeff")),
        language_code="en",
    )
    db.save_page_translation(
        provider_id=db.PROVIDER_LOCAL,
        manga_title="One Piece",
        chapter_number=2.0,
        page_number=1,
        bubbles=_bubbles(("冒険の始まり", "The adventure begins")),
        language_code="en",
    )
    # Manga 2: Naruto, mangadex provider
    db.save_page_translation(
        provider_id=db.PROVIDER_MANGADEX,
        manga_title="Naruto",
        chapter_number=1.0,
        page_number=1,
        bubbles=_bubbles(("忍たま乱太郎", "Ninja Academy"), ("うずまきナルト", "Uzumaki Naruto")),
        language_code="en",
    )
    print("Seed data inserted.")
    mangas = db.list_mangas(order_by="updated_at", order_desc=True)
    print("\n=== list_mangas ===")
    for m in mangas:
        print(f"  {m['provider_id']} | {m['manga_title']}")

    print("\n=== list_chapters (first manga) ===")
    if mangas:
        first = mangas[0]
        chapters = db.list_chapters(first["manga_title"], provider_id=first["provider_id"])
        for c in chapters:
            print(f"  {c['provider_id']} | {c['manga_title']} ch.{c['chapter_number']} | {c['updated_at']}")


if __name__ == "__main__":
    seed()
