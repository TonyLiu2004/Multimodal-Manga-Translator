"""
Test that database add, update, and delete work.
Run from backend folder:  python test_db.py
Uses DATABASE_URL from backend/.env (or set it).
"""

import db

TEST_PROVIDER = "test_provider"
TEST_MANGA = "Test Manga"
TEST_CHAPTER = 99.0
TEST_PAGE = 1
TEST_LANG = "ja"


def fake_bubbles(original: str, translated: str):
    """Two fake segments for testing."""
    return [
        {"bubble_index": 0, "x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 50.0, "original_text": original, "translated_text": translated},
        {"bubble_index": 1, "x1": 10.0, "y1": 60.0, "x2": 100.0, "y2": 90.0, "original_text": original + "2", "translated_text": translated + " 2"},
    ]


def main():
    print("1. Init DB (create table if needed)...")
    db.init_db()
    print("   OK\n")

    print("2. ADD: save one page of segments...")
    db.save_page_translation(
        provider_id=TEST_PROVIDER,
        manga_title=TEST_MANGA,
        chapter_number=TEST_CHAPTER,
        page_number=TEST_PAGE,
        bubbles=fake_bubbles("こんにちは", "Hello"),
        language_code=TEST_LANG,
    )
    rows = db.get_segments(provider_id=TEST_PROVIDER, manga_title=TEST_MANGA, chapter_number=TEST_CHAPTER, page_number=TEST_PAGE)
    print(f"   Rows after add: {len(rows)}")
    assert len(rows) == 2, "Expected 2 segments"
    print(f"   First segment translated_text: {rows[0]['translated_text']}")
    print("   ADD OK\n")

    print("3. LIST: list_entries...")
    entries = db.list_entries()
    found = [e for e in entries if e["manga_title"] == TEST_MANGA and e["chapter_number"] == TEST_CHAPTER]
    print(f"   Found test entry: {len(found) > 0}")
    print("   LIST OK\n")

    print("4. UPDATE: save same page with different text (replace)...")
    db.save_page_translation(
        provider_id=TEST_PROVIDER,
        manga_title=TEST_MANGA,
        chapter_number=TEST_CHAPTER,
        page_number=TEST_PAGE,
        bubbles=fake_bubbles("さようなら", "Goodbye"),
        language_code=TEST_LANG,
    )
    rows = db.get_segments(provider_id=TEST_PROVIDER, manga_title=TEST_MANGA, chapter_number=TEST_CHAPTER, page_number=TEST_PAGE)
    print(f"   Rows after update: {len(rows)}")
    assert len(rows) == 2
    print(f"   First segment translated_text: {rows[0]['translated_text']}")
    assert rows[0]["translated_text"] == "Goodbye", "Update should change text"
    print("   UPDATE OK\n")

    print("5. DELETE: delete this page's segments...")
    db.delete_page_segments(TEST_PROVIDER, TEST_MANGA, TEST_CHAPTER, TEST_PAGE)
    rows = db.get_segments(provider_id=TEST_PROVIDER, manga_title=TEST_MANGA, chapter_number=TEST_CHAPTER, page_number=TEST_PAGE)
    print(f"   Rows after delete: {len(rows)}")
    assert len(rows) == 0, "Expected 0 segments after delete"
    print("   DELETE OK\n")

    print("All tests passed: add, list, update, delete work.")


if __name__ == "__main__":
    main()
