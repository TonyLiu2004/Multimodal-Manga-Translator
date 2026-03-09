"""
Tests for db module.
Run from backend: python -m unittest db.test_db -v
"""

import os
import unittest

import db


def _bubbles(original: str, translated: str) -> list[dict]:
    return [
        {"bubble_index": 0, "x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 50.0, "original_text": original, "translated_text": translated},
        {"bubble_index": 1, "x1": 10.0, "y1": 60.0, "x2": 100.0, "y2": 90.0, "original_text": original + "2", "translated_text": translated + " 2"},
    ]


class TestDb(unittest.TestCase):
    """Integration tests against a real database."""

    MANGA = "Test Manga DB"
    CHAPTER = 999.0
    PAGE = 1
    LANG = "ja"

    @classmethod
    def setUpClass(cls):
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL not set")
        db.init_db()

    def setUp(self):
        """Clean up test data before each test."""
        db.delete_chapter_segments(db.PROVIDER_LOCAL, self.MANGA, self.CHAPTER)

    def tearDown(self):
        db.delete_chapter_segments(db.PROVIDER_LOCAL, self.MANGA, self.CHAPTER)

    def test_init_db(self):
        """init_db creates tables without error."""
        db.init_db()

    def test_save_and_get_segments(self):
        """save_page_translation stores segments; get_segments returns them."""
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            bubbles=_bubbles("こんにちは", "Hello"),
            language_code=self.LANG,
        )
        rows = db.get_segments(provider_id=db.PROVIDER_LOCAL, manga_title=self.MANGA, chapter_number=self.CHAPTER, page_number=self.PAGE)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["translated_text"], "Hello")
        self.assertEqual(rows[0]["provider_id"], db.PROVIDER_LOCAL)
        self.assertEqual(rows[0]["manga_title"], self.MANGA)
        self.assertEqual(rows[0]["chapter_number"], self.CHAPTER)
        self.assertEqual(rows[0]["page_number"], self.PAGE)

    def test_save_replaces_existing(self):
        """save_page_translation replaces existing segments for same page."""
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            bubbles=_bubbles("first", "First"),
            language_code=self.LANG,
        )
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            bubbles=_bubbles("second", "Second"),
            language_code=self.LANG,
        )
        rows = db.get_segments(provider_id=db.PROVIDER_LOCAL, manga_title=self.MANGA, chapter_number=self.CHAPTER, page_number=self.PAGE)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["translated_text"], "Second")

    def test_get_chapter_segments(self):
        """get_chapter_segments returns all segments for a chapter."""
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=1,
            bubbles=_bubbles("a", "A"),
            language_code=self.LANG,
        )
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=2,
            bubbles=_bubbles("b", "B"),
            language_code=self.LANG,
        )
        rows = db.get_chapter_segments(db.PROVIDER_LOCAL, self.MANGA, self.CHAPTER)
        self.assertEqual(len(rows), 4)  # 2 bubbles per page, 2 pages

    def test_list_mangas(self):
        """list_mangas returns mangas; order_by and order_desc work."""
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            bubbles=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        entries = db.list_mangas()
        found = [e for e in entries if e["manga_title"] == self.MANGA]
        self.assertGreater(len(found), 0)
        self.assertIn("provider_id", found[0])
        self.assertIn("manga_title", found[0])
        self.assertIn("created_at", found[0])
        self.assertIn("updated_at", found[0])

        entries_asc = db.list_mangas(order_by="created_at", order_desc=False)
        self.assertIsInstance(entries_asc, list)

    def test_list_mangas_pagination(self):
        """list_mangas respects limit and offset."""
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            bubbles=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        all_entries = db.list_mangas()
        limited = db.list_mangas(limit=1)
        self.assertEqual(len(limited), 1)
        offset_entries = db.list_mangas(limit=1, offset=0)
        self.assertEqual(len(offset_entries), 1)

    def test_list_chapters(self):
        """list_chapters returns chapters for a manga; optional provider_id filter."""
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            bubbles=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        chapters = db.list_chapters(self.MANGA)
        found = [c for c in chapters if c["chapter_number"] == self.CHAPTER]
        self.assertGreater(len(found), 0)
        self.assertIn("manga_title", found[0])
        self.assertIn("provider_id", found[0])
        self.assertIn("id", found[0])
        self.assertIn("chapter_number", found[0])
        self.assertIn("created_at", found[0])
        self.assertIn("updated_at", found[0])
        self.assertEqual(found[0]["manga_title"], self.MANGA)
        self.assertEqual(found[0]["provider_id"], db.PROVIDER_LOCAL)

        chapters_with_provider = db.list_chapters(self.MANGA, provider_id=db.PROVIDER_LOCAL)
        self.assertGreater(len(chapters_with_provider), 0)

    def test_list_chapters_pagination(self):
        """list_chapters respects limit and offset."""
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            bubbles=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        all_chapters = db.list_chapters(self.MANGA, provider_id=db.PROVIDER_LOCAL)
        limited = db.list_chapters(self.MANGA, provider_id=db.PROVIDER_LOCAL, limit=1)
        self.assertEqual(len(limited), 1)
        offset_chapters = db.list_chapters(self.MANGA, provider_id=db.PROVIDER_LOCAL, limit=1, offset=0)
        self.assertEqual(len(offset_chapters), 1)

    def test_get_segments_pagination(self):
        """get_segments respects limit and offset."""
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            bubbles=_bubbles("a", "A"),
            language_code=self.LANG,
        )
        all_rows = db.get_segments(provider_id=db.PROVIDER_LOCAL, manga_title=self.MANGA, chapter_number=self.CHAPTER)
        self.assertEqual(len(all_rows), 2)
        limited = db.get_segments(provider_id=db.PROVIDER_LOCAL, manga_title=self.MANGA, chapter_number=self.CHAPTER, limit=1)
        self.assertEqual(len(limited), 1)
        offset_rows = db.get_segments(provider_id=db.PROVIDER_LOCAL, manga_title=self.MANGA, chapter_number=self.CHAPTER, limit=1, offset=1)
        self.assertEqual(len(offset_rows), 1)
        self.assertEqual(offset_rows[0]["translated_text"], "A 2")

    def test_delete_page_segments(self):
        """delete_page_segments removes page and its segments."""
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            bubbles=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        db.delete_page_segments(db.PROVIDER_LOCAL, self.MANGA, self.CHAPTER, self.PAGE)
        rows = db.get_segments(provider_id=db.PROVIDER_LOCAL, manga_title=self.MANGA, chapter_number=self.CHAPTER, page_number=self.PAGE)
        self.assertEqual(len(rows), 0)

    def test_delete_chapter_segments(self):
        """delete_chapter_segments removes chapter and all its pages/segments."""
        db.save_page_translation(
            provider_id=db.PROVIDER_LOCAL,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            bubbles=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        db.delete_chapter_segments(db.PROVIDER_LOCAL, self.MANGA, self.CHAPTER)
        rows = db.get_chapter_segments(db.PROVIDER_LOCAL, self.MANGA, self.CHAPTER)
        self.assertEqual(len(rows), 0)

    def test_provider_id_validation_save(self):
        """save_page_translation raises ValueError for invalid provider_id."""
        with self.assertRaises(ValueError) as ctx:
            db.save_page_translation(
                provider_id="invalid_provider",
                manga_title=self.MANGA,
                chapter_number=self.CHAPTER,
                page_number=self.PAGE,
                bubbles=_bubbles("x", "X"),
                language_code=self.LANG,
            )
        self.assertIn("provider_id must be one of", str(ctx.exception))

    def test_provider_id_validation_get_chapter_segments(self):
        """get_chapter_segments raises ValueError for invalid provider_id."""
        with self.assertRaises(ValueError):
            db.get_chapter_segments("invalid_provider", self.MANGA, self.CHAPTER)

    def test_provider_id_validation_delete_page(self):
        """delete_page_segments raises ValueError for invalid provider_id."""
        with self.assertRaises(ValueError):
            db.delete_page_segments("invalid_provider", self.MANGA, self.CHAPTER, self.PAGE)

    def test_provider_id_validation_delete_chapter(self):
        """delete_chapter_segments raises ValueError for invalid provider_id."""
        with self.assertRaises(ValueError):
            db.delete_chapter_segments("invalid_provider", self.MANGA, self.CHAPTER)

    def test_provider_id_validation_get_segments(self):
        """get_segments raises ValueError when provider_id is invalid."""
        with self.assertRaises(ValueError):
            db.get_segments(provider_id="invalid_provider")

    def test_provider_mangadex(self):
        """PROVIDER_MANGADEX is accepted."""
        db.save_page_translation(
            provider_id=db.PROVIDER_MANGADEX,
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            bubbles=_bubbles("m", "M"),
            language_code=self.LANG,
        )
        rows = db.get_segments(provider_id=db.PROVIDER_MANGADEX, manga_title=self.MANGA, chapter_number=self.CHAPTER)
        self.assertEqual(len(rows), 2)
        db.delete_chapter_segments(db.PROVIDER_MANGADEX, self.MANGA, self.CHAPTER)


class TestDbNoUrl(unittest.TestCase):
    """Tests that work without DATABASE_URL."""

    def test_get_engine_raises_without_url(self):
        """get_engine raises ValueError when DATABASE_URL is not set."""
        orig = os.environ.pop("DATABASE_URL", None)
        try:
            with self.assertRaises(ValueError) as ctx:
                db.get_engine()
            self.assertIn("No database URL", str(ctx.exception))
        finally:
            if orig is not None:
                os.environ["DATABASE_URL"] = orig


if __name__ == "__main__":
    unittest.main()
