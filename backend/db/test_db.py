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
        db.delete_chapter_panels(self.MANGA, self.CHAPTER)

    def tearDown(self):
        db.delete_chapter_panels(self.MANGA, self.CHAPTER)

    def test_init_db(self):
        """init_db creates tables without error."""
        db.init_db()

    def test_save_and_get_panels(self):
        """save_panels_translation stores panels; get_panels returns them."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("こんにちは", "Hello"),
            language_code=self.LANG,
        )
        rows = db.get_panels(manga_title=self.MANGA, chapter_number=self.CHAPTER, page_number=self.PAGE)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].translated_text, "Hello")
        self.assertEqual(rows[0].manga_title, self.MANGA)
        self.assertEqual(rows[0].chapter_number, self.CHAPTER)
        self.assertEqual(rows[0].page_number, self.PAGE)

    def test_save_replaces_existing(self):
        """save_panels_translation replaces existing panels for same page."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("first", "First"),
            language_code=self.LANG,
        )
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("second", "Second"),
            language_code=self.LANG,
        )
        rows = db.get_panels(manga_title=self.MANGA, chapter_number=self.CHAPTER, page_number=self.PAGE)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].translated_text, "Second")

    def test_get_chapter_panels(self):
        """get_chapter_panels returns all panels for a chapter."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=1,
            panels=_bubbles("a", "A"),
            language_code=self.LANG,
        )
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=2,
            panels=_bubbles("b", "B"),
            language_code=self.LANG,
        )
        rows = db.get_chapter_panels(self.MANGA, self.CHAPTER)
        self.assertEqual(len(rows), 4)  # 2 bubbles per page, 2 pages

    def test_list_mangas(self):
        """list_mangas returns mangas; order_by and order_desc work."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        entries = db.list_mangas()
        found = [e for e in entries if e.manga_title == self.MANGA]
        self.assertGreater(len(found), 0)
        self.assertTrue(hasattr(found[0], "id"))
        self.assertTrue(hasattr(found[0], "manga_title"))
        self.assertTrue(hasattr(found[0], "created_at"))
        self.assertTrue(hasattr(found[0], "updated_at"))

        entries_asc = db.list_mangas(order_by="created_at", order_desc=False)
        self.assertIsInstance(entries_asc, list)

    def test_list_mangas_pagination(self):
        """list_mangas respects limit and offset."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        all_entries = db.list_mangas()
        limited = db.list_mangas(limit=1)
        self.assertEqual(len(limited), 1)
        offset_entries = db.list_mangas(limit=1, offset=0)
        self.assertEqual(len(offset_entries), 1)

    def test_list_chapters(self):
        """list_chapters returns chapters for a manga."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        chapters = db.list_chapters(self.MANGA)
        found = [c for c in chapters if c.chapter_number == self.CHAPTER]
        self.assertGreater(len(found), 0)
        self.assertTrue(hasattr(found[0], "manga_title"))
        # single-provider setup: no provider_id stored on chapters
        self.assertTrue(hasattr(found[0], "id"))
        self.assertTrue(hasattr(found[0], "chapter_number"))
        self.assertTrue(hasattr(found[0], "created_at"))
        self.assertTrue(hasattr(found[0], "updated_at"))
        self.assertEqual(found[0].manga_title, self.MANGA)
        # provider removed

        self.assertGreater(len(chapters), 0)

    def test_list_chapters_pagination(self):
        """list_chapters respects limit and offset."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        all_chapters = db.list_chapters(self.MANGA)
        limited = db.list_chapters(self.MANGA, limit=1)
        self.assertEqual(len(limited), 1)
        offset_chapters = db.list_chapters(self.MANGA, limit=1, offset=0)
        self.assertEqual(len(offset_chapters), 1)

    def test_get_panels_pagination(self):
        """get_panels respects limit and offset."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("a", "A"),
            language_code=self.LANG,
        )
        all_rows = db.get_panels(manga_title=self.MANGA, chapter_number=self.CHAPTER)
        self.assertEqual(len(all_rows), 2)
        limited = db.get_panels(manga_title=self.MANGA, chapter_number=self.CHAPTER, limit=1)
        self.assertEqual(len(limited), 1)
        offset_rows = db.get_panels(manga_title=self.MANGA, chapter_number=self.CHAPTER, limit=1, offset=1)
        self.assertEqual(len(offset_rows), 1)
        self.assertEqual(offset_rows[0].translated_text, "A 2")

    def test_delete_page_panels(self):
        """delete_page_panels removes panels for one page."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        db.delete_page_panels(self.MANGA, self.CHAPTER, self.PAGE)
        rows = db.get_panels(manga_title=self.MANGA, chapter_number=self.CHAPTER, page_number=self.PAGE)
        self.assertEqual(len(rows), 0)

    def test_delete_chapter_panels(self):
        """delete_chapter_panels removes chapter and all its panels."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("x", "X"),
            language_code=self.LANG,
        )
        db.delete_chapter_panels(self.MANGA, self.CHAPTER)
        rows = db.get_chapter_panels(self.MANGA, self.CHAPTER)
        self.assertEqual(len(rows), 0)

    def test_save_requires_no_provider_id(self):
        """save_panels_translation does not require a provider_id."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("x", "X"),
            language_code=self.LANG,
        )

    def test_provider_id_validation_delete_page(self):
        """delete_page_panels deletes without a provider_id."""
        # No exception expected; function no longer takes a provider.
        db.delete_page_panels(self.MANGA, self.CHAPTER, self.PAGE)

    def test_provider_id_validation_delete_chapter(self):
        """delete_chapter_panels deletes without a provider_id."""
        db.delete_chapter_panels(self.MANGA, self.CHAPTER)

    def test_provider_mangadex(self):
        """Basic provider-less flow works."""
        db.save_panels_translation(
            manga_title=self.MANGA,
            chapter_number=self.CHAPTER,
            page_number=self.PAGE,
            panels=_bubbles("m", "M"),
            language_code=self.LANG,
        )
        rows = db.get_panels(manga_title=self.MANGA, chapter_number=self.CHAPTER)
        self.assertEqual(len(rows), 2)
        db.delete_chapter_panels(self.MANGA, self.CHAPTER)


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
