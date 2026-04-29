"""
Run get/list operations and print results (no writes).
Run from backend:  python -m db.test_functions.query_data ***
Requires DATABASE_URL in backend/.env.
"""

import db


def run():
    print("=== list_mangas (default: newest first) ===\n")
    mangas = db.list_mangas()
    for m in mangas:
        print(f"  {m.manga_title}")
    print(f"\n  Total: {len(mangas)} manga(s)\n")

    if not mangas:
        print("No data. Run:  python -m db.test_functions.create_data")
        return

    first_manga = mangas[0]
    manga_title = first_manga.manga_title

    chapters = db.list_chapters(manga_title)
    if not chapters:
        print(f"No chapters for {manga_title}. Run create_data to add pages.")
        return
    chapter_number = chapters[0].chapter_number

    print(f"=== list_chapters({manga_title!r}) ===\n")
    for c in chapters[:5]:
        print(f"  ch.{c.chapter_number} (id={c.id})")
    if len(chapters) > 5:
        print(f"  ... and {len(chapters) - 5} more chapter(s)")
    print(f"\n  Total: {len(chapters)} chapter(s)\n")

    print(f"=== get_chapter_panels({manga_title!r}, {chapter_number}) ===\n")
    panels = db.get_chapter_panels(manga_title, chapter_number)
    for i, s in enumerate(panels[:5]):  # show first 5
        print(
            f"  [{i+1}] page {s.page_number} bubble {s.bubble_index}: "
            f"{s.original_text[:30]!r} → {s.translated_text[:30]!r} "
            f"(url={s.panel_url!r})"
        )
    if len(panels) > 5:
        print(f"  ... and {len(panels) - 5} more panel(s)")
    print(f"\n  Total: {len(panels)} panel(s) in chapter\n")

    if panels and panels[0].panel_url:
        url = panels[0].panel_url
        print(f"=== get_panel_by_panel_url({url!r}) ===\n")
        p = db.get_panel_by_panel_url(url)
        if p:
            print(
                f"  found: {p.manga_title} ch.{p.chapter_number} "
                f"page={p.page_number} bubble={p.bubble_index} url={p.panel_url!r}"
            )
        else:
            print("  not found")
        print()

    print(f"=== get_panels (manga_title={manga_title!r}, chapter_number={chapter_number}, page_number=1) ===\n")
    page_panels = db.get_panels(
        manga_title=manga_title,
        chapter_number=chapter_number,
        page_number=1,
    )
    for s in page_panels:
        print(f"  bubble {s.bubble_index}: {s.translated_text!r}")
    print(f"\n  Total: {len(page_panels)} panel(s) on page 1\n")

    print("=== get_panels (no filters: all panels) ===\n")
    all_panels = db.get_panels()
    for s in all_panels:
        print(
            f"  {s.manga_title} ch.{s.chapter_number} | {s.page_number} | "
            f"{s.bubble_index}: {s.translated_text!r} (url={s.panel_url!r})"
        )
    print(f"\n  Total: {len(all_panels)} panel(s)\n")

if __name__ == "__main__":
    run()
