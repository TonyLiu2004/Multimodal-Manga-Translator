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
        print(f"  {m['provider_id']} | {m['manga_title']}")
    print(f"\n  Total: {len(mangas)} manga(s)\n")

    if not mangas:
        print("No data. Run:  python -m db.test_functions.create_data")
        return

    first_manga = mangas[0]
    provider_id = first_manga["provider_id"]
    manga_title = first_manga["manga_title"]

    chapters = db.list_chapters(manga_title, provider_id=provider_id)
    if not chapters:
        print(f"No chapters for {manga_title}. Run create_data to add pages.")
        return
    chapter_number = chapters[0]["chapter_number"]

    print(f"=== list_chapters({manga_title!r}, provider_id={provider_id!r}) ===\n")
    for c in chapters[:5]:
        print(f"  ch.{c['chapter_number']} (id={c['id']})")
    if len(chapters) > 5:
        print(f"  ... and {len(chapters) - 5} more chapter(s)")
    print(f"\n  Total: {len(chapters)} chapter(s)\n")

    print(f"=== get_chapter_segments({provider_id!r}, {manga_title!r}, {chapter_number}) ===\n")
    segments = db.get_chapter_segments(provider_id, manga_title, chapter_number)
    for i, s in enumerate(segments[:5]):  # show first 5
        print(f"  [{i+1}] page {s['page_number']} seg {s['segment_index']}: {s['original_text'][:30]!r} → {s['translated_text'][:30]!r}")
    if len(segments) > 5:
        print(f"  ... and {len(segments) - 5} more segment(s)")
    print(f"\n  Total: {len(segments)} segment(s) in chapter\n")

    print(f"=== get_segments (provider_id={provider_id!r}, manga_title={manga_title!r}, chapter_number={chapter_number}, page_number=1) ===\n")
    page_segments = db.get_segments(
        provider_id=provider_id,
        manga_title=manga_title,
        chapter_number=chapter_number,
        page_number=1,
    )
    for s in page_segments:
        print(f"  seg {s['segment_index']}: {s['translated_text']!r}")
    print(f"\n  Total: {len(page_segments)} segment(s) on page 1\n")

    print("=== get_segments (no filters: all segments) ===\n")
    all_segments = db.get_segments()
    for s in all_segments:
        print(f"  {s['provider_id']} | {s['manga_title']} ch.{s['chapter_number']} | {s['page_number']} | {s['segment_index']}: {s['translated_text']!r}")
    print(f"\n  Total: {len(all_segments)} segment(s)\n")

if __name__ == "__main__":
    run()
