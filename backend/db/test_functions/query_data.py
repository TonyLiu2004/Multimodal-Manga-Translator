"""
Run get/list operations and print results (no writes).
Run from backend:  python -m db.test_functions.query_data ***
Requires DATABASE_URL in backend/.env.
"""

import db


def run():
    print("=== list_entries (default: newest first) ===\n")
    entries = db.list_entries()
    for e in entries:
        print(f"  {e['provider_id']} | {e['manga_title']} ch.{e['chapter_number']} | {e['last_updated']}")
    print(f"\n  Total: {len(entries)} chapter(s)\n")

    print("=== list_entries (order_by manga_title, asc) ===\n")
    entries_by_title = db.list_entries(order_by="manga_title", order_desc=False)
    for e in entries_by_title:
        print(f"  {e['provider_id']} | {e['manga_title']} ch.{e['chapter_number']}")
    print()

    if not entries:
        print("No data. Run:  python -m db.seed_data")
        return

    # Use first entry for get_segments / get_chapter_segments examples
    first = entries[0]
    provider_id = first["provider_id"]
    manga_title = first["manga_title"]
    chapter_number = first["chapter_number"]

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
