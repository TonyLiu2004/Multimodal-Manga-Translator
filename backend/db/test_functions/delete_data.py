"""
Delete sample/seed data from the database.
Run from backend:  python -m db.test_functions.delete_data ***
Requires DATABASE_URL in backend/.env.
"""

import db


def clear_seed_data():
    """Remove the manga chapters inserted by seed_data.py."""
    # Match the data from seed_data.py
    to_delete = [
        (db.PROVIDER_LOCAL, "One Piece", 1.0),
        (db.PROVIDER_LOCAL, "One Piece", 2.0),
        (db.PROVIDER_MANGADEX, "Naruto", 1.0),
    ]
    for provider_id, manga_title, chapter_number in to_delete:
        db.delete_chapter_segments(provider_id, manga_title, chapter_number)
        print(f"Deleted {provider_id} | {manga_title} ch.{chapter_number}")
    print("Seed data cleared.")


def clear_all():
    """Delete all chapters, then all manga."""
    entries = db.list_entries()
    for e in entries:
        db.delete_chapter_segments(e["provider_id"], e["manga_title"], e["chapter_number"])
        print(f"Deleted {e['provider_id']} | {e['manga_title']} ch.{e['chapter_number']}")
    print(f"Cleared {len(entries)} chapter(s).")
    db.delete_all_manga()
    print("Cleared all manga.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        clear_all()
    else:
        clear_seed_data()
