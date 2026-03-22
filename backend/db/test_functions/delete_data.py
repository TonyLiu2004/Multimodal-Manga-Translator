"""
Delete sample/seed data from the database.
Run from backend:  python -m db.test_functions.delete_data ***
Requires DATABASE_URL in backend/.env.
"""

import db


def clear_seed_data():
    """Remove the manga chapters inserted by create_data.py."""
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
    """Delete all segments, pages, chapters, reading list, manga_source, and umbrella manga."""
    db.delete_all_manga()
    print("Cleared all manga and related data.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        clear_all()
    else:
        clear_seed_data()
