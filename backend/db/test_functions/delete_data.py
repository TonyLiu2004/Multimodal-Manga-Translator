"""
Delete sample/seed data from the database.
Run from backend:  python -m db.test_functions.delete_data ***
Requires DATABASE_URL in backend/.env.
"""

import db


def clear_seed_data():
    """Remove the manga chapters inserted by create_data.py."""
    to_delete = [
        ("One Piece", 1.0),
        ("One Piece", 2.0),
        ("Naruto", 1.0),
    ]
    for manga_title, chapter_number in to_delete:
        db.delete_chapter_panels(manga_title, chapter_number)
        print(f"Deleted {manga_title} ch.{chapter_number}")
    print("Seed data cleared.")


def clear_all():
    """Delete all panels, chapters, reading list, and umbrella manga."""
    db.delete_all_manga()
    print("Cleared all manga and related data.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        clear_all()
    else:
        clear_seed_data()
