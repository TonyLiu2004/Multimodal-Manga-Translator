"""
Alembic env.py - uses db models and DATABASE_URL.
Run from backend: alembic revision --autogenerate -m "description"
                   alembic upgrade head
"""
import os
from pathlib import Path

from alembic import context
from alembic.config import Config
from sqlalchemy import create_engine
from sqlmodel import SQLModel

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Import all models so they're registered with SQLModel.metadata
from db.models import (
    Manga,
    Chapters,
    Panels,
    Users,
    ReadingListCollection,
    ReadingListItem,
)

# context.config exists at runtime; Alembic's context stub omits it for static analysis.
config: Config = getattr(context, "config")
target_metadata = SQLModel.metadata


def get_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("Set DATABASE_URL in .env")
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generate SQL only)."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connect to DB)."""
    url = get_url()
    engine = create_engine(url, pool_size=5, max_overflow=10)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if getattr(context, "is_offline_mode")():
    run_migrations_offline()
else:
    run_migrations_online()
