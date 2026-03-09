# Alembic Migrations

Alembic manages schema changes instead of `init_db` / `create_all` for production.

## Setup

```bash
cd backend
pip install alembic
```

Ensure `DATABASE_URL` is set in `.env`.

## Commands

| Command                                      | Description                               |
| -------------------------------------------- | ----------------------------------------- |
| `alembic revision --autogenerate -m "add X"` | Generate migration from model changes     |
| `alembic upgrade head`                       | Apply all pending migrations              |
| `alembic downgrade -1`                       | Undo last migration                       |
| `alembic stamp head`                         | Mark DB as up to date (no migrations run) |
| `alembic current`                            | Show current revision                     |

## When doing Schema changes

1. Edit `db/models.py`
2. Run `alembic revision --autogenerate -m "description"`
3. Review `alembic/versions/...py`, fix if needed
4. Run `alembic upgrade head`
