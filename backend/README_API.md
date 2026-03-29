# Read API for Frontend

The backend exposes **read-only** HTTP endpoints so the frontend can list chapters and get segments.

## Setup

```bash
cd backend
pip install fastapi "uvicorn[standard]"
```

Or install all deps: `pip install -r requirements.txt`

### Environment (reading lists / Supabase auth)

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL for app data |
| **Either** `SUPABASE_JWT_SECRET` **or** remote verification below | Verify the user’s `access_token` |

**If your project no longer exposes a legacy JWT secret** (dashboard says to use signing keys / publishable keys), skip `SUPABASE_JWT_SECRET` and set:

| Variable | Where to copy |
|----------|----------------|
| `SUPABASE_URL` | Same as frontend `EXPO_PUBLIC_SUPABASE_URL` (e.g. `https://xxxx.supabase.co`) |
| `SUPABASE_ANON_KEY` | Same as frontend `EXPO_PUBLIC_SUPABASE_ANON_KEY` (anon / publishable key) |

The API then calls `GET {SUPABASE_URL}/auth/v1/user` with the client’s Bearer token to validate it (no local JWT secret needed).

If `SUPABASE_JWT_SECRET` is set, it is used first (offline verification; faster).

See `backend/.env.example`.

uvicorn is ASGI server, in production will need something like to use to start

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app

```

## Run the server

```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

- **API base:** http://localhost:8000
- **Docs (Swagger):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Endpoints

| Method | Path                 | Description                                                                                        |
| ------ | -------------------- | -------------------------------------------------------------------------------------------------- |
| GET    | `/entries`           | List all chapters. Query: `order_by`, `order_desc`                                                 |
| GET    | `/segments`          | Get segments. Query: `provider_id`, `manga_title`, `chapter_number`, `page_number` (all optional)  |
| GET    | `/chapters/segments` | Get all segments for one chapter. Query: `provider_id`, `manga_title`, `chapter_number` (required) |
| GET    | `/health`            | Health check                                                                                       |

## Example requests (frontend or curl)

```bash
# List chapters (newest first)
curl "http://localhost:8000/entries"

# List chapters by manga title A–Z
curl "http://localhost:8000/entries?order_by=manga_title&order_desc=false"

# Get all segments for a chapter
curl "http://localhost:8000/chapters/segments?provider_id=local&manga_title=One%20Piece&chapter_number=1"

# Get segments for one page
curl "http://localhost:8000/segments?provider_id=local&manga_title=One%20Piece&chapter_number=1&page_number=1"
```

## CORS

CORS is enabled for all origins so your frontend (e.g. Expo) can call the API. Restrict `allow_origins` in production if needed.
