# Read API for Frontend

The backend exposes **read-only** HTTP endpoints so the frontend can list chapters and get segments.

## Setup

```bash
cd backend
pip install fastapi "uvicorn[standard]"
```

Or install all deps: `pip install -r requirements.txt`

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
