# Multimodal-Manga-Translator

A cross-platform manga reader that dynamically pulls content from provider APIs and uses an AI-powered backend to overlay real-time translations directly onto the source media.

The system leverages a FastAPI backend to perform automated bubble-text detection, OCR, and machine translation, delivering a structured JSON response to the frontend for low-latency and non-destructive rendering over the original images.

NOTES:

- Dockerfile.nvidia on the backend has not been tested

## Deployed

Deployed Frontend URL: https://manglify-d6ebc.web.app/
Deployed Backend URL: https://tonyliu404-manglify-backend.hf.space/

- Huggingface Page: https://huggingface.co/spaces/tonyliu404/Manglify_Backend

### Hugging Face backend: why the DB is “missing”

GitHub Actions only pushes the **`backend/`** folder to the Space. **`backend/.env` is gitignored**, so **`DATABASE_URL` is never deployed** unless you add it yourself.

On the Space: **Settings → Variables and secrets** (repository secrets), create:

- **`DATABASE_URL`** — PostgreSQL URL for your hosted DB (for example Supabase). Many hosts need TLS; try appending **`?sslmode=require`** if connections fail. Use **`postgresql+psycopg2://user:pass@host:5432/db`** if your URL omits the driver.

For authenticated routes (reading lists, profile):

- **`SUPABASE_JWT_SECRET`**, **or** **`SUPABASE_URL`** plus **`SUPABASE_ANON_KEY`** (same behavior as local `auth_supabase.py`).

Restart or redeploy the Space after saving secrets. Verify DB connectivity with **`GET /health/db`** on the deployed API. Run **`alembic upgrade head`** from your machine with that same **`DATABASE_URL`** so schema matches production.

## Prerequisites

- Windows Users: Docker Desktop with the WSL 2 backend enabled.

- AMD GPU Users: Ensure you have the latest AMD Adrenalin Drivers installed on the host.

## Getting Started

1. Prepare Models & Fonts
   Models and fonts are automatically downloaded into local directories (these sync to the container via volumes):

- backend/models

- backend/fonts/NotoSansCJK.ttc

2. Configuration
   The system is configured via docker-compose.yml.

- For AMD GPU: Ensure dockerfile: backend/Dockerfile.amd is set.

- For CPU only: Change to dockerfile: backend/Dockerfile.cpu.

- For NVIDIA GPU: Not implemented yet

## Build and Start

Run these commands in the root directory:

### Build the image

```
docker compose build
```

### Start the application

```
docker compose up
```

Once the logs show Uvicorn running, access the interactive API documentation at:
http://localhost:8000/docs

Server runs at:
http://localhost:8000/

Frontend runs at:
http://localhost:8081/

## Local database setup

Use the built-in Postgres container and helper scripts so everyone can test DB code easily.

1. **Prerequisites**
   - Docker Desktop running
   - Python 3.x
   - Backend dependencies installed:
     ```
     cd backend
     pip install -r requirements.txt
     ```

2. **Configure backend database URL**

   Create `backend/.env` (or edit it) with:

   ```
   DATABASE_URL=postgresql+psycopg2://mmt_user:mmt_password@db:5432/mmt_test
   ```

3. **Start services (backend, frontend, DB)**

   From the repo root:

   ```
   docker compose up -d
   ```

4. **Create tables and seed sample data**

   From the `backend` folder:

   ```
   cd backend
   python -m db.test_functions.create_data
   ```

   or any other file in test_functions

5. **Verify via API (optional)**

   Open `http://localhost:8000/docs` and call:
   - `GET /mangas` – you should see entries such as `local / One Piece` and `mangadex / Naruto`

## Testing hosted frontend

Frontend is deployed at https://manglify-d6ebc.web.app/

- Browsers block public websites from reaching local addresses, so we use ngrok to tunnel our backend

In Terminal 1:

```
python main.py
```

In Terminal 2:

```
ngrok http 8000
```

Take the url provided by ngrok and paste it into the BACKEND_URL variable in frontend/app/config.ts

### Deployment and redeployment

In frontend folder run:

```
npx expo export --platform web # builds frontend website and puts it in the dist folder

firebase deploy --only hosting # deploys from dist folder

```

---

## Troubleshooting & Maintenance

<details>
<summary><b>Click to expand: Advanced Docker Commands</b></summary>

#### Completely rebuild images from scratch without using cached layers

```
docker-compose build --no-cache
```

#### Stop services and remove containers, networks, and old 'orphan' services

```
docker-compose down --remove-orphans
```

</details>
