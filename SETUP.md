# Multimodal Manga Translator – Setup Guide

Follow these steps to run the project on your computer (Windows, macOS, or Linux).

---

## 1. Prerequisites

- **Python 3.10 or 3.11** (3.12 may work; avoid 3.13 until dependencies support it)
- **Git** (already have the repo)
- **~10–15 GB free disk** for models and dependencies (Qwen is ~15GB; others are smaller)
- **8 GB+ RAM** recommended (16 GB preferred for the translation model)

---

## 2. Create a virtual environment

Open a terminal in the project root (`Multimodal-Manga-Translator`):

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your prompt.

---

## 3. Install PyTorch first (recommended for Windows)

The project uses **PyTorch**. On Windows, the CPU-only build often installs more reliably from PyTorch’s index:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you have an **NVIDIA GPU** and want to use it:

```powershell
# CUDA 12.x (check your driver)
pip install torch torchvision
```

---

## 4. Install backend dependencies

From the **project root** (with the venv active):

```powershell
pip install -r backend/requirements.txt
```

If you already installed PyTorch in step 3 and pip tries to reinstall it from PyPI and fails, run the PyTorch command again, then:

```powershell
pip install -r backend/requirements.txt --no-deps
pip install -r backend/requirements.txt
```

(First pass skips reinstalling deps of existing packages; second pass fills in anything missing.)

---

## 5. Run from the backend directory

All commands below assume the virtual environment is activated.

```powershell
cd backend
python main.py
```

**First run will:**

1. **Download models** (one-time, can take a while):
   - **Bubble detector** (Kiuyha) → `backend/models/kiuyha.pt`
   - **Japanese OCR** (Kha-white) → `backend/models/Kha-white/`
   - **GLM OCR** (general) → `backend/models/GlmOcr/`
   - **Qwen translation** (~15 GB) → `backend/models/Qwen/`
2. **Download font** if missing: Noto Sans CJK → `backend/fonts/NotoSansCJK.ttc`

The **TencentHY** (Manhua) translator is optional; if you don’t have it under `backend/models/TencentHY/`, the app will start and only skip that model.

---

## 6. Provide a test image

`main()` in `main.py` uses a test image path. Either:

**Option A – Use the default path**

- Create a folder named `test_images` in the **project root** (next to `backend/`).
- Add a manga page image named `test_2.png` (or change the path in `main()`).

**Option B – Point to your own file**

- Edit `main()` at the bottom of `backend/main.py`:

```python
def main():
    img_path = ROOT / "path" / "to" / "your" / "manga_page.png"
    img = process_image(img_path, "japanese")
    img.show()
```

Then run again:

```powershell
cd backend
python main.py
```

A window should open with the manga page with bubbles detected and text translated to English.

---

## 7. Database (PostgreSQL)

Translated chapter data (text + coordinates only, no images) is stored in a **PostgreSQL** server.

1. **Install and run PostgreSQL** (locally or use a hosted service like Neon, Supabase, or Railway).

2. **Create a database** (e.g. `manga_translations`) and set the connection URL:

   ```text
   postgresql://USER:PASSWORD@HOST:5432/DATABASE
   ```

3. **Set the URL** when running the app:
   - **Environment variable (recommended):** set `DATABASE_URL` to that URL (e.g. in a `.env` file or your shell).
   - **In code:** pass `db_url="postgresql://..."` to `process_chapter(..., db_url=...)` or to `manga_db.init_db(db_url)` etc.

4. **Install the driver** (included in backend requirements):

   ```powershell
   pip install psycopg2-binary
   ```

5. **First use:** tables are created automatically when you call `process_chapter(...)` or `manga_db.init_db()`.

Example:

```python
import os
os.environ["DATABASE_URL"] = "postgresql://user:password@localhost:5432/manga_translations"
from main import process_chapter
process_chapter("My Manga", 1, ["page1.png", "page2.png"], db_url=os.environ["DATABASE_URL"])
```

---

## 8. Troubleshooting

| Issue | What to do |
|-------|------------|
| `ModuleNotFoundError: No module named 'services'` | Run from inside the `backend` folder: `cd backend` then `python main.py`. |
| `FileNotFoundError: ... test_2.png` | Create `test_images` in the project root and add an image, or change `img_path` in `main()` to your file. |
| PyTorch install fails on Windows | Use step 3: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` then retry step 4. |
| Out of memory when loading Qwen | Close other apps; the 7B model needs several GB RAM. You can try a smaller model by changing the Qwen service (advanced). |
| Font not found | Font should auto-download. If it fails, run from project root: `python backend/setup.py` to download the CJK font. |
| Slow first run | First run downloads all models; later runs use the cached files in `backend/models/`. |
| `No database URL` / DB errors | Set `DATABASE_URL` (PostgreSQL URL) or pass `db_url=...`. Ensure PostgreSQL is running and the database exists. |

---

## 9. Optional: run only font setup

To only download the CJK font (e.g. after cloning on a new machine):

```powershell
cd backend
python -c "from helpers import setup_fonts; setup_fonts()"
```

Or from project root:

```powershell
python backend/setup.py
```

---

## Quick reference

```powershell
# From project root, one-time
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

# Every time you want to run
cd backend
python main.py
```

Put manga images in `test_images/` (e.g. `test_2.png`) or set `img_path` in `main()` to your file.
