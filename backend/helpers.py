from pathlib import Path
import requests
import zipfile
import io
import os

def get_project_root() -> Path:
    """Finds the root directory by looking for the .git folder."""
    curr = Path(__file__).resolve()
    for parent in curr.parents:
        if (parent / ".git").exists():
            return parent
    return curr.parent  # Fallback to current folder

ROOT = get_project_root()

def setup_fonts():
    url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/SuperOTC/NotoSansCJK.ttc.zip"
    extract_to = Path(os.getenv("FONT_PATH", ROOT / "backend" / "fonts"))
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if Path(extract_to / "NotoSansCJK.ttc").exists():
        print(f"Font file exists at {extract_to}")
        return

    zip_path = os.path.join(extract_to, "NotoSansCJK.ttc.zip")

    print("Downloading Noto Sans CJK SuperOTC...")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        os.remove(zip_path)
        print(f"Done! Font extracted to {extract_to}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")