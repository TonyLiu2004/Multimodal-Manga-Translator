from transformers import AutoModelForImageTextToText, AutoProcessor
from pathlib import Path
import requests
import zipfile
import io
import os
from huggingface_hub import hf_hub_download
from helpers import get_project_root
# BASE_DIR = Path(__file__).resolve()

ROOT = get_project_root()

#download OCR model locally
def load_OCR_model():
    GLMOCR_MODEL_DIR = ROOT / "backend" / "models" / "GlmOcr"
    MODEL_PATH = "zai-org/GLM-OCR"
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH)
    model.save_pretrained(GLMOCR_MODEL_DIR / "model")
    processor = AutoProcessor.from_pretrained( MODEL_PATH)
    processor.save_pretrained(GLMOCR_MODEL_DIR / "processor")

def load_bubble_detector_kiuyha():
    model_dir = ROOT / "backend" / "models"
    target_path = model_dir / "kiuyha.pt"

    if target_path.exists():
        print(f"Model already exists at {target_path}")
        return str(target_path)
    
    downloaded_path = hf_hub_download(
        repo_id="Kiuyha/Manga-Bubble-YOLO",
        filename="model.pt",
        local_dir=model_dir
    )

    final_path = Path(downloaded_path).rename(target_path)
    print(f"Downloaded Kiuyha bubble detector to: {final_path}")
    return str(final_path)

def load_bubble_detector_kitsumed():
    model_dir = ROOT / "backend" / "models"
    target_path = model_dir / "kitsumed.pt"

    if target_path.exists():
        print(f"Model already exists at {target_path}")
        return str(target_path)

    downloaded_path = hf_hub_download(
        repo_id="kitsumed/yolov8m_seg-speech-bubble",
        filename="model.pt",
        local_dir=model_dir
    )

    final_path = Path(downloaded_path).rename(target_path)
    print(f"Downloaded Kitsumed bubble detector to: {final_path}")
    return str(final_path)

def setup_fonts():
    url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/SuperOTC/NotoSansCJK.ttc.zip"
    extract_to = ROOT / "backend" / "fonts"
    if Path(extract_to / "NotoSansCJK.ttc").exists():
        print(f"Font file exists at {extract_to}")
        return

    zip_path = os.path.join(extract_to, "NotoSansCJK.ttc.zip")

    # 2. Download the file
    print("Downloading Noto Sans CJK SuperOTC (this may take a minute)...")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

        # 3. Extract the ZIP
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # 4. Cleanup the zip file to save space
        os.remove(zip_path)
        print(f"Done! Font extracted to {extract_to}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    # print(ROOT)
    load_OCR_model()
    setup_fonts()
    load_bubble_detector_kitsumed()