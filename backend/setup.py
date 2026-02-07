from transformers import AutoModelForImageTextToText, AutoProcessor
from pathlib import Path
import requests
import zipfile
import io
import os
BASE_DIR = Path(__file__).resolve().parent.parent

#download OCR model locally
def load_OCR_model():
    GLMOCR_MODEL_DIR = BASE_DIR / "backend" / "models" / "GlmOcr"
    MODEL_PATH = "zai-org/GLM-OCR"
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH)
    model.save_pretrained(GLMOCR_MODEL_DIR / "model")
    processor = AutoProcessor.from_pretrained( MODEL_PATH)
    processor.save_pretrained(GLMOCR_MODEL_DIR / "processor")


def setup_fonts():
    url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/SuperOTC/NotoSansCJK.ttc.zip"
    extract_to = BASE_DIR
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
    load_OCR_model()
    setup_fonts()