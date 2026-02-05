from services.ocr_service import OCR_Service
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
GLMOCR_MODEL_DIR = BASE_DIR / "backend" / "models" / "GlmOcr"
ocr = OCR_Service(GLMOCR_MODEL_DIR)

test_dir = BASE_DIR / "test_images"
for i in range(1, 10):
    try:
        image_url = test_dir / f"test_{i}.png"
        text = ocr.runOCR(image_url)
        print(f"=============={i}==============")
        print(text)
    except:
        print(f"failed on {i}")
        break